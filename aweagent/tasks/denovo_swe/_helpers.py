"""Pure helpers used by :class:`DeNovoSWEEvaluator`.

Module-level functions extracted from the evaluator class so the main file
stays focused on the eval orchestration.  None of these carry evaluator
state — they take everything they need as arguments.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shlex
import subprocess
from typing import TYPE_CHECKING

from aweagent.core.task.types import EvalResult
from aweagent.tasks.denovo_swe._scripts import REMOVE_TESTS_SCRIPT

if TYPE_CHECKING:
    from aweagent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)


# Path used inside the sandbox to stage the binary archive.  One eval
# session corresponds to one container, so a fixed name is safe.  We
# always clean it up after extraction.
BINARY_ARCHIVE_REMOTE = "/tmp/_awe_test_binaries.tar.gz"


# ── Output truncation ──────────────────────────────────────────────────


def head_tail_slice(text: str, head_bytes: int = 6000, tail_bytes: int = 4000) -> str:
    """Return a head+tail slice of ``text`` with a truncation marker.

    The legacy ``text[-1000:]`` slice repeatedly cost us diagnosability —
    pytest fatal errors (``Fatal Python error: Segmentation fault``),
    collection-time import failures, and ``ERROR: not found: …`` lines
    all appear at the **beginning** of the output, while a pure tail
    slice keeps only the post-error stack frames.

    Defaults raised to 6 KB head + 4 KB tail after observing that the
    previous 1.5 KB head was eaten by pytest's bootstrap traceback
    (~800 chars of ``_pytest/python.py``/``importlib`` frames) before
    reaching the user-relevant ``test_X.py:NN: in <module>`` line — so
    the actual import name that triggered the failure was truncated
    away.
    """
    if len(text) <= head_bytes + tail_bytes:
        return text
    return (
        text[:head_bytes]
        + f"\n\n... [truncated {len(text) - head_bytes - tail_bytes} chars] ...\n\n"
        + text[-tail_bytes:]
    )


# ── Multi-iteration aggregation ────────────────────────────────────────


def aggregate_iterations(
    instance_id: str,
    results: list[EvalResult],
    total_duration: float,
) -> EvalResult:
    """Combine per-iteration EvalResults into a single one.

    * ``score`` — arithmetic mean of per-iteration scores.
    * ``accepted`` — strict: only True when EVERY iteration accepts.
    * ``duration`` — total wall-clock of all iterations.
    * ``details`` — first iter's details (preserved for back-compat
      consumers reading ``details.output``), plus a compact
      ``iterations`` field listing per-iter ``score`` /
      ``accepted`` / ``error``.
    """
    if not results:
        return EvalResult(
            accepted=False, score=0.0,
            details={"error": "no_iterations"},
            duration=total_duration,
        )

    scores = [r.score or 0.0 for r in results]
    accepted_flags = [bool(r.accepted) for r in results]
    score_mean = sum(scores) / len(scores)
    # Population stdev — fine for N=3..5.
    score_var = sum((s - score_mean) ** 2 for s in scores) / len(scores)
    score_std = score_var ** 0.5

    # Preserve a representative iter's details so any field name
    # a downstream consumer expected continues to exist.  Prefer
    # the first NON-CRASH iter: results[0] may have
    # ``details = {"error": "iteration_crash:..."}`` from an early
    # runtime hiccup, and propagating that as the aggregated
    # ``details.error`` would falsely advertise the whole row as
    # crashed even when iters 1+ produced honest outcomes.  Fall
    # back to results[0] if every iter crashed.
    def _is_crash_only(r: EvalResult) -> bool:
        d = r.details or {}
        return (
            bool(d)
            and len(d) == 1
            and isinstance(d.get("error"), str)
            and d["error"].startswith("iteration_crash:")
        )

    base_iter = next(
        (r for r in results if not _is_crash_only(r)),
        results[0],
    )
    base = dict(base_iter.details or {})
    iter_summary = []
    for i, r in enumerate(results):
        iter_summary.append({
            "iter": i,
            "score": r.score,
            "accepted": r.accepted,
            "duration": r.duration,
            "error": (r.details or {}).get("error"),
        })
    base["iterations"] = iter_summary
    base["iter_count"] = len(results)
    base["score_mean"] = score_mean
    base["score_std"] = score_std
    base["all_accepted"] = all(accepted_flags)
    base["any_accepted"] = any(accepted_flags)

    logger.info(
        "Eval: aggregated %d iterations for %s → mean=%.3f std=%.3f "
        "all_accepted=%s",
        len(results), instance_id, score_mean, score_std,
        base["all_accepted"],
    )
    return EvalResult(
        accepted=base["all_accepted"],
        score=score_mean,
        details=base,
        duration=total_duration,
    )


# ── Image lifecycle ────────────────────────────────────────────────────


async def delete_docker_image(image: str) -> None:
    """Delete a docker image after use."""
    try:
        subprocess.run(
            ["docker", "rmi", "-f", image],
            capture_output=True, timeout=60,
        )
        logger.info("Deleted image: %s", image)
    except Exception as exc:
        logger.warning("Failed to delete image %s: %s", image, exc)


# ── Patch parsing ──────────────────────────────────────────────────────


def parse_patch_files(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch.

    Looks for ``+++ b/path/to/file`` lines and returns unique paths.
    """
    files = set()
    for m in re.finditer(r'^\+\+\+ b/(.+)$', patch, re.MULTILINE):
        path = m.group(1).strip()
        if path and path != "/dev/null":
            files.add(path)
    return sorted(files)


def files_added_by_test_patch(patch: str) -> list[str]:
    """Extract paths the patch creates as NEW files (additions only).

    Only returns paths where the diff is ``--- /dev/null`` immediately
    followed by ``+++ b/<path>``.  Modify-type hunks (where the ``---``
    side is a real path, e.g. for files preserved by ``clean.sh`` such
    as ``setup.py``, ``pyproject.toml``, ``.gitignore``) are intentionally
    NOT returned — pre-cleaning these paths would break ``test_patch``'s
    modify hunks against pre-existing files.

    Used by ``evaluate()`` to pre-remove agent-created files that sit at
    the same path as files ``test_patch`` is about to newly add, which
    would otherwise cause the ``test_patch`` hunks to be rejected.
    """
    added: list[str] = []
    seen: set[str] = set()
    lines = patch.splitlines()
    for i, line in enumerate(lines):
        if line.strip() != "--- /dev/null":
            continue
        if i + 1 >= len(lines):
            continue
        m = re.match(r'^\+\+\+ b/(.+)$', lines[i + 1])
        if not m:
            continue
        path = m.group(1).strip()
        if path and path != "/dev/null" and path not in seen:
            seen.add(path)
            added.append(path)
    return added


# ── Pytest grouping & collection ───────────────────────────────────────


def group_tests_by_file(test_ids: list[str]) -> dict[str, list[str]]:
    """Group test IDs by their source file.

    ``tests/test_a.py::TestX::test_1`` → ``{"tests/test_a.py": [...]}``
    """
    groups: dict[str, list[str]] = {}
    for tid in test_ids:
        filepath = tid.split("::")[0]
        groups.setdefault(filepath, []).append(tid)
    return groups


async def collect_test_ids(
    session: RuntimeSession,
    workdir: str,
    filepath: str,
    timeout: int = 120,
) -> set[str] | None:
    """Return the set of pytest node IDs collectible from ``filepath``.

    Used to pre-flight which ``passed_ptp`` entries are actually
    resolvable BEFORE the real test run.  This sidesteps pytest's
    "abort the whole batch on first not-found id" behaviour, which
    previously turned a 95/96 outcome into score 0.

    Returns:
        * ``set[str]`` of fully-qualified node IDs on success
          (possibly empty if pytest collected zero items).
        * ``None`` if the collection command itself failed
          (import error in the test file, syntax error, missing
          dependency).  Caller treats this as "collection skew
          unknown — fall through to legacy behaviour."

    We pass the path to pytest both as the discovery target and
    as ``--rootdir`` so node IDs come out in the same shape as
    ``passed_ptp`` (relative to workdir).
    """
    # ``-q`` so each line is one node id, no decoration.
    # ``--no-header`` cuts boilerplate.
    # ``-o addopts=`` clears any per-repo addopts that would
    # otherwise inject ``-x`` / ``--maxfail=1`` / etc. — those
    # would make pre-flight stop at first failure.
    cmd = (
        f"python -m pytest --collect-only -q --no-header "
        f"-o addopts= --rootdir=. {shlex.quote(filepath)}"
    )
    try:
        result = await session.execute(cmd, cwd=workdir, timeout=timeout)
    except Exception as exc:
        logger.warning(
            "pytest --collect-only crashed for %s: %s",
            filepath, exc,
        )
        return None

    # Collection-time errors emit ``exit_code != 0`` AND no test
    # IDs.  Pytest exits 5 when nothing collected, 2 on import
    # errors, 4 on usage errors.  Treat any non-zero AND empty
    # output as "we can't tell" → return None so caller falls
    # through to legacy behaviour.
    output = (result.stdout or "") + (result.stderr or "")
    # Parse each line: node IDs look like
    #   tests/test_x.py::test_foo
    #   tests/test_x.py::TestClass::test_bar
    #   tests/test_x.py::test_param[case1]
    # Skip headers, summary lines, blank lines.
    ids: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if "::" not in line:
            continue
        # Pytest summary lines like ``5 tests collected in 0.02s``
        # don't contain ``::``, so they get filtered out above.
        # But ``ERROR`` lines like ``ERROR tests/x.py - SyntaxError``
        # don't either.  Defense in depth:
        if line.startswith(("==", "ERROR", "WARNING", "INTERNALERROR",
                            "<frozen", "  ")):
            continue
        ids.add(line)

    if not ids and not result.success:
        # Pre-flight itself failed — caller falls through.
        return None
    return ids


# ── Test file cleanup ──────────────────────────────────────────────────


async def clean_all_test_files(
    session: RuntimeSession,
    workdir: str,
) -> None:
    """Remove test-related files before applying test_patch.

    Goal: ensure ``test_patch`` (gold tests) applies cleanly and that
    agent-authored scratch test files don't pollute the run.

    Deletes:
    - All directories named test/tests/Test/Tests/testsuite/testing/
      test_suite (recursive, case-insensitive — mirrors clean.sh).
    - Test-pattern files (``test_*.py``, ``*_test.py``, ``*_tests.py``,
      ``conftest.py``) at the WORKSPACE ROOT only (``-maxdepth 1``).
      Deletion was previously recursive, which nuked legitimate package
      modules whose name happens to start with ``test_`` (e.g.
      ``src/flask_login/test_client.py`` — flask-login's public
      TestClient helper).  These modules ship in the agent's patch,
      must survive, and are never going to conflict with ``test_patch``
      (which only adds files under ``tests/`` directories — already
      wiped by the dir-name pass above).
    - Agent verification scratch (``verify_*.py``, ``check_*.py``):
      workspace root only.  Real packages never ship a top-level
      ``verify_*.py`` / ``check_*.py``; inside subpackages
      (e.g. ``pep8/checkers/check_E501.py``) these names are legit.
    """
    # 1) Test directories wholesale (case-insensitive).  Mirrors
    #    clean.sh exactly so any leftover ``tests/`` from upstream
    #    is gone before ``test_patch`` re-creates them.
    await session.execute(
        "find . -type d | while read -r d; do "
        "  dn=$(basename \"$d\" | tr '[:upper:]' '[:lower:]'); "
        "  case \"$dn\" in "
        "    test|tests|testsuite|testsuites|testing|test_suite) rm -rf \"$d\" ;; "
        "  esac; "
        "done 2>/dev/null || true",
        cwd=workdir,
    )

    # 2) Test-pattern files at workspace root only — see docstring.
    await session.execute(
        "find . -maxdepth 1 -type f \\( "
        "-iname 'test_*.py' -o -iname '*_test.py' -o -iname '*_tests.py' "
        "-o -iname 'conftest.py' "
        "-o -iname 'verify_*.py' -o -iname 'verify_implementation.py' "
        "-o -iname 'check_*.py' "
        "\\) -delete 2>/dev/null || true",
        cwd=workdir,
    )

    # 3) Caches that might interfere with collection.
    await session.execute(
        "find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true "
        "&& find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true",
        cwd=workdir,
    )

    logger.info("Cleaned all test files from %s", workdir)


async def remove_failed_tests(
    session: RuntimeSession,
    workdir: str,
    failed_ptp: list[str],
) -> None:
    """Physically remove failed_ptp test functions/methods from test files.

    This prevents known-broken tests from crashing pytest collection
    (e.g. import errors, missing classes) which would take down all
    tests in the same file.
    """
    # Group failed test IDs by file: {filepath: [[parts], ...]}
    file_parts: dict[str, list[list[str]]] = {}
    for tid in failed_ptp:
        parts = tid.split("::")
        if len(parts) < 2:
            continue
        filepath = parts[0]
        remainder = parts[1:]
        file_parts.setdefault(filepath, []).append(remainder)

    if not file_parts:
        return

    # Upload the removal script
    await session.upload_file(
        "/tmp/_awe_remove_tests.py",
        REMOVE_TESTS_SCRIPT.encode(),
    )

    # Upload config
    config_data = json.dumps({"failed_tests": file_parts})
    await session.upload_file(
        "/tmp/_awe_remove_tests_config.json",
        config_data.encode(),
    )

    # Execute
    result = await session.execute(
        "python /tmp/_awe_remove_tests.py /tmp/_awe_remove_tests_config.json",
        cwd=workdir, timeout=60,
    )
    if result.success:
        logger.info(
            "Removed failed_ptp tests: %s", result.stdout.strip(),
        )
    else:
        logger.warning(
            "Failed to remove failed_ptp tests: %s", result.stderr[:500],
        )


# ── Binary fixture transport ───────────────────────────────────────────


async def apply_binary_archive(
    session: RuntimeSession,
    workdir: str,
    archive_b64: str,
    expected_files: list[str],
    instance_id: str,
) -> str | None:
    """Decode + upload + extract the binary fixture archive.

    Returns ``None`` on success or a short error code on failure.
    """
    if not archive_b64:
        return None

    try:
        archive_bytes = base64.b64decode(archive_b64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        logger.error(
            "Eval: invalid base64 in test_binary_archive for %s: %s",
            instance_id, exc,
        )
        return "invalid_base64"

    if not archive_bytes:
        return None

    logger.info(
        "Eval: uploading binary fixture archive for %s "
        "(%d bytes raw, %d files expected)",
        instance_id, len(archive_bytes), len(expected_files),
    )
    try:
        await session.upload_file(
            BINARY_ARCHIVE_REMOTE, archive_bytes,
        )
    except Exception as exc:
        logger.error(
            "Eval: upload of binary archive failed for %s: %s",
            instance_id, exc,
        )
        return f"upload_failed:{type(exc).__name__}"

    # Extract straight into the working tree.  ``--no-same-owner``
    # avoids chown errors when extracting as root in containers
    # where the source uid no longer exists.  ``--overwrite`` so
    # any placeholder created by an earlier test-patch step is
    # replaced with the real binary content.
    # ``--overwrite`` is enough — adding ``--overwrite-dir``
    # caused ``tar: '--overwrite-dir' cannot be used with
    # '--overwrite'`` in the runtime's tar (a minimal build that
    # treats them as mutually exclusive).  Cost of the verified
    # regression in the 20260512_201031 run: 88 instances flipped
    # from honest scores to score=0 ``binary_archive_apply_failed``.
    extract = await session.execute(
        f"tar --no-same-owner --overwrite -xzf "
        f"{shlex.quote(BINARY_ARCHIVE_REMOTE)} "
        f"-C {shlex.quote(workdir)}",
        cwd=workdir, timeout=300,
    )
    # Always try to remove the staging file, success or not.
    await session.execute(
        f"rm -f {shlex.quote(BINARY_ARCHIVE_REMOTE)}",
        cwd=workdir, timeout=10,
    )
    if not extract.success:
        logger.error(
            "Eval: tar extraction failed for %s (exit=%d): %s",
            instance_id, extract.exit_code, extract.stderr[:500],
        )
        return f"tar_extract_failed:{extract.exit_code}"

    # Best-effort sanity check that the expected files exist now.
    if expected_files:
        sample = expected_files[: min(3, len(expected_files))]
        checks = " && ".join(
            f"test -e {shlex.quote(f)}" for f in sample
        )
        verify = await session.execute(checks, cwd=workdir, timeout=20)
        if not verify.success:
            logger.warning(
                "Eval: binary archive extracted but expected files "
                "missing for %s (sample=%s)",
                instance_id, sample,
            )

    logger.info(
        "Eval: binary fixtures extracted for %s",
        instance_id,
    )
    return None
