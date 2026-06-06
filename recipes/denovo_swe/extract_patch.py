"""doc2repo_extract_patch — Extract unit-test patches from sandbox images.

Reads an input JSONL file, launches a sandbox for each instance, finds
all unit-test files, extracts their git diff (patch), and writes the
patch info into a new JSONL file that serves as input for DeNovoSWE.

Each output record carries:

* ``test_patch``                  — text-only unified diff that
                                    ``git apply`` can replay.
* ``test_files``                  — list of every file path covered
                                    (text + binary).
* ``test_binary_archive_b64``     — base64 tar.gz of all binary
                                    fixtures (``""`` when none). Unified
                                    diffs cannot carry binary content,
                                    so the evaluator uploads + extracts
                                    this archive separately.
* ``test_binary_files``           — list of files actually packed in
                                    the archive.
* ``passed_ptp`` / ``failed_ptp`` — pytest node IDs, with the
                                    ``<dir>/TestX.py::<rest>`` malformation
                                    (Bug E) sanitized.
* ``pypi_name`` / ``pypi_name_candidates`` / ``import_names`` /
  ``package_setup_files`` / ``package_extract_error`` —
                                    backfilled by running the shared
                                    doc2repo package-name extractor
                                    against the parent_commit tree
                                    when the input row didn't already
                                    have these.  Drives the cheating
                                    warning logic for Bug B.

Usage:
    python recipes/denovo_swe/extract_patch.py \\
        --input data/doc2repo_ready_denovoswe.jsonl \\
        --output data/denovoswe_with_patches.jsonl \\
        --config configs/tasks/denovoswe.yaml \\
        --max-concurrent 20

    # Dry run — list instances without launching sandboxes
    python recipes/denovo_swe/extract_patch.py \\
        --input data.jsonl --output out.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import shlex
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

# ── Binary-archive caps ───────────────────────────────────────────────
# A single instance's binary fixtures must fit comfortably in the JSONL
# row.  The base64 encoding inflates by ~4/3, so a 100 MB raw cap maps
# to ~135 MB on the wire.  Instances exceeding the cap are recorded
# without binary content and flagged via ``extract_error``.
_MAX_BINARY_ARCHIVE_BYTES = 100 * 1024 * 1024  # 100 MB raw

# Tarball staging path inside the sandbox.  One container == one
# instance, so a fixed name is safe; we never reuse a container.
_BINARY_TAR_REMOTE = "/tmp/_awe_extract_binaries.tar.gz"

# Bug E sanitization (``<dir>/TestX.py::<rest>`` → ``<dir>.py::TestX::<rest>``)
# lives in :mod:`awe_agent.core.eval.utils` so the evaluator can apply the
# same rewrite as a safety net.
from awe_agent.core.eval.utils import sanitize_test_ids as _sanitize_test_ids  # noqa: E402

# clean.sh deletion predictor — used by the Bug B whitelist filter so we
# only ship files clean.sh would actually delete (sibling helper dirs
# preserve their non-code data files, which don't need to round-trip).
from awe_agent.tasks.denovo_swe.clean_predict import will_clean_delete  # noqa: E402

# Package-name extractor (pypi_name / import_names / package_setup_files).
# Bundled with the denovo_swe task so this recipe stays self-contained.
from awe_agent.tasks.denovo_swe.package_extractor import (  # noqa: E402
    get_extractor_script,
)

# ── Bug A / Bug B caps ────────────────────────────────────────────────
# Per-instance limits on how much we drag into test_files via the
# ancestor walker (Bug A) and the sibling whitelist (Bug B).  These
# are *soft* warnings — we log + truncate rather than fail, because
# even a partial test_patch is more useful than none.
_MAX_TEST_FILES = 5000
_MAX_ANCESTOR_FILES_PER_SCAN = 2000

# ── Bug B sibling-helper whitelist ────────────────────────────────────
# Conservative list of directory names at the **repo root** that are
# almost always test infrastructure rather than production code.  Each
# candidate is then filtered through ``will_clean_delete`` so we only
# ship the files clean.sh would actually nuke (.py / .json / .yml /
# .md / .rst / .ipynb …).  Files that survive clean.sh untouched
# (.csv / .tf / .heic / .png …) are skipped — they remain in the eval
# workspace on their own.
_BUG_B_SIBLING_DIRS = (
    "helpers", "helper",
    "fixtures", "fixture",
    "mocks", "mock",
    "fakes", "stubs",
    "testdata", "test_data", "test-data", "test_fixtures",
    "example", "examples",
    "samples", "sample",
    "demo", "demos",
)

def _extract_test_dirs_and_files(test_ids: list[str]) -> tuple[set[str], set[str]]:
    """Extract directories and direct file paths from pytest node IDs.

    Returns (directories, direct_files) where:
    - directories: unique parent dirs of test files
    - direct_files: the test file paths themselves

    E.g. ``testsuite/test_all.py::TestClass::test_method``
         -> dirs={"testsuite"}, files={"testsuite/test_all.py"}
    """
    dirs = set()
    files = set()
    for tid in test_ids:
        if not tid:
            continue
        filepath = tid.split("::")[0].strip()
        if not filepath:
            continue
        files.add(filepath)
        if "/" in filepath:
            dirs.add(filepath.rsplit("/", 1)[0])
        else:
            dirs.add(".")
    return dirs, files


# Patterns that identify test-related files.
# Used for directories whose name does NOT contain "test" —
# we only grab files matching these patterns from such dirs.
#
# For directories whose name DOES contain "test" (e.g. tests/, testsuite/,
# test_utils/), we grab ALL files since the whole dir is test infrastructure.
_TEST_FILE_PATTERNS = [
    # Test files
    "test_*.py",
    "*_test.py",
    "*_tests.py",
    # Pytest infrastructure
    "conftest.py",
    # Package markers (needed for imports within test packages)
    "__init__.py",
    # Test helpers / utilities
    "*helper*.py",
    "*helpers*.py",
    "*fixture*.py",
    "*fixtures*.py",
    "*factory*.py",
    "*factories*.py",
    "*mock*.py",
    "*mocks*.py",
    "*fake*.py",
    "*fakes*.py",
    "*stub*.py",
    "*stubs*.py",
    "*dummy*.py",
    "*util*.py",
    "*utils*.py",
    "*common*.py",
    "*base_test*.py",
    "*testcase*.py",
    "*testing*.py",
    # Test data / resources (common names)
    "*testdata*",
    "*test_data*",
    # Test runner / config
    "pytest.ini",
    "setup.cfg",
    "tox.ini",
    "pyproject.toml",
]

# Directory **basenames** that are recognised as test-only dirs.
#
# Mirrors ``clean.sh``'s wipe-dir set (line 87-94) — the substring fuzz
# we used previously matched too much (``testing/`` as production
# subpackage of ``nipy/nipype``, ``test_utils/`` mixing prod+test
# code, ``contest/`` legitimately not-test).  Exact basename match keeps
# the two systems in lock-step: a directory is treated as test
# infrastructure here iff clean.sh would also wipe it.
_TEST_DIR_NAMES = {
    "test", "tests",
    "testsuite", "testsuites",
    "testing", "test_suite",
}


def _is_test_directory(dirname: str) -> bool:
    """Check if a directory's basename is a precise test-dir name."""
    leaf = dirname.lower().split("/")[-1] if "/" in dirname else dirname.lower()
    return leaf in _TEST_DIR_NAMES


async def _discover_submodules(
    session, workdir: str,
) -> dict[str, str]:
    """Return ``{submodule_path: submodule_url}`` for every git
    submodule in ``workdir``.

    ``git diff`` between commits handles a submodule as a single
    "gitlink" entry (mode 160000) referencing a sha — it does NOT emit
    a per-file diff for the submodule's contents.  That means our
    text-patch + binary-archive pipeline silently drops every file
    inside submodules.  Real bite: ``html5lib_html5lib-python_pr-1``
    has ``html5lib/tests/testdata`` as a submodule; the test conftest
    aborts with ``"Exit: testdata not available!"`` because the
    extracted dataset contains no contents for that directory.

    The fix is to enumerate submodules here, expand each into its
    actual file list (from the **checked-out working tree** since
    submodule HEAD is currently materialised after ``git checkout
    -f parent_commit``), and feed those files into the binary tarball
    pipeline alongside the regular test_files.

    Output is parsed from ``git submodule status --recursive``
    which prints one ``<sha> <path> (<describe>)`` line per
    submodule.  We only need ``<path>`` (the URL is collected for
    audit logging).  Recursive so nested submodules are caught.
    """
    result = await session.execute(
        "git submodule status --recursive 2>/dev/null || true",
        cwd=workdir, timeout=60,
    )
    if not result.success:
        return {}
    submodules: dict[str, str] = {}
    for line in result.stdout.splitlines():
        # Line shapes:
        #   ' <sha> <path> (<describe>)'   -- initialized + clean
        #   '-<sha> <path>'                 -- not initialized
        #   '+<sha> <path> (<describe>)'    -- modified
        #   'U<sha> <path>'                 -- merge conflicts
        # We accept all of them; the path is the second whitespace-
        # separated token after stripping the leading status char.
        stripped = line.lstrip("-+U ").lstrip()
        parts = stripped.split(None, 2)
        if len(parts) < 2:
            continue
        path = parts[1]
        # Look up the URL from .gitmodules — informational only.
        url_result = await session.execute(
            f"git config -f .gitmodules --get "
            f"submodule.{shlex.quote(path)}.url 2>/dev/null || true",
            cwd=workdir, timeout=10,
        )
        url = (url_result.stdout or "").strip() if url_result.success else ""
        submodules[path] = url
    return submodules


async def _expand_submodule_files(
    session, workdir: str, submodule_path: str,
    max_files: int = 5000,
) -> list[str]:
    """Enumerate every regular file inside a submodule's checked-out
    working tree, with paths relative to ``workdir``.

    Uses ``find`` (not ``git ls-files``) because the submodule's
    ``.git`` directory may not be fully wired up after the parent
    checkout — and we want what's physically on disk anyway, since
    that's what the agent will see post-clean.  ``.git`` itself is
    excluded.

    The ``max_files`` cap protects against pathological submodules
    (e.g. a vendored test-fixture set with tens of thousands of
    tiny files).  When exceeded we return the first ``max_files``
    sorted alphabetically and log a warning.
    """
    cmd = (
        f"cd {shlex.quote(submodule_path)} && "
        f"find . -type f -not -path './.git/*' "
        f"-not -path '*/.git/*' 2>/dev/null | sort"
    )
    result = await session.execute(cmd, cwd=workdir, timeout=120)
    if not result.success or not result.stdout.strip():
        return []
    files: list[str] = []
    for line in result.stdout.splitlines():
        f = line.strip()
        if not f or f == ".":
            continue
        # ``find .`` prefixes everything with ``./`` — strip it.
        if f.startswith("./"):
            f = f[2:]
        # Rebuild full workspace-relative path.
        files.append(f"{submodule_path}/{f}")
    if len(files) > max_files:
        logger.warning(
            "Submodule %s has %d files (> cap %d); truncating",
            submodule_path, len(files), max_files,
        )
        files = files[:max_files]
    return files


async def _detect_binary_files(
    session, workdir: str, ref: str, files: list[str],
) -> tuple[set[str], set[str]]:
    """Return ``(binary_set, text_set)`` for ``files`` vs git ref.

    Uses ``git diff --numstat`` which prints ``-\\t-\\t<path>`` for
    binary files and ``<added>\\t<removed>\\t<path>`` for text files.
    Files not modified by the diff are reported as text by default.
    """
    if not files:
        return set(), set()

    # ``git diff --numstat`` doesn't quote paths by default unless
    # ``core.quotePath`` is enabled.  Disable quoting so we read raw
    # bytes back as-is.
    quoted = " ".join(shlex.quote(f) for f in files)
    result = await session.execute(
        f"git -c core.quotePath=false diff --numstat {shlex.quote(ref)} -- {quoted}",
        cwd=workdir, timeout=120,
    )
    binary: set[str] = set()
    text: set[str] = set()
    if not result.success:
        # If diff fails outright treat everything as text (best effort).
        logger.warning(
            "git diff --numstat failed (exit=%d, stderr=%s); "
            "assuming all files are text",
            result.exit_code, result.stderr[:200],
        )
        return set(), set(files)

    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, removed = parts[0], parts[1]
        path = "\t".join(parts[2:]).strip()
        if not path:
            continue
        if added == "-" and removed == "-":
            binary.add(path)
        else:
            text.add(path)

    # Any file we asked about but didn't see in the diff output had no
    # changes vs ``ref``.  Treat it as text (it'll show as empty in the
    # patch but won't blow up).
    seen = binary | text
    for f in files:
        if f not in seen:
            text.add(f)
    return binary, text


_PACKAGE_FIELDS = (
    "pypi_name", "pypi_name_candidates",
    "import_names", "package_setup_files",
    "package_extract_error",
)


def _has_package_info(raw: dict) -> bool:
    """Return True when ``raw`` already carries the package-name fields
    from a previous post-3 run.

    We treat the row as already-extracted when *any* of the three
    primary fields (``pypi_name`` / ``import_names`` /
    ``package_setup_files``) is non-empty.  An all-empty row triggers
    re-extraction, which mirrors how ``post_3_extract_package_name``
    treats those instances (the upstream extractor failed and we
    should try again with the current sandbox state).
    """
    if raw.get("pypi_name"):
        return True
    if raw.get("import_names"):
        return True
    if raw.get("package_setup_files"):
        return True
    return False


# Cached once per process to avoid re-rendering the 660-line script
# on every instance.
_EXTRACTOR_SCRIPT_CACHE: str | None = None


def _get_cached_extractor() -> str:
    global _EXTRACTOR_SCRIPT_CACHE
    if _EXTRACTOR_SCRIPT_CACHE is None:
        _EXTRACTOR_SCRIPT_CACHE = get_extractor_script()
    return _EXTRACTOR_SCRIPT_CACHE


async def _extract_package_info(
    session, workdir: str, instance_id: str,
    instance_timeout: int = 180,
) -> dict:
    """Run the doc2repo package-name extractor inside the sandbox.

    Returns a dict with keys matching the post_3 schema:
    ``pypi_name`` / ``pypi_name_candidates`` / ``import_names`` /
    ``package_setup_files`` / ``package_extract_error``.

    On any failure we return placeholder values + a non-null
    ``package_extract_error`` so downstream filters
    (``package_extract_error is None``) can drop the instance.

    The extractor runs *after* ``git checkout -f <parent_commit>``
    in the caller, so it inspects the canonical pre-clean tree.
    """
    script_remote = "/tmp/_awe_extract_pkg.py"
    output_remote = "/tmp/_awe_extract_pkg_output.json"
    meta_remote = "/tmp/_awe_extract_pkg_meta.json"
    result_default = {
        "pypi_name": "",
        "pypi_name_candidates": [],
        "import_names": [],
        "package_setup_files": [],
        "package_extract_error": None,
    }
    try:
        await session.upload_file(
            script_remote, _get_cached_extractor().encode("utf-8"),
        )
    except Exception as exc:
        result_default["package_extract_error"] = (
            f"upload_failed: {type(exc).__name__}: {exc}"
        )
        return result_default

    # Try python3 then python — some legacy images only ship the
    # bare ``python`` name on PATH.
    args = (
        f"{shlex.quote(script_remote)} "
        f"--workdir {shlex.quote(workdir)} "
        f"--output {shlex.quote(output_remote)} "
        f"--meta-output {shlex.quote(meta_remote)}"
    )
    cmd_candidates = [f"python3 {args}", f"python {args}"]
    last_err = ""
    ran_ok = False
    for cmd in cmd_candidates:
        try:
            res = await session.execute(
                cmd, cwd=workdir, timeout=instance_timeout,
            )
        except Exception as exc:
            last_err = f"{cmd.split(' ', 1)[0]} exception: {exc}"
            continue
        if res.success:
            ran_ok = True
            break
        last_err = (
            f"{cmd.split(' ', 1)[0]} exit={res.exit_code}: "
            f"{(res.stderr or res.stdout)[-500:]}"
        )

    if not ran_ok:
        result_default["package_extract_error"] = last_err or "command failed"
        # Clean up best-effort.
        await session.execute(
            f"rm -f {shlex.quote(script_remote)} "
            f"{shlex.quote(output_remote)} "
            f"{shlex.quote(meta_remote)}",
            cwd=workdir, timeout=10,
        )
        return result_default

    # Download both output + meta.  Meta carries any error the script
    # surfaced itself (parser failures, etc.) without crashing.
    helper_err = None
    try:
        meta_blob = await session.download_file(meta_remote)
        helper_meta = json.loads(meta_blob.decode("utf-8", errors="replace"))
        helper_err = helper_meta.get("error")
    except FileNotFoundError:
        helper_meta = {}
    except Exception as exc:
        helper_err = f"meta_download_failed: {type(exc).__name__}: {exc}"

    try:
        out_blob = await session.download_file(output_remote)
        extracted = json.loads(out_blob.decode("utf-8", errors="replace"))
    except Exception as exc:
        result_default["package_extract_error"] = (
            f"output_download_failed: {type(exc).__name__}: {exc}"
        )
        await session.execute(
            f"rm -f {shlex.quote(script_remote)} "
            f"{shlex.quote(output_remote)} "
            f"{shlex.quote(meta_remote)}",
            cwd=workdir, timeout=10,
        )
        return result_default

    # Cleanup sandbox-side artifacts.
    await session.execute(
        f"rm -f {shlex.quote(script_remote)} "
        f"{shlex.quote(output_remote)} "
        f"{shlex.quote(meta_remote)}",
        cwd=workdir, timeout=10,
    )

    return {
        "pypi_name": str(extracted.get("pypi_name") or ""),
        "pypi_name_candidates": list(extracted.get("pypi_name_candidates") or []),
        "import_names": list(extracted.get("import_names") or []),
        "package_setup_files": list(extracted.get("setup_files") or []),
        "package_extract_error": helper_err or None,
    }


async def _git_diff_via_file(
    session, workdir: str, ref: str, files: list[str],
    instance_id: str,
    timeout: int = 300,
    tmp_path: str = "/tmp/_awe_extract_diff.patch",
) -> str:
    """Run ``git diff`` and download the output via ``session.download_file``.

    ``session.execute`` truncates stdout at 24 MiB (Portal + Docker
    backends both have ``_MAX_OUTPUT_BYTES = 24 * 1024 * 1024``).  Very
    large test_patches (e.g. ``stwcs``, ``cmdstanpy``, ``onnxruntime``
    extensions) hit that ceiling silently and end up as ``corrupt patch
    at line N`` on the apply side.

    Writing the diff to a file inside the sandbox and downloading it
    via the file-transfer API (base64 over Portal HTTP / Docker
    put_archive) sidesteps the stdout cap entirely.

    Returns the patch text (utf-8 decoded with ``errors='replace'`` for
    odd encodings) or an empty string if the diff fails / file is
    missing.
    """
    if not files:
        return ""
    files_arg = " ".join(shlex.quote(f) for f in files)
    # Redirect to file inside sandbox.  We always cleanup at the end.
    run_result = await session.execute(
        f"git diff {shlex.quote(ref)} -- {files_arg} > "
        f"{shlex.quote(tmp_path)}",
        cwd=workdir, timeout=timeout,
    )
    if not run_result.success:
        logger.warning(
            "Instance %s: git diff (to file) failed exit=%d: %s",
            instance_id, run_result.exit_code, run_result.stderr[:300],
        )
        await session.execute(
            f"rm -f {shlex.quote(tmp_path)}",
            cwd=workdir, timeout=10,
        )
        return ""
    try:
        patch_bytes = await session.download_file(tmp_path)
    except FileNotFoundError:
        # ``git diff`` with no changes writes a 0-byte file that may
        # not have been created at all on some filesystems — treat as
        # empty diff.
        logger.info(
            "Instance %s: git diff produced no output (file missing)",
            instance_id,
        )
        return ""
    finally:
        await session.execute(
            f"rm -f {shlex.quote(tmp_path)}",
            cwd=workdir, timeout=10,
        )
    return patch_bytes.decode("utf-8", errors="replace")


async def _build_binary_archive(
    session, workdir: str, binary_files: list[str],
    instance_id: str,
) -> tuple[str, list[str], str | None]:
    """Pack ``binary_files`` into a tar.gz inside the sandbox and
    download as base64.  Returns ``(b64_archive, included_files, error)``.

    ``error`` is None on success.  On any failure the archive is empty
    and ``error`` is a short reason code.  Edge cases handled:

    * Empty ``binary_files`` → ("", [], None).
    * tar exit non-zero → returns error code ``"tar_failed"``.
    * Resulting archive exceeds :data:`_MAX_BINARY_ARCHIVE_BYTES` →
      ``"binary_too_large"``.
    """
    if not binary_files:
        return "", [], None

    # Build the file list via ``printf '%s\0' arg1 arg2 ...`` which
    # emits a real NUL between each entry (POSIX argv cannot contain
    # NUL bytes, so we can't pass the joined list as a single
    # argument).  ``tar --null -T -`` then reads the list from stdin,
    # which tolerates names containing whitespace / quotes /
    # non-ASCII characters.
    files_args = " ".join(shlex.quote(f) for f in binary_files)
    cmd = (
        f"printf '%s\\0' {files_args} | "
        f"tar --null -T - -czf {shlex.quote(_BINARY_TAR_REMOTE)}"
    )
    tar_result = await session.execute(cmd, cwd=workdir, timeout=600)
    if not tar_result.success:
        logger.warning(
            "Instance %s: tar of %d binary files failed (exit=%d): %s",
            instance_id, len(binary_files),
            tar_result.exit_code, tar_result.stderr[:300],
        )
        # Best effort cleanup.
        await session.execute(
            f"rm -f {shlex.quote(_BINARY_TAR_REMOTE)}",
            cwd=workdir, timeout=10,
        )
        return "", [], "tar_failed"

    # Download the archive bytes.
    try:
        archive_bytes = await session.download_file(_BINARY_TAR_REMOTE)
    except FileNotFoundError:
        logger.warning(
            "Instance %s: tar produced no archive at %s",
            instance_id, _BINARY_TAR_REMOTE,
        )
        return "", [], "archive_missing"
    finally:
        # Always clean up the staged archive inside the sandbox.
        await session.execute(
            f"rm -f {shlex.quote(_BINARY_TAR_REMOTE)}",
            cwd=workdir, timeout=10,
        )

    if len(archive_bytes) > _MAX_BINARY_ARCHIVE_BYTES:
        logger.warning(
            "Instance %s: binary archive %d bytes exceeds cap %d, "
            "skipping binary inclusion",
            instance_id, len(archive_bytes), _MAX_BINARY_ARCHIVE_BYTES,
        )
        return "", [], "binary_too_large"

    archive_b64 = base64.b64encode(archive_bytes).decode("ascii")
    logger.info(
        "Instance %s: packed %d binary files into %d bytes "
        "(b64=%d bytes)",
        instance_id, len(binary_files), len(archive_bytes), len(archive_b64),
    )
    return archive_b64, sorted(binary_files), None


async def _delete_docker_image(image: str) -> None:
    """Try to delete a docker image after use."""
    import subprocess
    try:
        subprocess.run(
            ["docker", "rmi", "-f", image],
            capture_output=True, timeout=60,
        )
        logger.info("Deleted image: %s", image)
    except Exception as exc:
        logger.warning("Failed to delete image %s: %s", image, exc)


async def extract_patch_for_instance(
    raw: dict,
    config,
    semaphore: asyncio.Semaphore,
    del_done_images: bool = False,
    extract_package_info: bool = True,
) -> dict:
    """Extract test patch for a single instance.

    Returns the raw dict augmented with ``test_patch`` and ``test_files``.
    """
    from awe_agent.core.runtime import RuntimeConfig
    from awe_agent.core.task.runner import runtime_registry

    instance_id = raw.get("instance_id", "unknown")

    async with semaphore:
        logger.info("Processing %s ...", instance_id)

        image = raw.get("image", raw.get("image_url", ""))
        workdir = raw.get("workdir", "/workspace")
        parent_commit = raw.get("parent_commit", "")

        if not image:
            logger.warning("Instance %s has no image, skipping", instance_id)
            raw["test_patch"] = ""
            raw["test_files"] = []
            raw["test_binary_archive_b64"] = ""
            raw["test_binary_files"] = []
            raw["bug_b_overrides_agent_code"] = []
            raw["extract_error"] = "no_image"
            return raw

        try:
            runtime_config = config.runtime.model_copy(
                update={"image": image, "workdir": workdir},
            )
            runtime_cls = runtime_registry.get(config.runtime.backend)
            runtime = runtime_cls(runtime_config)

            async with runtime.session(image) as session:
                # Checkout the parent commit
                if parent_commit:
                    checkout_result = await session.execute(
                        f"git checkout -f {parent_commit}",
                        cwd=workdir, timeout=120,
                    )
                    if not checkout_result.success:
                        logger.error(
                            "Instance %s: git checkout %s failed: %s",
                            instance_id, parent_commit, checkout_result.stderr[:300],
                        )
                        raw["test_patch"] = ""
                        raw["test_files"] = []
                        raw["test_binary_archive_b64"] = ""
                        raw["test_binary_files"] = []
                        raw["bug_b_overrides_agent_code"] = []
                        raw["extract_error"] = f"checkout_failed: {checkout_result.stderr[:200]}"
                        return raw

                # Backfill package-name fields (pypi_name / import_names /
                # package_setup_files) when missing.  These power the
                # downstream cheating-warning logic — without them the
                # detector can't tell which sibling-whitelist .py files
                # are production code the agent was meant to write.
                #
                # ``_has_package_info`` skips the run when the input row
                # already carries non-empty values (the common case for
                # datasets produced by ``post_3_extract_package_name``).
                if extract_package_info and not _has_package_info(raw):
                    pkg_info = await _extract_package_info(
                        session, workdir, instance_id,
                    )
                    for k, v in pkg_info.items():
                        # Don't clobber a non-empty user-provided value.
                        if not raw.get(k):
                            raw[k] = v
                    if pkg_info.get("package_extract_error"):
                        logger.warning(
                            "Instance %s: package_extract_error=%s",
                            instance_id, pkg_info["package_extract_error"][:200],
                        )
                    elif pkg_info.get("pypi_name"):
                        logger.info(
                            "Instance %s: extracted pypi_name=%r, "
                            "import_names=%s",
                            instance_id, pkg_info["pypi_name"],
                            pkg_info["import_names"][:5],
                        )

                # Sanitize malformed pytest node IDs in passed_ptp/failed_ptp
                # (Bug E).  Mutates the raw record so downstream consumers
                # see the corrected ids.  We also re-emit ``passed_ptp_raw``
                # for audit if any rewrites occurred.
                passed_ptp_raw = list(raw.get("passed_ptp", []) or [])
                failed_ptp_raw = list(raw.get("failed_ptp", []) or [])
                passed_ptp, ptp_rewrites = await _sanitize_test_ids(
                    passed_ptp_raw, session, workdir,
                )
                failed_ptp, fptp_rewrites = await _sanitize_test_ids(
                    failed_ptp_raw, session, workdir,
                )
                if ptp_rewrites or fptp_rewrites:
                    logger.info(
                        "Instance %s: sanitized %d passed_ptp + %d failed_ptp "
                        "test ids (Bug E)",
                        instance_id, len(ptp_rewrites), len(fptp_rewrites),
                    )
                    raw["passed_ptp"] = passed_ptp
                    raw["failed_ptp"] = failed_ptp
                    raw["ptp_sanitizer_rewrites"] = {
                        "passed": ptp_rewrites,
                        "failed": fptp_rewrites,
                    }

                # Extract test dirs and direct test files from the
                # (now sanitized) passed_ptp list.
                test_dirs, direct_test_files = _extract_test_dirs_and_files(passed_ptp)

                if not direct_test_files:
                    logger.warning(
                        "Instance %s: no test files found in passed_ptp", instance_id,
                    )
                    raw["test_patch"] = ""
                    raw["test_files"] = []
                    raw["test_binary_archive_b64"] = ""
                    raw["test_binary_files"] = []
                    raw["bug_b_overrides_agent_code"] = []
                    raw["extract_error"] = "no_test_files"
                    return raw

                # Collect all test-related files.  Strategy:
                #   0. **Always** scan repo root for test infrastructure
                #      + project-metadata files (conftest, settings, setup.py,
                #      README, MANIFEST.in, LICENSE, ...).  These are needed
                #      whether or not any test sits at the root, because
                #      ``pip install -e .`` reads them and pytest reads
                #      ``conftest.py`` from every ancestor.
                #   1. For dirs in test_dirs with "test" basename → grab
                #      everything in that dir (recursive).
                #   2. For other dirs in test_dirs → only files matching
                #      test patterns.
                #   3. Ancestor walker grabs conftest at each non-root
                #      level + recursive grab when the ancestor itself
                #      is a test dir.
                #   4. Bug B sibling-helper whitelist.
                all_test_files: set[str] = set()

                # Step 0 — unconditional root scan, **filtered by
                # ``will_clean_delete``**.  We ship a root-level file in
                # the test_patch ONLY when clean.sh would otherwise
                # delete it.  Shipping clean-preserved files like
                # ``setup.py`` / ``pyproject.toml`` would silently
                # **overwrite the agent's own build configuration**
                # with the upstream golden version, which has caused
                # ~140 regressions (e.g. ``csachs_pyproject-flake8``:
                # upstream pyproject uses ``dynamic = ["version"]`` via
                # flit, but the agent's ``__init__.py`` doesn't expose
                # ``__version__`` → install fail).
                #
                # The pattern list is intentionally broad — the
                # ``will_clean_delete`` filter ensures we only keep the
                # ones the workspace will actually lose after clean.
                ROOT_PATTERNS = [
                    # Test infrastructure (mostly deleted by clean)
                    "conftest.py",
                    "test_*.py", "*_test.py", "*_tests.py",
                    "test_settings.py", "test_config.py",
                    "settings.py",
                    ".coveragerc",
                    # Build / packaging metadata
                    # (most are clean-preserved → filter drops them)
                    "setup.py", "setup.cfg", "pyproject.toml",
                    "pytest.ini", "tox.ini",
                    "MANIFEST.in",
                    "Dockerfile", "Makefile",
                    ".python-version",
                    # Project files
                    "README", "README.md", "README.rst", "README.txt",
                    "LICENSE", "LICENSE.md", "LICENSE.txt",
                    "LICENCE", "LICENCE.md", "LICENCE.txt",
                    # Native source occasionally at root
                    "*.c", "*.cpp", "*.cxx", "*.cc", "*.h", "*.hpp",
                    "*.pyx", "*.pxd",
                    # Root __init__.py for top-level packages
                    "__init__.py",
                ]
                globs = " ".join(shlex.quote(p) for p in ROOT_PATTERNS)
                root_result = await session.execute(
                    f"git ls-files {globs} 2>/dev/null || true",
                    cwd=workdir, timeout=30,
                )
                # M5: track root-level files we keep so the cheating-
                # detection pass below can also audit them — without
                # this, root ``__init__.py`` / ``settings.py`` (legit
                # parts of a flat-layout package) get shipped as
                # "test_patch" and silently overwrite the agent's own
                # implementation.
                root_kept_files: set[str] = set()
                if root_result.success and root_result.stdout.strip():
                    n_kept = 0
                    n_filtered = 0
                    for ln in root_result.stdout.splitlines():
                        f = ln.strip()
                        if not f or "/" in f:
                            continue
                        # Only ship files clean.sh would delete — those
                        # are the ones the workspace will lose.  Files
                        # like ``setup.py`` survive clean and should
                        # remain whatever the agent wrote.
                        if will_clean_delete(f):
                            all_test_files.add(f)
                            root_kept_files.add(f)
                            n_kept += 1
                        else:
                            n_filtered += 1
                    if n_kept or n_filtered:
                        logger.debug(
                            "Instance %s: root scan kept %d, "
                            "filtered (clean-preserved) %d",
                            instance_id, n_kept, n_filtered,
                        )

                for d in test_dirs:
                    if d == ".":
                        # Tests live at the repo root; the unconditional
                        # root scan below covers this case.  Nothing
                        # special to do here.
                        pass
                    elif _is_test_directory(d):
                        # Directory name contains "test" → grab everything
                        result = await session.execute(
                            f'git ls-files "{d}/" 2>/dev/null || true',
                            cwd=workdir, timeout=30,
                        )
                        if result.success and result.stdout.strip():
                            all_test_files.update(
                                f.strip() for f in result.stdout.strip().split("\n") if f.strip()
                            )
                    else:
                        # Non-test dir → only grab files matching test patterns
                        globs = " ".join(f'"{d}/{p}"' for p in _TEST_FILE_PATTERNS)
                        result = await session.execute(
                            f"git ls-files {globs} 2>/dev/null || true",
                            cwd=workdir, timeout=30,
                        )
                        if result.success and result.stdout.strip():
                            all_test_files.update(
                                f.strip() for f in result.stdout.strip().split("\n") if f.strip()
                            )

                # ── Bug A: ancestor walker ─────────────────────────
                # For each direct test dir, walk UP toward the repo
                # root and decide per-ancestor what to grab:
                #
                #   * Ancestor IS a test directory by name → recursive
                #     scan (catches ``tests/mocked_bot.py``,
                #     ``tests/mocks/`` data, etc.).  Cap at
                #     ``_MAX_ANCESTOR_FILES_PER_SCAN``; if exceeded,
                #     fall back to one-level pattern glob (mirrors the
                #     non-test direct-dir handling above).
                #   * Ancestor is NOT a test directory → conservative
                #     ``conftest.py`` only.  Broader patterns would
                #     match production source files (``utils.py``,
                #     ``base.py``).
                #
                # The walker is globally deduped across the outer
                # ``for d in test_dirs`` loop so each ancestor is
                # scanned at most once.
                ancestor_visited: set[str] = set()
                all_ancestors: list[str] = []
                for d in test_dirs:
                    parent = d
                    while True:
                        parent = parent.rsplit("/", 1)[0] if "/" in parent else "."
                        if parent in ancestor_visited or not parent:
                            break
                        ancestor_visited.add(parent)
                        all_ancestors.append(parent)
                        if parent == ".":
                            break

                for parent in all_ancestors:
                    if parent == ".":
                        # Root branch — already covered by the ``d == "."``
                        # case inside the direct-dir loop (Bug G fix).
                        # Don't double-scan.
                        continue
                    if _is_test_directory(parent):
                        # Recursive scan with size cap.
                        result = await session.execute(
                            f"git ls-files {shlex.quote(parent + '/')} "
                            f"2>/dev/null || true",
                            cwd=workdir, timeout=60,
                        )
                        new_files: list[str] = []
                        if result.success and result.stdout.strip():
                            new_files = [
                                ln.strip()
                                for ln in result.stdout.splitlines()
                                if ln.strip()
                            ]
                        if len(new_files) > _MAX_ANCESTOR_FILES_PER_SCAN:
                            logger.warning(
                                "Instance %s: ancestor %s has %d files "
                                "(> cap %d); falling back to one-level "
                                "pattern scan",
                                instance_id, parent, len(new_files),
                                _MAX_ANCESTOR_FILES_PER_SCAN,
                            )
                            globs = " ".join(
                                shlex.quote(f"{parent}/{p}")
                                for p in _TEST_FILE_PATTERNS
                            )
                            lvl = await session.execute(
                                f"git ls-files {globs} 2>/dev/null || true",
                                cwd=workdir, timeout=30,
                            )
                            if lvl.success and lvl.stdout.strip():
                                all_test_files.update(
                                    ln.strip()
                                    for ln in lvl.stdout.splitlines()
                                    if ln.strip()
                                )
                        else:
                            all_test_files.update(new_files)
                    else:
                        # Non-test ancestor: conservative conftest only.
                        result = await session.execute(
                            f"git ls-files "
                            f"{shlex.quote(parent + '/conftest.py')} "
                            f"2>/dev/null || true",
                            cwd=workdir, timeout=10,
                        )
                        if result.success and result.stdout.strip():
                            all_test_files.update(
                                ln.strip()
                                for ln in result.stdout.splitlines()
                                if ln.strip()
                            )

                # Always include the direct test files themselves
                all_test_files.update(direct_test_files)

                # ── Bug B: sibling-helper whitelist ────────────────
                # Scan the well-known sibling helper dir names at the
                # **repo root**.  Each candidate file is filtered via
                # ``will_clean_delete`` so we only ship files clean.sh
                # would actually nuke (non-code data files survive on
                # their own).  Skip directories already covered by
                # the test-dir walker above (avoids redundant scans
                # when ``tests/helpers/`` exists, for example).
                cheating_candidates: list[str] = []
                bug_b_skipped_dirs: list[dict] = []
                for sib in _BUG_B_SIBLING_DIRS:
                    # Hard cap: if we're already at the overall ceiling,
                    # stop adding from any further whitelist dirs.  The
                    # remaining ones get recorded for audit but not
                    # scanned.
                    if len(all_test_files) >= _MAX_TEST_FILES:
                        bug_b_skipped_dirs.append({
                            "dir": sib,
                            "reason": "global_cap_reached",
                        })
                        continue

                    # Skip if already covered by a direct or ancestor
                    # test_dir scan (e.g. ``tests/helpers`` lives
                    # under ``tests/`` and was already grabbed).
                    if any(tf.startswith(sib + "/") or tf == sib
                           for tf in all_test_files):
                        # Already represented — no need to rescan.
                        continue
                    sib_result = await session.execute(
                        f"git ls-files {shlex.quote(sib + '/')} "
                        f"2>/dev/null || true",
                        cwd=workdir, timeout=20,
                    )
                    if not sib_result.success or not sib_result.stdout.strip():
                        continue
                    candidates = [
                        ln.strip() for ln in sib_result.stdout.splitlines()
                        if ln.strip()
                    ]
                    # Only include files clean.sh would delete.  Files
                    # that survive clean (.csv, .tf, .heic, ...) are
                    # still in the eval workspace; no need to ship.
                    will_be_deleted = [
                        c for c in candidates if will_clean_delete(c)
                    ]
                    if not will_be_deleted:
                        continue

                    # Per-dir cap: refuse to drag in pathological cases
                    # (e.g. an ``examples/`` containing 1000s of files).
                    # If exceeded, skip this dir entirely rather than
                    # truncating arbitrarily — the audit field tells
                    # the dataset owner to investigate manually.
                    if len(will_be_deleted) > _MAX_ANCESTOR_FILES_PER_SCAN:
                        logger.warning(
                            "Instance %s: Bug B whitelist '%s/' has %d "
                            "deletable files (> cap %d); skipping",
                            instance_id, sib, len(will_be_deleted),
                            _MAX_ANCESTOR_FILES_PER_SCAN,
                        )
                        bug_b_skipped_dirs.append({
                            "dir": sib,
                            "reason": "per_dir_cap_exceeded",
                            "deletable_count": len(will_be_deleted),
                        })
                        continue

                    logger.info(
                        "Instance %s: Bug B whitelist '%s/' adds %d/%d files "
                        "(filtered by will_clean_delete)",
                        instance_id, sib,
                        len(will_be_deleted), len(candidates),
                    )
                    all_test_files.update(will_be_deleted)
                    # Track .py files for the cheating-risk audit field.
                    cheating_candidates.extend(
                        f for f in will_be_deleted if f.endswith(".py")
                    )

                # Cheating-risk warning: if any sibling-whitelist file
                # shares its top-level segment with the agent's expected
                # package (``pypi_name`` / ``import_names``), it might
                # be production code the agent should have written.
                # Emit a per-file warning + JSONL field so the dataset
                # owner can audit; we do NOT auto-drop them — that's a
                # human call.
                pkg_identifiers: set[str] = set()
                for src in (
                    [raw.get("pypi_name", "")],
                    raw.get("pypi_name_candidates", []) or [],
                    raw.get("import_names", []) or [],
                    [raw.get("repo", "").rsplit("/", 1)[-1]],
                ):
                    for item in src:
                        if isinstance(item, dict):
                            item = item.get("name", "")
                        if isinstance(item, str) and item:
                            normalized = item.lower().replace("-", "_")
                            pkg_identifiers.add(normalized)

                # Build cheating-risk overrides.  We flag a captured
                # ``.py`` file if EITHER:
                #   (a) its top-level path segment matches a known
                #       package identifier (``pypi_name`` /
                #       ``import_names`` / ``repo``); OR
                #   (b) it lives in a sibling-whitelist dir that we
                #       captured >= 5 ``.py`` files from — this is the
                #       kobotoolbox pattern where ``helpers/`` is
                #       actually a production package the agent had to
                #       write (heuristic, not perfect, but the only
                #       signal available when ``pypi_name`` is empty).
                # The override list is **advisory**: extract_patch
                # still ships the files; the downstream auditor
                # decides whether the score is trustworthy.
                cheating_overrides: list[dict] = []
                heavy_sibling_dirs: dict[str, int] = {}
                for cand in cheating_candidates:
                    top = cand.split("/", 1)[0].lower().replace("-", "_")
                    heavy_sibling_dirs[top] = heavy_sibling_dirs.get(top, 0) + 1
                _HEAVY_THRESHOLD = 5
                for cand in cheating_candidates:
                    top = cand.split("/", 1)[0].lower().replace("-", "_")
                    reasons = []
                    if top in pkg_identifiers:
                        reasons.append(f"matches_pkg_identifier:{top}")
                    if heavy_sibling_dirs.get(top, 0) >= _HEAVY_THRESHOLD:
                        reasons.append(
                            f"heavy_sibling_dir:{top}"
                            f"({heavy_sibling_dirs[top]}_py_files)"
                        )
                    if reasons:
                        cheating_overrides.append({
                            "path": cand,
                            "reasons": reasons,
                        })
                # M5: also audit ROOT_PATTERNS captures.  Root-level
                # ``__init__.py`` / ``settings.py`` / ``test_settings.py``
                # may BE the agent's flat-layout package init or its
                # config module — shipping them as test_patch silently
                # overrides agent code.  Flag any such file whose
                # basename (sans ``.py``) matches a pkg_identifier so
                # the dataset auditor can decide.
                for f in sorted(root_kept_files):
                    if not f.endswith(".py"):
                        continue
                    stem = f[:-3].lower().replace("-", "_")
                    matches: list[str] = []
                    # Direct stem match: ``foo.py`` for pkg ``foo``.
                    if stem in pkg_identifiers:
                        matches.append(f"root_stem:{stem}")
                    # ``__init__.py`` at root indicates a flat-layout
                    # package — the file itself is the package init
                    # for ANY pkg_identifier that the repo names
                    # itself after (no way to disambiguate without
                    # peeking inside).
                    if stem == "__init__" and pkg_identifiers:
                        matches.append("root_init_py_flat_layout")
                    if matches:
                        cheating_overrides.append({
                            "path": f,
                            "reasons": matches,
                        })
                if cheating_overrides:
                    logger.warning(
                        "Instance %s: %d Bug B whitelist .py files flagged "
                        "as possible cheating (kobotoolbox-style helpers "
                        "being golden-overridden) — see "
                        "bug_b_overrides_agent_code field for audit",
                        instance_id, len(cheating_overrides),
                    )
                raw["bug_b_overrides_agent_code"] = cheating_overrides
                if bug_b_skipped_dirs:
                    raw["bug_b_skipped_dirs"] = bug_b_skipped_dirs

                # Enforce overall test_files cap.
                if len(all_test_files) > _MAX_TEST_FILES:
                    logger.warning(
                        "Instance %s: %d test files exceeds cap %d; "
                        "truncating to first %d (sorted) — keeping all "
                        "%d direct_test_files unconditionally",
                        instance_id, len(all_test_files), _MAX_TEST_FILES,
                        _MAX_TEST_FILES, len(direct_test_files),
                    )
                    # ``direct_test_files`` are the literal files
                    # referenced by ``passed_ptp`` — dropping them
                    # breaks evaluation entirely (the test ids point
                    # to files we no longer ship).  So always keep
                    # those + the next slice of helpers up to the cap.
                    must_keep = set(direct_test_files) & set(all_test_files)
                    remainder_budget = max(_MAX_TEST_FILES - len(must_keep), 0)
                    helpers = sorted(all_test_files - must_keep)[:remainder_budget]
                    raw["test_files_truncated"] = True
                    raw["test_files_truncated_from"] = len(all_test_files)
                    all_test_files = must_keep | set(helpers)
                    if len(must_keep) > _MAX_TEST_FILES:
                        # direct_test_files alone exceeds the cap —
                        # we keep them all anyway (correctness > cap).
                        logger.warning(
                            "Instance %s: direct_test_files (%d) > cap "
                            "(%d); cap relaxed to keep all directs",
                            instance_id, len(must_keep), _MAX_TEST_FILES,
                        )

                # ── Git submodule expansion ─────────────────────────
                # ``git diff`` between commits represents a submodule
                # as a single gitlink entry (mode 160000) referencing a
                # sha — it does NOT emit per-file diffs for the
                # submodule's contents.  That means our text-patch +
                # binary-archive pipeline silently drops every file
                # inside submodules.  Confirmed bite:
                # ``html5lib_html5lib-python_pr-1`` has
                # ``html5lib/tests/testdata`` as a submodule; the test
                # conftest aborts with ``"Exit: testdata not available!
                # Please run git submodule update --init --recursive"``
                # because nothing was shipped.
                #
                # Fix here: enumerate submodules touched by our test
                # scan, list their working-tree contents directly
                # (the parent ``git checkout -f parent_commit``
                # already materialised them), and route those files
                # into the binary-archive path.  They CANNOT go
                # through the parent repo's git diff because they're
                # tracked in a different repo entirely.
                all_submodules = await _discover_submodules(session, workdir)
                submodule_files: set[str] = set()
                submodule_paths_touched: list[str] = []
                submodule_uninitialized: list[str] = []
                if all_submodules:
                    for sub_path in all_submodules:
                        # Only expand submodules our test scan actually
                        # reaches — a vendored submodule deep in unrelated
                        # code shouldn't blow up the archive.
                        touched = (
                            sub_path in all_test_files
                            or any(tf.startswith(sub_path + "/") for tf in all_test_files)
                            or any(
                                td == sub_path
                                or td.startswith(sub_path + "/")
                                or sub_path.startswith(td + "/")
                                for td in test_dirs
                            )
                        )
                        if not touched:
                            continue
                        expanded = await _expand_submodule_files(
                            session, workdir, sub_path,
                        )
                        if not expanded:
                            # C1: a TOUCHED submodule produced zero
                            # files — almost always means ``git
                            # submodule update --init`` was never run
                            # against the parent image.  Silently
                            # skipping leaves the dataset missing the
                            # fixture, the test patch applies cleanly,
                            # and evaluation only fails at test-collection
                            # time with a confusing "file not found"
                            # error.  Record the path so the dataset
                            # auditor (and downstream re-eval tooling)
                            # can surface the broken instance up front.
                            submodule_uninitialized.append(sub_path)
                            logger.warning(
                                "Instance %s: submodule %s appears "
                                "uninitialized (no files); test patch "
                                "will be incomplete",
                                instance_id, sub_path,
                            )
                            continue
                        submodule_paths_touched.append(sub_path)
                        submodule_files.update(expanded)
                        # The bare submodule path is a gitlink entry —
                        # drop it so the git-diff dance doesn't choke
                        # on it.
                        all_test_files.discard(sub_path)
                    if submodule_files:
                        logger.info(
                            "Instance %s: expanded %d submodule(s) into "
                            "%d files (will ship via binary archive)",
                            instance_id, len(submodule_paths_touched),
                            len(submodule_files),
                        )
                        raw["test_submodule_paths"] = submodule_paths_touched
                if submodule_uninitialized:
                    raw["submodule_uninitialized"] = submodule_uninitialized
                # Submodule files become part of test_files for
                # record-keeping; the binary partition below force-
                # routes them through the tar path.
                all_test_files |= submodule_files

                test_files = sorted(all_test_files)

                logger.info(
                    "Instance %s: found %d test-related files "
                    "(direct=%d, ancestors=%d, sibling-whitelist=%d, submodule=%d)",
                    instance_id, len(test_files), len(direct_test_files),
                    len(all_ancestors),
                    sum(1 for _ in _BUG_B_SIBLING_DIRS),
                    len(submodule_files),
                )

                # ── Build the patch ─────────────────────────────────
                # Strategy: remove the test files, commit, restore them,
                # then ``git diff`` against the removal commit gives us
                # a clean "add file" patch.  This avoids the empty-tree
                # hash which doesn't exist in shallow/pruned repos.
                #
                # Binary files (Bug C):
                # Unified text diffs collapse binary content into
                # ``Binary files … differ`` markers, so ``git apply``
                # ends up with empty/missing files on the eval side.
                # Instead we partition the file list:
                #   * text  → included in ``test_patch`` (text diff)
                #   * binary → packed into a tar.gz, shipped as
                #              ``test_binary_archive_b64`` and
                #              extracted by the evaluator directly.
                # Submodule files can't go through the parent repo's
                # rm + commit + restore dance — they're tracked in a
                # different git, so the parent ``git ls-files`` /
                # ``git diff`` flow ignores them.  Carve them out:
                # they stay on disk untouched throughout the dance,
                # and get tar-packed at the end alongside the
                # binary-detected files.
                parent_repo_files = [f for f in test_files if f not in submodule_files]
                files_arg_all = " ".join(shlex.quote(f) for f in parent_repo_files)

                await session.execute(
                    "git stash --include-untracked 2>/dev/null || true",
                    cwd=workdir, timeout=60,
                )
                await session.execute(
                    'git config user.email "extract@denovoswe.local" && '
                    'git config user.name "ExtractPatch"',
                    cwd=workdir, timeout=10,
                )
                # Remove test files and commit the removal.
                await session.execute(
                    f"rm -f {files_arg_all}",
                    cwd=workdir, timeout=30,
                )
                await session.execute(
                    'git add -A && git commit -m "remove test files" --allow-empty',
                    cwd=workdir, timeout=30,
                )
                no_tests_commit = (await session.execute(
                    "git rev-parse HEAD", cwd=workdir, timeout=10,
                )).stdout.strip()
                if not no_tests_commit:
                    # ``git rev-parse HEAD`` failed (corrupt index,
                    # missing HEAD ref, etc.) — every downstream
                    # ``git show <ref>:`` / ``git diff <ref>`` will
                    # silently degrade to "compare against working
                    # tree" and emit a misleading patch.  Bail out
                    # explicitly so the instance lands in the
                    # extract_error bucket instead of producing a
                    # confidently-wrong test_patch.
                    logger.error(
                        "Instance %s: could not resolve HEAD after "
                        "removing test files; aborting test-patch build",
                        instance_id,
                    )
                    raw["test_patch"] = ""
                    raw["test_files"] = sorted(all_test_files)
                    raw["test_binary_archive_b64"] = ""
                    raw["test_binary_files"] = []
                    raw["extract_error"] = "no_tests_commit_resolution_failed"
                    return raw

                # Restore the test files back into the working tree.
                await session.execute(
                    "git stash pop 2>/dev/null || true",
                    cwd=workdir, timeout=30,
                )
                checkout_tests = await session.execute(
                    f"git checkout HEAD~1 -- . 2>/dev/null; "
                    f"git checkout {shlex.quote(parent_commit)} -- "
                    f"{files_arg_all} 2>/dev/null || true",
                    cwd=workdir, timeout=60,
                )

                # Partition into binary vs text via git's own
                # binary-detection (numstat: ``-\t-\t<path>``).
                # Only feed parent-repo files — submodule contents
                # aren't in the parent's git index and would get
                # mis-classified as "text-but-empty" otherwise.
                binary_set, text_set = await _detect_binary_files(
                    session, workdir, no_tests_commit,
                    list(parent_repo_files),
                )
                # Force submodule files into the binary archive — the
                # only transport that works for content not tracked by
                # the parent repo.  ``_build_binary_archive`` tars
                # straight from the working tree so the files are
                # picked up exactly as they appear after the parent
                # checkout.
                if submodule_files:
                    binary_set |= submodule_files
                    text_set -= submodule_files
                text_files_sorted = sorted(text_set)
                binary_files_sorted = sorted(binary_set)
                if binary_files_sorted:
                    logger.info(
                        "Instance %s: %d binary fixtures detected (out of %d test files)",
                        instance_id, len(binary_files_sorted), len(test_files),
                    )

                # Text patch: only ask git for text files so the patch
                # never contains ``Binary files … differ`` noise.
                # Use ``_git_diff_via_file`` so we sidestep the 24 MiB
                # stdout cap that the runtime backends impose — large
                # patches (FITS-heavy repos, monorepos with thousands
                # of generated files) were silently truncated before.
                test_patch = ""
                if text_files_sorted:
                    raw_diff = await _git_diff_via_file(
                        session, workdir, no_tests_commit,
                        text_files_sorted, instance_id,
                    )
                    test_patch = raw_diff.strip()
                    if test_patch:
                        test_patch += "\n"

                # Fallback when the text diff is empty: synthesize
                # one from ``git show``.  This used to read ``HEAD``
                # which is the "no-tests" commit (i.e. the file is
                # already gone) — use the actual parent_commit
                # instead (Bug D).
                if text_files_sorted and not test_patch:
                    logger.warning(
                        "Instance %s: git diff produced empty patch for %d text "
                        "files; trying git show fallback against parent_commit",
                        instance_id, len(text_files_sorted),
                    )
                    parts = []
                    fallback_ref = parent_commit or "HEAD~1"
                    for tf in text_files_sorted:
                        show_result = await session.execute(
                            f"git show {shlex.quote(fallback_ref)}:{shlex.quote(tf)} "
                            f"2>/dev/null",
                            cwd=workdir, timeout=30,
                        )
                        content = show_result.stdout
                        if not content:
                            continue
                        # Strip trailing newline so we don't add a
                        # spurious blank ``+`` line at end of file.
                        if content.endswith("\n"):
                            content = content[:-1]
                        lines = content.split("\n")
                        header = (
                            f"diff --git a/{tf} b/{tf}\n"
                            f"new file mode 100644\n"
                            f"--- /dev/null\n"
                            f"+++ b/{tf}\n"
                            f"@@ -0,0 +1,{len(lines)} @@\n"
                        )
                        body = "\n".join(f"+{line}" for line in lines)
                        parts.append(header + body)
                    if parts:
                        test_patch = "\n".join(parts) + "\n"
                        logger.info(
                            "Instance %s: git show fallback succeeded "
                            "(%d bytes, %d files)",
                            instance_id, len(test_patch), len(parts),
                        )

                # Binary archive (Bug C).  Empty bins → empty archive,
                # which is fine — the evaluator skips the upload step.
                binary_archive_b64 = ""
                binary_archive_error: str | None = None
                packed_files: list[str] = []
                if binary_files_sorted:
                    (binary_archive_b64, packed_files,
                     binary_archive_error) = await _build_binary_archive(
                        session, workdir, binary_files_sorted, instance_id,
                    )

                # If we have neither a text patch nor binary content,
                # something went wrong upstream — record an error so
                # the row isn't silently mis-used.
                if not test_patch and not binary_archive_b64:
                    logger.warning(
                        "Instance %s: could not generate patch nor binary archive "
                        "for %d test files",
                        instance_id, len(test_files),
                    )
                    raw["test_patch"] = ""
                    raw["test_files"] = test_files
                    raw["test_binary_archive_b64"] = ""
                    raw["test_binary_files"] = []
                    raw.setdefault("bug_b_overrides_agent_code", [])
                    raw["extract_error"] = "patch_generation_failed"
                    return raw

                raw["test_patch"] = test_patch
                raw["test_files"] = test_files
                raw["test_binary_archive_b64"] = binary_archive_b64
                raw["test_binary_files"] = packed_files
                if binary_archive_error:
                    raw["extract_error"] = binary_archive_error
                logger.info(
                    "Instance %s: extracted patch (text=%d bytes, %d text files, "
                    "binary=%d bytes b64 / %d files)",
                    instance_id, len(test_patch), len(text_files_sorted),
                    len(binary_archive_b64), len(packed_files),
                )

        except Exception as exc:
            logger.error(
                "Instance %s: extraction failed: %s", instance_id, exc,
            )
            raw["test_patch"] = ""
            raw["test_files"] = []
            raw["test_binary_archive_b64"] = ""
            raw["test_binary_files"] = []
            raw["bug_b_overrides_agent_code"] = []
            raw["extract_error"] = str(exc)

        # Delete image after use if requested (Docker only)
        if del_done_images and image and config.runtime.backend == "docker":
            await _delete_docker_image(image)

        return raw


def _build_run_dir(output_base: str) -> Path:
    """Build output directory: ``output_base/extract_patch_{YYYYMMDD_HHMMSS}``."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_base) / f"extract_patch_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def run_extraction(
    input_file: str,
    output_dir: str,
    config,
    max_concurrent: int = 20,
    instance_ids: list[str] | None = None,
    dry_run: bool = False,
    del_done_images: bool = False,
    extract_package_info: bool = True,
) -> None:
    """Run patch extraction for all instances in the input JSONL."""

    # Load input
    records = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d instances from %s", len(records), input_file)

    # Filter by instance IDs if specified
    if instance_ids:
        id_set = {iid.lower() for iid in instance_ids}
        records = [r for r in records if r.get("instance_id", "").lower() in id_set]
        logger.info("Filtered to %d instances", len(records))

    if dry_run:
        print(f"\nDry run — {len(records)} instances:")
        for r in records:
            iid = r.get("instance_id", "?")
            img = r.get("image", r.get("image_url", ""))[:60]
            print(f"  {iid}  image={img}")
        return

    # Create output directory
    run_dir = _build_run_dir(output_dir)
    results_file = run_dir / "results.jsonl"
    status_file = run_dir / "status.jsonl"
    log_file = run_dir / "extract.log"

    # Add file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    # Write run config
    run_config = {
        "input_file": input_file,
        "output_dir": str(run_dir),
        "max_concurrent": max_concurrent,
        "instance_count": len(records),
        "del_done_images": del_done_images,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Output: {run_dir}")
    print(f"Instances: {len(records)}")

    # Lock for thread-safe incremental writes
    write_lock = asyncio.Lock()
    counters = {"success": 0, "error": 0, "no_tests": 0, "done": 0}

    async def _process_and_write(raw: dict) -> dict:
        result = await extract_patch_for_instance(
            raw, config, semaphore,
            del_done_images=del_done_images,
            extract_package_info=extract_package_info,
        )

        # Determine status.  ``binary_too_large`` is recorded as a
        # warning (not a hard error) — the text patch is still usable,
        # only the binary fixtures are missing.
        instance_id = result.get("instance_id", "unknown")
        if isinstance(result, Exception):
            status = "exception"
            counters["error"] += 1
        elif result.get("extract_error") in (
            "binary_too_large", "tar_failed", "archive_missing",
        ):
            status = f"warn:{result['extract_error']}"
            if result.get("test_patch"):
                counters["success"] += 1
            else:
                counters["no_tests"] += 1
        elif result.get("extract_error"):
            if result["extract_error"] == "no_test_files":
                status = "no_test_files"
                counters["no_tests"] += 1
            else:
                status = f"error:{result['extract_error']}"
                counters["error"] += 1
        elif result.get("test_patch") or result.get("test_binary_archive_b64"):
            status = "success"
            counters["success"] += 1
        else:
            status = "no_patch"
            counters["no_tests"] += 1

        counters["done"] += 1

        # Write incrementally under lock
        async with write_lock:
            # Append to results.jsonl
            with open(results_file, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Append to status.jsonl
            status_record = {
                "instance_id": instance_id,
                "status": status,
                "n_test_files": len(result.get("test_files", [])),
                "n_binary_files": len(result.get("test_binary_files", [])),
                "patch_bytes": len(result.get("test_patch", "")),
                "binary_archive_bytes": len(result.get("test_binary_archive_b64", "")),
            }
            with open(status_file, "a") as f:
                f.write(json.dumps(status_record) + "\n")

        logger.info(
            "[%d/%d] %s: %s",
            counters["done"], len(records), instance_id, status,
        )
        return result

    # Process instances concurrently
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_process_and_write(r) for r in records]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Summary
    print(f"\nExtraction complete:")
    print(f"  Total:      {len(records)}")
    print(f"  Success:    {counters['success']}")
    print(f"  No tests:   {counters['no_tests']}")
    print(f"  Errors:     {counters['error']}")
    print(f"  Output dir: {run_dir}")
    print(f"  Results:    {results_file}")
    print(f"  Status:     {status_file}")
    print(f"  Log:        {log_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract unit-test patches from sandbox images for DeNovoSWE",
    )
    p.add_argument("--input", "-i", required=True, help="Input JSONL file")
    p.add_argument("--output", "-o", default="./results/denovoswe", help="Output directory (default: ./results/denovoswe)")
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/denovoswe.yaml",
        help="YAML config for runtime settings",
    )
    p.add_argument("--max-concurrent", type=int, default=20, help="Max parallel sandboxes")
    p.add_argument("--instance-ids", nargs="*", default=None, help="Filter specific instances")
    p.add_argument("--dry-run", action="store_true", help="List instances without processing")
    p.add_argument("--del-done-images", action="store_true", default=True, help="Delete docker image after each instance completes (default: on)")
    p.add_argument("--no-del-done-images", dest="del_done_images", action="store_false", help="Keep docker images after completion")
    p.add_argument(
        "--extract-package-info", action="store_true", default=True,
        help="Backfill pypi_name / import_names / package_setup_files / "
             "package_extract_error in instances that lack them. Runs the "
             "shared doc2repo package-name extractor inside the same "
             "sandbox session. Default: on (skips instances that already "
             "carry these fields).",
    )
    p.add_argument(
        "--no-extract-package-info", dest="extract_package_info",
        action="store_false",
        help="Disable the package-info backfill (saves ~3-15s per "
             "instance for datasets that already have these fields).",
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG level logging")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from awe_agent.core.config.loader import load_config
    config = load_config(args.config)

    await run_extraction(
        input_file=args.input,
        output_dir=args.output,
        config=config,
        max_concurrent=args.max_concurrent,
        instance_ids=args.instance_ids,
        dry_run=args.dry_run,
        del_done_images=args.del_done_images,
        extract_package_info=args.extract_package_info,
    )


if __name__ == "__main__":
    asyncio.run(main())
