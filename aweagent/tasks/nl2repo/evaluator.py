"""NL2RepoEvaluator — evaluates an NL2Repo agent submission.

Faithful port of the original NL2RepoBench post-processing pipeline
(``openhands/post_processor.post_process_task`` →
``run_test_commands`` → ``analyze_pytest_results``), with the same
tar-based workspace handover that swalm-core's ``run_nl2repo_task``
uses.  Crucially, this evaluator does **not** use ``git`` anywhere —
some sandbox runtimes don't ship it.

Pipeline (aligned with the official flow):

1. The agent session (run by the ``TaskRunner``) tars its workspace
   into ``workspace.tar.gz`` and hands the bytes to this evaluator
   through ``instance.metadata["_agent_artifact"]``.

2. Evaluator boots a fresh container from the per-instance evaluation
   image (``Instance.metadata["eval_image"]``), which already contains
   the project's source and golden tests under ``{workdir}``.

3. Upload the tarball to ``/tmp/workspace.tar.gz`` and extract it into
   ``/tmp`` — producing ``/tmp/workspace/...`` (matches the official
   ``COPY workspace /workspace`` layout and swalm-core's
   ``tar -xzf /tmp/workspace.tar.gz``).

4. Strip package-management files (``setup.py`` / ``pyproject.toml`` /
   ``requirements*.txt`` / ...) from ``/tmp/workspace`` (recursively).
   Same rationale: the base image ships the canonical packaging files
   so the test environment has the expected dependencies.  This mirrors
   ``post_processor.remove_package_files``.

5. Strip agent-written variants of the golden test paths
   (``py_test_file_list``) from ``/tmp/workspace`` so that when we
   overlay onto the container's ``{workdir}`` the base image's golden
   tests survive byte-identical.  This is the direct analogue of
   ``post_processor.remove_test_files`` + ``COPY``: the agent cannot
   ship an adversarial ``tests/test_evil.py``.

6. Overlay ``/tmp/workspace`` onto ``{workdir}`` with ``cp -r``.  Files
   only present in the agent's tar are added, files also present in the
   base image are overwritten by the agent's version, and files
   deleted in steps 4/5 keep the base image's golden copy.

7. Run each shell command in ``test_shell`` sequentially inside the
   container under ``{workdir}`` with
   ``PYTHONPATH={workdir}:$PYTHONPATH`` (matches the official
   Dockerfile ``ENV PYTHONPATH=/workspace:$PYTHONPATH``).  Failures are
   caught per-command and reported as synthetic ``exit_code=-1``
   entries so transient docker/portal errors don't abort the whole
   evaluation.

8. Score with the byte-for-byte port of ``analyze_pytest_results``:
   regex-sum ``(\\d+) passed`` / ``(\\d+) failed`` / ``(\\d+) error``
   across every pytest command's output, compute
   ``success_rate = min(passed/total, 1)``.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shlex
import time
from typing import TYPE_CHECKING

from aweagent.core.task.protocol import Evaluator
from aweagent.core.task.types import EvalResult, Instance
from aweagent.tasks.nl2repo.task import _test_cleanup_relpaths

if TYPE_CHECKING:
    from aweagent.core.runtime.protocol import Runtime, RuntimeSession

logger = logging.getLogger(__name__)

# Per-instance evaluation timeout — generous since some NL2Repo projects
# have multi-minute test suites and ``test_shell`` may include
# ``pip install -e .``.
_NL2REPO_TIMEOUT = 1800

# Ported verbatim from ``post_processor.remove_package_files``.  These are
# stripped before running the test commands so that the project's
# pre-installed dependencies in the base image are used during evaluation.
_PACKAGE_FILES_TO_REMOVE: list[str] = [
    "setup.py",
    "pyproject.toml",
    "setup.cfg",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "tox.ini",
    "pytest.ini",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "environment.yml",
    "conda-env.yaml",
    "manifest.in",
    "MANIFEST.in",
]

# Regexes for parsing pytest summary lines.
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) error")

# Matches a pytest final summary line — the line that contains test counts
# AND a timing marker ("in X.XXs") AND is wrapped in "=" decorations.
# Example: ``======= 21 passed, 163 failed in 80.82s =======``
# This distinguishes the top-level summary from sub-process summaries
# that pytest plugin test suites (e.g. pytest-cov) emit during test runs.
_PYTEST_SUMMARY_RE = re.compile(
    r"=+\s*(?:.*\d+\s+(?:passed|failed|error).*)\s+in\s+[\d.]+s.*=+",
)

# Path inside the eval container where the agent's tar.gz is uploaded and
# extracted.  Constants match swalm-core's ``run_nl2repo_task`` so the two
# implementations are byte-level interchangeable.
_ARCHIVE_PATH = "/tmp/workspace.tar.gz"
_EXTRACT_PARENT = "/tmp"
_EXTRACT_DIR = "/tmp/workspace"


class NL2RepoEvaluator(Evaluator):
    """Evaluator for the NL2RepoBench benchmark.

    Unlike ``PatchTestEvaluator`` subclasses, this evaluator consumes a
    tar.gz of the agent's workspace (handed over via
    ``instance.metadata["_agent_artifact"]``) rather than a git patch.
    That lets it run in sandboxes with no ``git`` binary and keeps the
    handover byte-for-byte identical to the official NL2RepoBench
    pipeline.

    Example::

        evaluator = NL2RepoEvaluator(timeout=1800)
        result = await evaluator.evaluate(instance, patch, runtime)
    """

    def __init__(self, timeout: int = _NL2REPO_TIMEOUT) -> None:
        self._timeout = timeout

    # ── Evaluator Protocol ─────────────────────────────────────────────

    async def evaluate(
        self,
        instance: Instance,
        patch: str,  # unused; present for Evaluator protocol compatibility
        runtime: Runtime,
    ) -> EvalResult:
        """Run the full NL2Repo evaluation in a fresh per-instance container.

        The ``patch`` argument is ignored — NL2Repo submissions are the
        whole workspace captured as a tarball in
        ``instance.metadata["_agent_artifact"]``.
        """
        start = time.monotonic()

        # Route to the authoritative per-instance evaluation image.
        # ``NL2RepoTask`` stores it on ``metadata["eval_image"]``; when
        # an ``AGENT_RUN_DOCKER`` override is in play ``instance.image``
        # (the agent-side image) may differ.
        eval_image = instance.metadata.get("eval_image") or instance.image
        if eval_image != instance.image:
            instance = dataclasses.replace(instance, image=eval_image)

        artifact: bytes | None = instance.metadata.get("_agent_artifact")
        if not artifact:
            logger.error(
                "No workspace artifact for %s — cannot evaluate", instance.id,
            )
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": "missing_agent_artifact"},
                duration=time.monotonic() - start,
            )

        try:
            async with runtime.session(eval_image) as session:
                return await self._evaluate_in_session(instance, artifact, session, start)
        except Exception as exc:
            logger.exception("Evaluation failed for %s", instance.id)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": str(exc)},
                duration=time.monotonic() - start,
            )

    # ── Internals ──────────────────────────────────────────────────────

    async def _evaluate_in_session(
        self,
        instance: Instance,
        artifact: bytes,
        session: RuntimeSession,
        start: float,
    ) -> EvalResult:
        workdir = instance.workdir
        test_shell: list[str] = instance.metadata.get("test_shell", []) or []
        py_test_file_list: list[str] = instance.metadata.get(
            "py_test_file_list", [],
        ) or []
        test_case_count: int = int(instance.metadata.get("test_case_count", 0) or 0)

        if not test_shell:
            logger.warning("Instance %s has no test_shell commands", instance.id)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": "no_test_shell"},
                duration=time.monotonic() - start,
            )

        # 1. Upload the tarball + extract into /tmp/workspace.
        await self._stage_artifact(session, artifact, workdir)

        # 2. Strip package-management files (recursive: these can live at
        #    any nesting level inside the agent workspace).
        await self._remove_from_staging(
            session, _PACKAGE_FILES_TO_REMOVE, recursive_basenames=True,
        )

        # 3. Strip agent-written test paths so the base image's golden
        #    tests survive the subsequent ``cp -r`` overlay.  Besides the
        #    explicit ``verify_files`` list, strip safe test-looking
        #    pytest targets parsed from ``verify_cmd``.  This covers cases
        #    such as ``test_schema.py`` or ``cerberus/benchmarks/test_*.py``
        #    without deleting broad implementation roots like ``src``/``doc``.
        staging_test_paths = _test_cleanup_relpaths(
            test_shell=test_shell,
            py_test_file_list=py_test_file_list,
            workdir=workdir,
            include_broad_pytest_targets=False,
        )
        await self._remove_from_staging(
            session, staging_test_paths, recursive_basenames=False,
        )

        # 3a. Catch-all: remove any remaining pytest-discoverable test
        #     files from staging.  This handles agent-written tests that
        #     fall outside the ``verify_files`` paths (e.g. bare
        #     ``pytest`` with no path argument discovers the whole tree).
        #     Golden tests live in the base image and survive the overlay.
        await self._remove_pytest_discoverable_from_staging(session)

        # 4. Overlay onto the container's {workdir}.
        await self._overlay_staging(session, workdir)

        # 5. Resolve PYTHONPATH once to mirror Dockerfile ``ENV``
        #    semantics (expand against the base image env once, then
        #    pass the literal through per-exec ``env=``).
        probe = await session.execute(
            'printf %s "${PYTHONPATH-}"', cwd=workdir, timeout=60,
        )
        base_pythonpath = (probe.output or "").strip()
        pythonpath_value = (
            f"{workdir}:{base_pythonpath}" if base_pythonpath else workdir
        )
        exec_env = {"PYTHONPATH": pythonpath_value}

        # 6. Run test_shell commands sequentially
        #    (= post_processor.run_test_commands).
        timeout = min(self._timeout, _NL2REPO_TIMEOUT)
        command_results: list[dict[str, object]] = []
        last_exit_code = 0
        for idx, command in enumerate(test_shell):
            logger.info(
                "[%s] Executing command %d/%d: %s",
                instance.id, idx + 1, len(test_shell), command,
            )
            try:
                result = await session.execute(
                    command, cwd=workdir, timeout=timeout, env=exec_env,
                )
                command_results.append({
                    "command": command,
                    "exit_code": result.exit_code,
                    "output": result.output,
                })
                last_exit_code = result.exit_code
            except Exception as exc:
                logger.error("[%s] Command execution failed: %s", instance.id, exc)
                command_results.append({
                    "command": command,
                    "exit_code": -1,
                    "output": str(exc),
                })
                last_exit_code = -1

        # 7. Score (byte-for-byte port of analyze_pytest_results).
        pytest_results = self._analyze_pytest_results(command_results, test_case_count)

        # 7a. Diagnostic only.  The official NL2RepoBench implementation
        #     does not reject or zero scores when pytest reports a count
        #     different from ``test_cases_num``; it only uses the count
        #     as the pass-rate denominator.  Keep the mismatch visible for
        #     audits, but do not put it in ``details["error"]`` because the
        #     runner treats that as an evaluator failure.
        actual_total = (
            pytest_results["passed"]
            + pytest_results["failed"]
            + pytest_results["errors"]
        )
        count_mismatch = (
            test_case_count > 0 and actual_total != test_case_count
        )
        # eval > official → evaluation picked up extra tests (agent-written
        # test pollution or nested subprocess summary accumulation).
        test_count_error = (
            test_case_count > 0 and actual_total > test_case_count
        )
        if count_mismatch:
            logger.warning(
                "[%s] Test count mismatch: expected %d, got %d "
                "(passed=%d failed=%d errors=%d)",
                instance.id, test_case_count, actual_total,
                pytest_results["passed"],
                pytest_results["failed"],
                pytest_results["errors"],
            )

        truncated_results = [
            {
                "command": cr["command"],
                "exit_code": cr["exit_code"],
                "output": str(cr["output"])[-2000:],
            }
            for cr in command_results
        ]

        # NL2RepoBench does not have a binary "accepted" concept — it just
        # reports passed / total / pass_rate.  For runner compatibility we
        # mark a run accepted when at least the expected number of tests
        # pass and pytest reports no failures/errors.  Count mismatches are
        # recorded above, but they are not evaluator errors in the official
        # benchmark.
        accepted = (
            test_case_count > 0
            and pytest_results["passed"] >= test_case_count
            and pytest_results["failed"] == 0
            and pytest_results["errors"] == 0
        )

        details: dict[str, object] = {
            "test_case_count": test_case_count,
            "actual_total": actual_total,
            "count_mismatch": count_mismatch,
            "test_count_error": test_count_error,
            "passed": pytest_results["passed"],
            "failed": pytest_results["failed"],
            "errors": pytest_results["errors"],
            "total": pytest_results["total"],
            "success_rate": pytest_results["success_rate"],
            "last_exit_code": last_exit_code,
            "command_results": truncated_results,
        }
        if count_mismatch:
            details["count_mismatch_detail"] = (
                f"expected {test_case_count} tests (from test_cases_num), "
                f"pytest ran {actual_total} "
                f"(passed={pytest_results['passed']}, "
                f"failed={pytest_results['failed']}, "
                f"errors={pytest_results['errors']})"
            )

        return EvalResult(
            accepted=accepted,
            score=pytest_results["success_rate"],
            details=details,
            duration=time.monotonic() - start,
        )

    async def _stage_artifact(
        self,
        session: RuntimeSession,
        artifact: bytes,
        workdir: str,
    ) -> None:
        """Upload + extract the agent tarball into ``/tmp/workspace``.

        The tar produced by ``NL2RepoTask.collect_artifact`` has the
        workdir basename as its top-level entry, so extracting with
        ``-C /tmp`` gives ``/tmp/workspace`` (or ``/tmp/<basename>``).
        We normalise to ``/tmp/workspace`` by renaming when the
        basename differs so downstream paths can be constants.
        """
        await session.upload_file(_ARCHIVE_PATH, artifact)

        # Clean any previous staging dir — defence in depth against a
        # recycled container.
        await session.execute(
            f"rm -rf {shlex.quote(_EXTRACT_DIR)}",
            cwd=_EXTRACT_PARENT, timeout=60,
        )
        await session.execute(
            f"tar -xzf {shlex.quote(_ARCHIVE_PATH)} -C {shlex.quote(_EXTRACT_PARENT)}",
            cwd=_EXTRACT_PARENT, timeout=600,
        )

        # Normalise: the tar's top-level dir is the workdir basename.
        # If the agent used a non-``workspace`` workdir, rename so later
        # paths stay canonical.
        basename = (workdir.rstrip("/").rsplit("/", 1)[-1]) or "workspace"
        if basename != "workspace":
            extracted = f"{_EXTRACT_PARENT}/{basename}"
            await session.execute(
                f"if [ -d {shlex.quote(extracted)} ]; then "
                f"mv {shlex.quote(extracted)} {shlex.quote(_EXTRACT_DIR)}; "
                f"fi",
                cwd=_EXTRACT_PARENT, timeout=60,
            )

    async def _remove_from_staging(
        self,
        session: RuntimeSession,
        entries: list[str],
        recursive_basenames: bool,
    ) -> None:
        """Strip paths (or named files) from the staged workspace.

        ``recursive_basenames=False`` treats each entry as a path
        relative to the staged workspace root (``/tmp/workspace``) and
        deletes it with ``rm -rf`` — matching
        ``post_processor.remove_test_files`` semantics for
        ``py_test_file_list``.

        ``recursive_basenames=True`` treats each entry as a basename and
        removes every occurrence anywhere under the staged workspace
        via ``find``.  This matches
        ``post_processor.remove_package_files``, which walks
        ``os.walk(workspace_path)`` and removes matching basenames at
        any depth.
        """
        if not entries:
            return
        if recursive_basenames:
            names = " -o ".join(f"-name {shlex.quote(n)}" for n in entries)
            cmd = (
                f"find {shlex.quote(_EXTRACT_DIR)} "
                f"\\( -type f -o -type l \\) "
                f"\\( {names} \\) -delete 2>/dev/null || true"
            )
            await session.execute(cmd, cwd=_EXTRACT_PARENT, timeout=120)
            return
        for entry in entries:
            target = f"{_EXTRACT_DIR.rstrip('/')}/{entry.lstrip('/')}"
            await session.execute(
                f"rm -rf {shlex.quote(target)}",
                cwd=_EXTRACT_PARENT, timeout=60,
            )

    async def _overlay_staging(
        self,
        session: RuntimeSession,
        workdir: str,
    ) -> None:
        """Overlay the staged workspace onto the container's ``workdir``.

        Equivalent to the official ``COPY workspace /workspace`` step:
        files in the staging dir win, files only in the container's
        ``workdir`` are preserved.  The trailing ``/.`` on the source
        makes ``cp -r`` copy the contents (including dotfiles) rather
        than the ``/tmp/workspace`` directory itself.
        """
        await session.execute(
            f"mkdir -p {shlex.quote(workdir)}",
            cwd="/", timeout=30,
        )
        await session.execute(
            f"cp -a {shlex.quote(_EXTRACT_DIR)}/. {shlex.quote(workdir)}/",
            cwd=workdir, timeout=600,
        )

    async def _remove_pytest_discoverable_from_staging(
        self,
        session: RuntimeSession,
    ) -> None:
        """Remove all pytest-discoverable test files from the staged workspace.

        pytest auto-discovers ``test_*.py``, ``*_test.py``, and loads
        ``conftest.py`` at every level.  Any such files in the agent's
        tarball are agent-written (golden tests live in the base image)
        and must be stripped so they don't inflate the evaluation score.
        """
        await session.execute(
            f"find {shlex.quote(_EXTRACT_DIR)} "
            r"\( -name 'test_*.py' -o -name '*_test.py' -o -name 'conftest.py' \) "
            "-type f -delete 2>/dev/null || true",
            cwd=_EXTRACT_PARENT,
            timeout=120,
        )

    # ── Helpers (verbatim ports of post_processor functions) ───────────

    @staticmethod
    def _analyze_pytest_results(
        command_results: list[dict[str, object]],
        total_test_cases: int,
    ) -> dict[str, object]:
        """Parse pytest results from the **final summary line** of each command.

        Only the last line matching the pytest summary pattern (``=== N
        passed, M failed in X.XXs ===``) is parsed per command.  This
        avoids double-counting when the test suite spawns sub-processes
        that each emit their own summary (common in pytest-plugin
        projects like pytest-cov).

        Falls back to the legacy accumulating parser when no summary
        line is detected (e.g. pytest crashed before printing one).
        """
        pytest_results: dict[str, object] = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "total": total_test_cases,
            "success_rate": 0.0,
        }

        for result in command_results:
            command = str(result.get("command", ""))
            output = str(result.get("output", ""))
            if "pytest" not in command.lower():
                continue

            # Find the last pytest summary line (top-level, not sub-process)
            summary_line = None
            for line in reversed(output.split("\n")):
                if _PYTEST_SUMMARY_RE.search(line):
                    summary_line = line
                    break

            if summary_line is not None:
                m = _PASSED_RE.search(summary_line)
                if m:
                    pytest_results["passed"] += int(m.group(1))
                m = _FAILED_RE.search(summary_line)
                if m:
                    pytest_results["failed"] += int(m.group(1))
                m = _ERROR_RE.search(summary_line)
                if m:
                    pytest_results["errors"] += int(m.group(1))
            else:
                # Fallback: no summary line found — accumulate all matches.
                # This handles edge cases where pytest output is truncated
                # or the summary format is non-standard.
                for line in output.split("\n"):
                    m = _PASSED_RE.search(line)
                    if m:
                        pytest_results["passed"] += int(m.group(1))
                    m = _FAILED_RE.search(line)
                    if m:
                        pytest_results["failed"] += int(m.group(1))
                    m = _ERROR_RE.search(line)
                    if m:
                        pytest_results["errors"] += int(m.group(1))

        if total_test_cases > 0:
            pytest_results["success_rate"] = min(
                pytest_results["passed"] / total_test_cases, 1,
            )

        return pytest_results
