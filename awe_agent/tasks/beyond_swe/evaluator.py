"""BeyondSWEEvaluator — evaluates all four BeyondSWE task types.

Task types and their evaluation strategies:

- **doc2repo**: Read a local test-suite ZIP, upload it to the container,
  unzip, run the eval script, and check results.
- **cross-repo / refactor / domain** (function-level): Apply the ``f2p_patch``
  (which introduces failing tests), upload ``f2p_script`` as a test file,
  then run all F2P + P2P tests together via the injected runner.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from awe_agent.core.eval.base import PatchTestEvaluator
from awe_agent.core.eval.setup import PreAgentSetup
from awe_agent.core.eval.utils import (
    parse_test_ids,
    run_tests_with_runner,
)
from awe_agent.core.task.types import EvalResult, Instance

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)


class BeyondSWEEvaluator(PatchTestEvaluator):
    """Evaluator for the BeyondSWE benchmark.

    Dispatches to the appropriate evaluation strategy based on
    ``instance.metadata["task_type"]``:

    - ``doc2repo`` → :meth:`_eval_repo_level`
    - ``cross-repo`` / ``refactor`` / ``domain`` → :meth:`_eval_func_level`

    Example::

        evaluator = BeyondSWEEvaluator(timeout=1800)
        result = await evaluator.evaluate(instance, patch, runtime)
    """

    async def pre_patch_setup(self, instance, session) -> None:
        """Run setup commands + remove future commits before patch application."""
        setup = PreAgentSetup(session, instance.workdir)
        await setup.prepare(instance)

    async def run_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Dispatch to the appropriate evaluation strategy."""
        task_type = instance.metadata.get("task_type", "domainfix")

        if task_type == "doc2repo":
            return await self._eval_repo_level(instance, session)
        return await self._eval_func_level(instance, session)

    # ── Function-level evaluation (cross-repo, refactor, domain) ────────

    async def _eval_func_level(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Apply ``f2p_patch``, upload ``f2p_script`` as test file, run merged F2P+P2P.

        Changes from the original:
        - f2p_patch failure → immediate ``accepted=False`` (Issue 5)
        - f2p_script uploaded as test file, not executed (Issue 2)
        - F2P + P2P merged into one ``run_tests_with_runner`` call (Issue 3)
        """
        workdir = instance.workdir

        # ── 1. Apply f2p_patch → fail immediately if it doesn't apply ──
        f2p_patch = instance.metadata.get("f2p_patch", "")
        if f2p_patch:
            apply_result = await session.apply_patch(workdir, f2p_patch)
            if not apply_result.success:
                logger.error(
                    "f2p_patch failed for %s: %s",
                    instance.id,
                    apply_result.stderr[:200],
                )
                return EvalResult(
                    accepted=False,
                    score=0.0,
                    details={
                        "error": "f2p_patch_failed",
                        "stderr": apply_result.stderr[-2000:],
                    },
                )

        # ── 2. Upload f2p_script as a test file (NOT execute it) ────────
        f2p_script = instance.metadata.get("f2p_script", "")
        if f2p_script:
            await session.upload_file(
                f"{workdir}/test_fail_to_pass.py", f2p_script.encode(),
            )

        # ── 3. Merge F2P + P2P test IDs and run together ───────────────
        f2p_ids = parse_test_ids(instance.metadata.get("FAIL_TO_PASS"))
        p2p_ids = parse_test_ids(instance.metadata.get("PASS_TO_PASS"))
        all_tests = f2p_ids + p2p_ids

        if not all_tests:
            logger.warning("Instance %s has no F2P/P2P test IDs", instance.id)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": "no_test_ids"},
            )

        all_passed, raw_output, details = await run_tests_with_runner(
            session, workdir, all_tests, timeout=self._timeout,
        )

        details["f2p_count"] = len(f2p_ids)
        details["p2p_count"] = len(p2p_ids)
        details["output"] = raw_output[-2000:]

        return EvalResult(
            accepted=all_passed,
            score=1.0 if all_passed else 0.0,
            details=details,
        )

    # ── Repo-level evaluation (doc2repo) ────────────────────────────────

    async def _eval_repo_level(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Read local test-suite ZIP, upload, unzip, and run eval script.

        Changes from the original:
        - ``test_suite`` is a filename, not script content (Issue 4)
        - Reads ZIP from ``os.path.join(test_suite_path, test_suite)``
        - Uploads and unzips in container
        - Runs ``realswe_eval_script.py`` (included in the ZIP)
        """
        workdir = instance.workdir
        test_suite_name = instance.metadata.get("test_suite", "")
        test_suite_path = instance.metadata.get("test_suite_path", "")
        test_suite_num = instance.metadata.get("test_suite_num", 0)

        # ── 1. pip install -e . ─────────────────────────────────────────
        install_result = await session.execute(
            f"cd {workdir} && pip install -e .", timeout=300,
        )
        if not install_result.success:
            return EvalResult(
                accepted=False,
                score=0.0,
                details={
                    "error": "pip_install_failed",
                    "stderr": install_result.stderr[-2000:],
                },
            )

        if not test_suite_name or not test_suite_path:
            raise ValueError(
                f"doc2repo instance {instance.id} missing test_suite or test_suite_path"
            )

        # ── 2. Read local ZIP file ──────────────────────────────────────
        local_path = os.path.join(test_suite_path, test_suite_name)
        try:
            with open(local_path, "rb") as f:
                zip_bytes = f.read()
        except FileNotFoundError:
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": f"test_suite_zip_not_found: {local_path}"},
            )

        # ── 3. Upload ZIP + unzip in container ─────────────────────────
        await session.upload_file("/tmp/_awe_test_suite.zip", zip_bytes)
        await session.execute(
            f"cd {workdir} && unzip -o /tmp/_awe_test_suite.zip",
            timeout=120,
        )

        # ── 4. Execute the eval script from the ZIP ────────────────────
        result = await session.execute(
            f"cd {workdir} && python realswe_eval_script.py",
            timeout=self._timeout,
        )

        # ── 5. Check <pytest>true</pytest> marker ──────────────────────
        if "<pytest>true</pytest>" in result.output:
            return EvalResult(
                accepted=True,
                score=1.0,
                details={
                    "source": "pytest_marker",
                    "exit_code": result.exit_code,
                    "output": result.output[-2000:],
                },
            )

        # ── 6. Marker not found → fail ────────────────────────────────
        return EvalResult(
            accepted=False,
            score=0.0,
            details={
                "source": "pytest_marker_not_found",
                "test_suite_num": test_suite_num,
                "exit_code": result.exit_code,
                "output": result.output[-2000:],
            },
        )
