"""BeyondSWEEvaluator — evaluates all four BeyondSWE task types.

Task types and their evaluation strategies:

- **doc2repo**: Upload the provided test suite script, run it against the
  agent's generated repository, and check that all tests pass.
- **cross-repo / refactor / domain** (function-level): Apply the ``f2p_patch``
  (which introduces failing tests), optionally run ``f2p_script``, then
  evaluate via the standard FAIL_TO_PASS / PASS_TO_PASS flow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from awe_agent.core.eval.base import PatchTestEvaluator
from awe_agent.core.eval.utils import (
    build_pytest_command,
    parse_pytest_summary,
    run_f2p_p2p_eval,
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

    async def run_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Dispatch to the appropriate evaluation strategy."""
        task_type = instance.metadata.get("task_type", "domain")

        if task_type == "doc2repo":
            return await self._eval_repo_level(instance, session)
        return await self._eval_func_level(instance, session)

    # ── Function-level evaluation (cross-repo, refactor, domain) ────────

    async def _eval_func_level(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Apply ``f2p_patch``, run ``f2p_script``, then evaluate F2P / P2P.

        The ``f2p_patch`` introduces the test cases that should fail before
        the fix.  It is applied **after** test-file restoration (handled by
        the base class) so the new tests survive.
        """
        # ── Apply f2p_patch (test patch) ───────────────────────────────
        f2p_patch = instance.metadata.get("f2p_patch", "")
        if f2p_patch:
            apply_result = await session.apply_patch(
                instance.workdir, f2p_patch,
            )
            if not apply_result.success:
                logger.warning(
                    "f2p_patch failed for %s: %s",
                    instance.id,
                    apply_result.stderr[:200],
                )

        # ── Execute f2p_script (optional test setup) ───────────────────
        f2p_script = instance.metadata.get("f2p_script", "")
        if f2p_script:
            await session.upload_file(
                "/tmp/_awe_f2p_setup.py", f2p_script.encode(),
            )
            await session.execute(
                f"cd {instance.workdir} && python /tmp/_awe_f2p_setup.py",
                timeout=120,
            )

        # ── Standard F2P / P2P evaluation ──────────────────────────────
        return await run_f2p_p2p_eval(
            session, instance, timeout=self._timeout,
        )

    # ── Repo-level evaluation (doc2repo) ────────────────────────────────

    async def _eval_repo_level(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Run the provided test suite against the agent's generated repo.

        For ``doc2repo`` tasks the agent builds a repository from a
        specification document.  The instance provides a ``test_suite``
        (script content) that validates the implementation.
        """
        test_suite = instance.metadata.get("test_suite", "")
        test_suite_path = instance.metadata.get(
            "test_suite_path", "/tmp/_awe_test_suite.py",
        )

        if test_suite:
            # Upload the test suite provided by the dataset
            await session.upload_file(
                test_suite_path, test_suite.encode(),
            )
            cmd = build_pytest_command(
                [test_suite_path], instance.workdir,
            )
        else:
            # Fallback: discover and run all tests in the workspace
            cmd = build_pytest_command([], instance.workdir)
            cmd = (
                f"cd {instance.workdir} && "
                f"python -m pytest --tb=short --no-header -q"
            )

        result = await session.execute(
            cmd, cwd=instance.workdir, timeout=self._timeout,
        )

        summary = parse_pytest_summary(result.output)
        accepted = summary.all_passed

        return EvalResult(
            accepted=accepted,
            score=1.0 if accepted else 0.0,
            details={
                "passed": summary.passed,
                "failed": summary.failed,
                "errors": summary.errors,
                "skipped": summary.skipped,
                "exit_code": result.exit_code,
                "output": result.output[-2000:],
            },
        )
