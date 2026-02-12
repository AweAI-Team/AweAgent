"""PatchTestEvaluator — base class for the apply-patch-then-test evaluation pattern.

Most code-task evaluators follow the same lifecycle:

    1. Create a fresh, isolated container.
    2. Checkout the base commit.
    3. Apply the agent's patch.
    4. Restore test files (prevent the agent from tampering with tests).
    5. Run task-specific setup (e.g., apply ``f2p_patch``, install extras).
    6. Execute evaluation tests.
    7. Interpret the output into an ``EvalResult``.

``PatchTestEvaluator`` handles steps 1-4 so that subclasses only need to
implement the task-specific logic in ``run_tests``.
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import TYPE_CHECKING

from awe_agent.core.eval.utils import restore_test_files
from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.protocol import Evaluator
from awe_agent.core.task.types import EvalResult, Instance

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)


class PatchTestEvaluator(Evaluator):
    """Template evaluator for the common *apply-patch → run-tests* workflow.

    Subclasses **must** implement:

    - :meth:`run_tests` — execute the evaluation tests and return an
      ``EvalResult``.  The session is already set up (patch applied, test
      files restored).

    Subclasses **may** override:

    - :meth:`get_setup_commands` — extra shell commands to run after patch
      application (e.g., apply ``f2p_patch``, install dependencies).

    Example::

        class MyEvaluator(PatchTestEvaluator):
            async def run_tests(self, instance, session):
                result = await session.execute("cd /testbed && pytest tests/")
                accepted = result.exit_code == 0
                return EvalResult(accepted=accepted, score=float(accepted))
    """

    def __init__(
        self,
        timeout: int = 3600,
        restore_tests: bool = True,
    ) -> None:
        self._timeout = timeout
        self._restore_tests = restore_tests

    # ── Evaluator Protocol ──────────────────────────────────────────────

    async def evaluate(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        """Full evaluation lifecycle in an isolated container."""
        start = time.monotonic()
        image = instance.image

        try:
            async with runtime.session(image) as session:
                # 1 ── Checkout base commit
                if instance.base_commit:
                    await session.execute(
                        f"cd {instance.workdir} && "
                        f"git checkout {instance.base_commit}",
                    )

                # 2 ── Apply the agent's patch
                if patch and patch.strip():
                    apply_result = await session.apply_patch(
                        instance.workdir, patch,
                    )
                    if not apply_result.success:
                        return EvalResult(
                            accepted=False,
                            score=0.0,
                            details={
                                "error": "patch_apply_failed",
                                "stderr": apply_result.stderr[-2000:],
                            },
                            duration=time.monotonic() - start,
                        )

                # 3 ── Restore test files
                if self._restore_tests:
                    await restore_test_files(session, instance.workdir)

                # 4 ── Task-specific setup
                for cmd in self.get_setup_commands(instance):
                    await session.execute(cmd, timeout=300)

                # 5 ── Run tests (subclass implementation)
                eval_result = await self.run_tests(instance, session)
                eval_result.duration = time.monotonic() - start
                return eval_result

        except Exception as exc:
            logger.error("Evaluation failed for %s: %s", instance.id, exc)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": str(exc)},
                duration=time.monotonic() - start,
            )

    # ── Template methods ────────────────────────────────────────────────

    @abstractmethod
    async def run_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Execute evaluation tests and interpret the results.

        Called after the container has been fully prepared:

        - Base commit checked out.
        - Agent's patch applied.
        - Test files restored (if configured).
        - Setup commands executed.

        Args:
            instance: The task instance with metadata (test IDs, etc.).
            session: An isolated ``RuntimeSession`` ready for commands.

        Returns:
            ``EvalResult`` with ``accepted``, ``score``, and ``details``.
        """
        ...

    def get_setup_commands(self, instance: Instance) -> list[str]:
        """Optional shell commands to run after patch application.

        Override to apply additional patches (e.g., ``f2p_patch``), install
        extra dependencies, or perform other setup before the test run.

        These commands execute **after** test-file restoration, so changes
        they make to test files will survive.
        """
        return []
