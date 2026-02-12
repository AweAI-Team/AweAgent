"""Isolated evaluator — runs evaluation in a fresh container.

Key design: the evaluation container is completely separate from the agent's
working container. This prevents information leakage (e.g., agent artifacts
affecting test results).
"""

from __future__ import annotations

import logging
import time

from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.protocol import Evaluator
from awe_agent.core.task.types import EvalResult, Instance

logger = logging.getLogger(__name__)


class IsolatedEvaluator(Evaluator):
    """Evaluates patches in fresh, isolated containers.

    Usage:
        evaluator = IsolatedEvaluator(eval_script="cd /testbed && pytest tests/")
        result = await evaluator.evaluate(instance, patch, runtime)
    """

    def __init__(
        self,
        eval_script: str = "cd /testbed && pytest tests/ -x",
        setup_commands: list[str] | None = None,
        timeout: int = 3600,
    ) -> None:
        self._eval_script = eval_script
        self._setup_commands = setup_commands or []
        self._timeout = timeout

    async def evaluate(
        self,
        instance: Instance,
        patch: str,
        runtime: Runtime,
    ) -> EvalResult:
        """Evaluate a patch in a fresh container."""
        start = time.monotonic()
        image = instance.image

        try:
            async with runtime.session(image) as session:
                # Checkout base commit
                if instance.base_commit:
                    await session.execute(
                        f"cd {instance.workdir} && git checkout {instance.base_commit}"
                    )

                # Apply patch
                apply_result = await session.apply_patch(instance.workdir, patch)
                if not apply_result.success:
                    return EvalResult(
                        accepted=False,
                        score=0.0,
                        details={"error": "patch_apply_failed", "stderr": apply_result.stderr},
                        duration=time.monotonic() - start,
                    )

                # Setup commands
                for cmd in self._setup_commands:
                    await session.execute(cmd, timeout=300)

                # Run evaluation
                eval_result = await session.execute(
                    self._eval_script,
                    cwd=instance.workdir,
                    timeout=self._timeout,
                )

                accepted = eval_result.exit_code == 0
                return EvalResult(
                    accepted=accepted,
                    score=1.0 if accepted else 0.0,
                    details={
                        "exit_code": eval_result.exit_code,
                        "stdout": eval_result.stdout[-2000:],  # Truncate for storage
                        "stderr": eval_result.stderr[-2000:],
                    },
                    duration=time.monotonic() - start,
                )

        except Exception as e:
            logger.error("Evaluation failed for %s: %s", instance.id, e)
            return EvalResult(
                accepted=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.monotonic() - start,
            )
