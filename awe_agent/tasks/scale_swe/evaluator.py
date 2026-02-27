"""ScaleSWEEvaluator — evaluates ScaleSWE instances.

Reuses the BeyondSWE F2P+P2P test logic (``_eval_beyondswe``) directly,
skipping the doc2repo task-type dispatch since ScaleSWE is always
issue-resolving.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from awe_agent.core.task.types import EvalResult, Instance
from awe_agent.tasks.beyond_swe.evaluator import BeyondSWEEvaluator

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession


class ScaleSWEEvaluator(BeyondSWEEvaluator):
    """ScaleSWE evaluator: reuses BeyondSWE's F2P+P2P test logic."""

    async def run_tests(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> EvalResult:
        """Run F2P+P2P tests directly (no task_type dispatch)."""
        return await self._eval_beyondswe(instance, session)
