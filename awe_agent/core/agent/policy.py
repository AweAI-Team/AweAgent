"""Loop policies — composable lifecycle hooks around a single agent rollout.

A policy lets a scaffold shape the loop's lifecycle without subclassing the
loop or touching the task runner. It answers two questions:

- :meth:`LoopPolicy.should_retry` — after a rollout finishes, discard it and
  run a fresh one?
- :meth:`LoopPolicy.finalize` — transform the kept rollout before returning it.

The base class is a no-op, so an :class:`AgentLoop` given no policy runs exactly
one rollout and returns it unchanged.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awe_agent.core.agent.context import AgentContext
    from awe_agent.core.agent.loop import AgentLoop, AgentResult


class LoopPolicy(ABC):
    """Lifecycle hooks an :class:`AgentLoop` consults around each rollout."""

    def should_retry(
        self,
        attempt: int,
        result: AgentResult,
        ctx: AgentContext,
    ) -> bool:
        """Return ``True`` to discard ``result`` and run another fresh rollout.

        ``attempt`` is the zero-based index of the rollout that just finished.
        """
        return False

    async def finalize(
        self,
        loop: AgentLoop,
        result: AgentResult,
        ctx: AgentContext,
    ) -> AgentResult:
        """Return the result to hand back, optionally transforming it.

        Runs once, after the (possibly retried) rollout is chosen. ``loop`` is
        offered so a policy can drive extra LLM turns via
        :meth:`AgentLoop.append_extraction_turn`.
        """
        return result


class NoOpPolicy(LoopPolicy):
    """Run a single rollout and return it unchanged."""


class CompositePolicy(LoopPolicy):
    """Combine policies: retry if any asks to, finalize in declared order."""

    def __init__(self, *policies: LoopPolicy) -> None:
        self._policies = policies

    def should_retry(
        self,
        attempt: int,
        result: AgentResult,
        ctx: AgentContext,
    ) -> bool:
        return any(p.should_retry(attempt, result, ctx) for p in self._policies)

    async def finalize(
        self,
        loop: AgentLoop,
        result: AgentResult,
        ctx: AgentContext,
    ) -> AgentResult:
        for policy in self._policies:
            result = await policy.finalize(loop, result, ctx)
        return result
