"""DeepSearch loop policy — fresh-rollout retries and a forced final answer.

DeepSearch must hand back a text answer for grading. This policy enforces that
in two stages around the standard rollout:

- **Retry** — a rollout that runs out of steps or context without submitting an
  answer is discarded and re-run from scratch, up to ``rollout_retries`` times.
- **Forced answer** — when the kept rollout still carries no answer and
  ``force_final_answer`` is set, the model is asked once more, with no tools, to
  read its own history and state the answer. A first pass uses the (optionally
  condensed) conversation; if that yields nothing, a second pass folds away the
  tool observations; failing both, a fixed fallback string is recorded.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from awe_agent.core.agent.policy import LoopPolicy
from awe_agent.core.llm.types import Message
from awe_agent.scaffold.deepsearch.prompts import (
    NO_FINAL_ANSWER_FALLBACK,
    OMITTED_TOOL_RESULT,
    get_final_answer_prompts,
    resolve_from_task_info,
)

if TYPE_CHECKING:
    from awe_agent.core.agent.context import AgentContext
    from awe_agent.core.agent.loop import AgentLoop, AgentResult

logger = logging.getLogger(__name__)

_FINISH_TOOL_NAME = "finish"
_RETRY_FINISH_REASONS = {"max_steps", "context_length"}


class RetryThenForceAnswerPolicy(LoopPolicy):
    """Guarantee a submitted text answer via fresh-rollout retries + extraction."""

    def __init__(
        self,
        rollout_retries: int = 0,
        force_final_answer: bool = True,
    ) -> None:
        self._max_retries = max(rollout_retries, 0)
        self._force_final_answer = force_final_answer

    def should_retry(
        self,
        attempt: int,
        result: AgentResult,
        ctx: AgentContext,
    ) -> bool:
        # A rollout that submitted an answer ends with finish_reason "finish";
        # the retry reasons below mean the budget ran out before an answer.
        return (
            attempt < self._max_retries
            and result.finish_reason in _RETRY_FINISH_REASONS
        )

    async def finalize(
        self,
        loop: AgentLoop,
        result: AgentResult,
        ctx: AgentContext,
    ) -> AgentResult:
        submitted = self._submitted_answer(result, ctx)
        if submitted is not None:
            result.metadata["final_answer"] = submitted
            result.metadata["answer_provenance"] = {
                "agent_submitted_final_answer": True,
                "forced_final_answer": False,
            }
            return result

        if not self._force_final_answer:
            return result

        return await self._force_answer(loop, result, ctx)

    async def _force_answer(
        self,
        loop: AgentLoop,
        result: AgentResult,
        ctx: AgentContext,
    ) -> AgentResult:
        provenance: dict[str, Any] = {
            "agent_submitted_final_answer": False,
            "forced_final_answer": True,
            "forced_final_answer_reason": result.finish_reason,
        }
        prompts = self._final_answer_prompts(ctx.task_info)
        baseline = list(result.messages)

        # First pass: the conversation the agent itself would see next.
        current_view = list(baseline)
        if ctx.condenser is not None:
            current_view = await ctx.condenser.condense(current_view)
        answer, error = await self._extract(
            loop, current_view, prompts.current_history, commit=True,
        )
        if answer:
            return self._commit(ctx, result, answer, provenance, "current_history")

        # Second pass (skipped in training): the full rollout with tool results
        # folded away. A synthetic view like this is never tokenized into the
        # training token stream, so it cannot be committed during training.
        if ctx.training is None:
            folded_view = self._fold_observations(baseline)
            answer, fold_error = await self._extract(
                loop, folded_view, prompts.folded_history, commit=False,
            )
            if answer:
                return self._commit(ctx, result, answer, provenance, "folded_full_history")
            error = fold_error or error

        if error:
            provenance["forced_final_answer_reason"] = error
        return self._commit(
            ctx, result, NO_FINAL_ANSWER_FALLBACK, provenance, "no_answer_fallback",
        )

    async def _extract(
        self,
        loop: AgentLoop,
        view: list[Message],
        instruction: str,
        *,
        commit: bool,
    ) -> tuple[str | None, str | None]:
        try:
            action = await loop.append_extraction_turn(
                view, instruction, tools=None, commit=commit,
            )
        except Exception as exc:  # noqa: BLE001 — surfaced as provenance, not raised
            logger.error("DeepSearch forced answer extraction failed: %s", exc, exc_info=True)
            return None, f"forced_extraction_error: {exc}"
        if action.tool_calls:
            return None, "forced_extraction_returned_tool_calls"
        answer = action.content.strip() if action.content else ""
        if answer:
            return answer, None
        return None, "forced_extraction_empty_response"

    def _commit(
        self,
        ctx: AgentContext,
        result: AgentResult,
        answer: str,
        provenance: dict[str, Any],
        stage: str,
    ) -> AgentResult:
        provenance["forced_final_answer_stage"] = stage
        result.metadata["final_answer"] = answer
        result.metadata["answer_provenance"] = provenance
        result.finish_reason = "finish"
        result.messages = list(ctx.messages)
        return result

    def _final_answer_prompts(self, task_info: dict[str, Any]):
        _, _, key = resolve_from_task_info(task_info)
        return get_final_answer_prompts(key)

    def _submitted_answer(self, result: AgentResult, ctx: AgentContext) -> str | None:
        """Return the answer the agent submitted via ``finish``, if any."""
        if not result.trajectory.steps:
            return None
        action = result.trajectory.steps[-1].action
        if action.type != "finish":
            return None
        for tool_call in action.tool_calls:
            name = tool_call.get("name", tool_call.get("function", {}).get("name", ""))
            if name != _FINISH_TOOL_NAME:
                continue
            arguments = tool_call.get(
                "arguments",
                tool_call.get("function", {}).get("arguments", "{}"),
            )
            try:
                params = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                return None
            tool = ctx.get_tool(_FINISH_TOOL_NAME)
            submit = getattr(tool, "submit", None)
            if callable(submit):
                answer = submit(params)
                # An empty submission is not a usable answer; fall through so
                # forced extraction can recover one.
                if answer:
                    return str(answer)
        return None

    @staticmethod
    def _fold_observations(messages: list[Message]) -> list[Message]:
        """Copy ``messages`` with tool-result bodies replaced by a placeholder."""
        folded: list[Message] = []
        for message in messages:
            content = message.content
            if message.role == "tool":
                content = OMITTED_TOOL_RESULT
            elif (
                message.role == "user"
                and isinstance(message.content, str)
                and message.content.startswith("OBSERVATION:\n")
            ):
                header = message.content.split("\n", 2)[:2]
                content = "\n".join([*header, OMITTED_TOOL_RESULT])
            folded.append(Message(
                role=message.role,
                content=content,
                tool_calls=list(message.tool_calls) if message.tool_calls else None,
                tool_call_id=message.tool_call_id,
                name=message.name,
                reasoning_raw=message.reasoning_raw,
            ))
        return folded
