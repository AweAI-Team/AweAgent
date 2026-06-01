"""DeepSearch-specific execution loop.

This module keeps search-QA rollout policy out of the shared AgentLoop.  The
standard AgentLoop still executes one trajectory; DeepSearchLoop decides when
to retry fresh trajectories and how to extract a submitted text answer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentLoop, AgentResult
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.llm.types import Message
from awe_agent.scaffold.deepsearch.prompts import (
    NO_FINAL_ANSWER_FALLBACK,
    OMITTED_TOOL_RESULT,
    get_final_answer_prompts,
    resolve_from_task_info,
)

logger = logging.getLogger(__name__)


class DeepSearchLoop:
    """Run DeepSearch with optional fresh rollout retries."""

    def __init__(
        self,
        agent: Any,
        context: AgentContext,
        rollout_retries: int = 0,
        force_final_answer: bool = True,
    ) -> None:
        self.agent = agent
        self.ctx = context
        self.rollout_retries = max(rollout_retries, 0)
        self.force_final_answer = force_final_answer

    async def run(self, task_prompt: str) -> AgentResult:
        total_attempts = 1 + self.rollout_retries
        last_result: AgentResult | None = None

        for attempt in range(total_attempts):
            # Each retry is a fresh standard rollout; AgentLoop stays single-run only.
            result = await AgentLoop(self.agent, self.ctx).run(task_prompt)
            self._annotate_result(result, attempt + 1, total_attempts)
            last_result = result

            should_retry = (
                result.finish_reason in {"max_steps", "context_length"}
                and "final_answer" not in result.metadata
                and attempt < total_attempts - 1
            )
            if not should_retry:
                break

            logger.info(
                "DeepSearch rollout attempt %d/%d ended with %s; retrying from scratch",
                attempt + 1,
                total_attempts,
                result.finish_reason,
            )

        assert last_result is not None
        if (
            self.force_final_answer
            and "final_answer" not in last_result.metadata
        ):
            # If the model never submitted finish(answer=...), extract an answer without tools.
            return await self._force_final_answer(last_result)

        return last_result

    def _annotate_result(
        self,
        result: AgentResult,
        attempt: int,
        total_attempts: int,
    ) -> None:
        final_answer = (
            self._extract_final_answer(result.trajectory.steps[-1].action)
            if result.trajectory.steps
            else None
        )
        if final_answer is not None:
            result.metadata["final_answer"] = final_answer
            result.metadata.setdefault("agent_submitted_final_answer", True)
            result.metadata.setdefault("forced_final_answer", False)
        result.metadata["rollout_attempt"] = attempt
        result.metadata["rollout_attempts"] = total_attempts

    async def _force_final_answer(self, result: AgentResult) -> AgentResult:
        # Preserve the original rollout and append extraction steps for auditability.
        original_finish_reason = result.finish_reason
        metadata = result.metadata
        metadata["agent_submitted_final_answer"] = False
        metadata["forced_final_answer"] = True
        metadata["forced_final_answer_reason"] = original_finish_reason

        answer, error = await self._extract_answer_from_current_history(result)
        if answer:
            metadata["final_answer"] = answer
            metadata["forced_final_answer_stage"] = "current_history"
        else:
            if error:
                metadata["forced_final_answer_reason"] = error
            answer, error = await self._extract_answer_from_folded_history(result)
            if answer:
                metadata["final_answer"] = answer
                metadata["forced_final_answer_stage"] = "folded_full_history"
            else:
                metadata["final_answer"] = NO_FINAL_ANSWER_FALLBACK
                metadata["forced_final_answer_stage"] = "no_answer_fallback"
                if error:
                    metadata["forced_final_answer_reason"] = error

        result.finish_reason = "finish"
        self.ctx.messages = list(result.messages)
        self.ctx.trajectory = result.trajectory
        return result

    async def _extract_answer_from_current_history(
        self,
        result: AgentResult,
    ) -> tuple[str | None, str | None]:
        # First try the same context view the agent would normally see.
        messages = list(result.messages)
        if self.ctx.condenser is not None:
            messages = await self.ctx.condenser.condense(messages)
        prompts = self._get_final_answer_prompts()
        messages.append(Message(role="user", content=prompts.current_history))
        return await self._call_answer_extractor(
            result,
            messages,
            prompts.current_history,
        )

    async def _extract_answer_from_folded_history(
        self,
        result: AgentResult,
    ) -> tuple[str | None, str | None]:
        # Fallback uses the full final rollout with observations folded away.
        messages = self._build_folded_history_messages(result.messages)
        prompts = self._get_final_answer_prompts()
        messages.append(Message(role="user", content=prompts.folded_history))
        return await self._call_answer_extractor(
            result,
            messages,
            prompts.folded_history,
        )

    async def _call_answer_extractor(
        self,
        result: AgentResult,
        messages: list[Message],
        prompt: str,
    ) -> tuple[str | None, str | None]:
        step = len(result.trajectory.steps)
        result.messages.append(Message(role="user", content=prompt))

        try:
            # No tools are exposed here; this stage must produce text, not more search.
            response = await self.ctx.llm.chat(messages=messages, tools=None)
        except Exception as exc:
            logger.error("DeepSearch final answer extraction failed: %s", exc, exc_info=True)
            return None, f"forced_extraction_error: {exc}"

        tool_calls = [tc.to_dict() for tc in response.tool_calls]
        action = Action(
            type="message",
            content=response.content,
            reasoning_text=response.reasoning_text,
            reasoning_raw=response.reasoning_raw,
            tool_calls=tool_calls,
            token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
            weight_version=response.weight_version,
            finish_status=response.finish_status,
            usage=response.usage,
            llm_response_raw=response.raw,
        )
        result.trajectory.add_step(
            step=step,
            action=action,
            reasoning_text=action.reasoning_text,
            llm_response_raw=action.llm_response_raw,
        )

        if response.content:
            result.messages.append(Message(
                role="assistant",
                content=response.content,
                reasoning_raw=response.reasoning_raw,
            ))

        answer = response.content.strip() if response.content else None
        if answer:
            return answer, None
        if response.tool_calls:
            return None, "forced_extraction_returned_tool_calls"
        return None, "forced_extraction_empty_response"

    def _build_folded_history_messages(
        self,
        messages: list[Message],
    ) -> list[Message]:
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

    def _get_final_answer_prompts(self):
        _, _, final_answer_key = resolve_from_task_info(self.ctx.task_info)
        return get_final_answer_prompts(final_answer_key)

    def _extract_final_answer(self, action: Action) -> str | None:
        if action.type != "finish":
            return None
        for tool_call in action.tool_calls:
            name = tool_call.get("name", tool_call.get("function", {}).get("name", ""))
            if name != "finish":
                continue
            arguments = tool_call.get(
                "arguments",
                tool_call.get("function", {}).get("arguments", "{}"),
            )
            try:
                params = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                return None
            tool = self.ctx.get_tool("finish")
            submit = getattr(tool, "submit", None)
            if callable(submit):
                answer = submit(params)
                if answer is not None:
                    return str(answer)
        if action.tool_calls:
            return None
        return action.content.strip() if action.content else None
