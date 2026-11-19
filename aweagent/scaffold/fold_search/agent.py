"""FoldSearchAgent — QwenSearch-style loop with DeepSearch tool names."""
# ruff: noqa: E501

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop, AgentResult
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.stats import RunStats
from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.llm.types import Message, ToolCall
from aweagent.core.tool.protocol import Tool
from aweagent.core.tool.search import SearchConstraints
from aweagent.scaffold.fold_search.context import (
    RuntimeTokenContextRewriter,
    _as_positive_int,
    _build_tools,
    _load_tokenizer,
    _parse_arguments,
    _tool_call_dict,
    count_messages_tokens,
)
from aweagent.scaffold.fold_search.prompts import NO_TOOL_CALL_PROMPT

if TYPE_CHECKING:
    from aweagent.core.config.schema import AweAgentConfig
    from aweagent.core.llm.types import LLMResponse

logger = logging.getLogger(__name__)

BROWSECOMP_SYSTEM_PROMPT = """You are DeepSearch, an iterative web-search agent.

Your job is to solve the user's question through careful search and evidence checking. Use available search tools to discover sources, and reading or summary tools to inspect promising pages.

Available tools: {tool_names}

Guidelines:
- Iterate until you have enough evidence to answer confidently.
- Prefer primary or authoritative sources when possible.
- Cross-check important claims across sources when the answer is uncertain.
- Do not stop by writing a normal assistant message.
- When you have the answer, or cannot make further progress, call finish with the final answer in the required answer parameter.

Final answer requirements:
- You must submit the final answer by calling finish(answer=...).
- The answer value must contain only the shortest final answer string.
- Do not include reasoning, evidence, explanations, candidate lists, apologies, confidence scores, or uncertainty paragraphs inside answer.
- For a person, place, work, organization, or event, answer with only the name.
- For a numeric answer, include only the number and any necessary unit.
- For multiple required answers, separate them with commas.
"""

BROWSECOMP_SIMPLIFIED_SYSTEM_PROMPT = """You are DeepSearch, an iterative web-search agent.

Your job is to solve the user's question through careful search and evidence checking. Use available search tools to discover sources, and reading or summary tools to inspect promising pages.

Available tools: {tool_names}

Guidelines:
- Iterate until you have enough evidence to answer confidently.
- Prefer primary or authoritative sources when possible.
- Cross-check important claims across sources when the answer is uncertain.
- Do not stop by writing a normal assistant message.
- When you have the answer, or cannot make further progress, call finish with the final answer in the required answer parameter.

Final answer requirements:
- You must submit the final answer by calling finish(answer=...).
- Do not include reasoning, evidence, explanations, candidate lists, apologies, confidence scores, or uncertainty paragraphs inside answer.
- For multiple required answers, separate them with commas.
"""

FOLD_SEARCH_FINAL_HINT = (
    "Do not call search_api or link_summary_tool. "
    "Use only information already retrieved and submit the final answer now "
    "by calling finish(answer=...). The answer argument must contain only the "
    "shortest final answer string."
)

_FOLD_SEARCH_OPTIONS_KEY = "fold_search"
_MODEL_TOOL_NAMES = ("search_api", "link_summary_tool", "finish")

FOLD_SEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_api",
            "description": "Search for information from the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "link_summary_tool",
            "description": "Read a webpage and summarize content relevant to the prompt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The webpage URL.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The information to extract from the page.",
                    },
                },
                "required": ["url", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Only the shortest final answer string.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]

_FINISH_ONLY_TOOL_SCHEMAS = [FOLD_SEARCH_TOOL_SCHEMAS[-1]]


@dataclass(frozen=True)
class FoldSearchConfig:
    tokenizer_path: str
    max_context_tokens: int = 255 * 1024
    max_output_tokens_cap: int = 65500
    force_final_answer_remaining_tokens: int = 8192
    tool_context_max_tokens: int = 32000
    tool_context_target_tokens: int = 5000
    use_simplified_system_prompt: bool = False


def _as_non_negative_int(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _options(config: AweAgentConfig) -> dict[str, Any]:
    options = config.agent.tool_options.get(_FOLD_SEARCH_OPTIONS_KEY, {})
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("agent.tool_options.fold_search must be a mapping")
    return options


def _resolve_config(options: dict[str, Any]) -> FoldSearchConfig:
    tokenizer_path = options.get("tokenizer_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path.strip():
        raise ValueError("agent.tool_options.fold_search.tokenizer_path is required")
    use_simplified_system_prompt = options.get("use_simplified_system_prompt", False)
    if not isinstance(use_simplified_system_prompt, bool):
        raise ValueError("fold_search.use_simplified_system_prompt must be true or false")
    return FoldSearchConfig(
        tokenizer_path=tokenizer_path.strip(),
        max_context_tokens=_as_positive_int(
            options.get("max_context_tokens"),
            default=255 * 1024,
            field_name="fold_search.max_context_tokens",
        ),
        max_output_tokens_cap=_as_positive_int(
            options.get("max_output_tokens_cap"),
            default=65500,
            field_name="fold_search.max_output_tokens_cap",
        ),
        force_final_answer_remaining_tokens=_as_non_negative_int(
            options.get("force_final_answer_remaining_tokens"),
            default=8192,
            field_name="fold_search.force_final_answer_remaining_tokens",
        ),
        tool_context_max_tokens=_as_positive_int(
            options.get("tool_context_max_tokens"),
            default=32000,
            field_name="fold_search.tool_context_max_tokens",
        ),
        tool_context_target_tokens=_as_positive_int(
            options.get("tool_context_target_tokens"),
            default=5000,
            field_name="fold_search.tool_context_target_tokens",
        ),
        use_simplified_system_prompt=use_simplified_system_prompt,
    )


def _finish_answer_from_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    for tool_call in tool_calls:
        func = tool_call.get("function", {})
        name = func.get("name", "")
        if name != "finish":
            continue
        args = _parse_arguments(name, func.get("arguments", "{}"))
        answer = args.get("answer")
        if answer is not None:
            return str(answer).strip()
    return None


def _extract_text_answer(content: str | None) -> str:
    text = (content or "").strip()
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


class FoldSearchAgent(Agent):
    """Search scaffold exposing search_api/link_summary_tool/finish to the model."""

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> FoldSearchAgent:
        search_constraints = None
        if config.security.blocked_search_patterns:
            search_constraints = SearchConstraints(
                blocked_patterns=config.security.blocked_search_patterns,
            )
        return cls(
            config=_resolve_config(_options(config)),
            tool_options=config.agent.tool_options,
            search_constraints=search_constraints,
            rollout_retries=config.agent.rollout_retries,
            force_final_answer=config.agent.force_final_answer,
        )

    @classmethod
    def from_config_with_constraints(
        cls,
        config: AweAgentConfig,
        instance_constraints: SearchConstraints,
    ) -> FoldSearchAgent:
        search_constraints = None
        if config.security.blocked_search_patterns:
            search_constraints = SearchConstraints(
                blocked_patterns=config.security.blocked_search_patterns,
            )
        search_constraints = (
            search_constraints.merge(instance_constraints)
            if search_constraints is not None
            else instance_constraints
        )
        return cls(
            config=_resolve_config(_options(config)),
            tool_options=config.agent.tool_options,
            search_constraints=search_constraints,
            rollout_retries=config.agent.rollout_retries,
            force_final_answer=config.agent.force_final_answer,
        )

    def __init__(
        self,
        *,
        config: FoldSearchConfig,
        tool_options: dict[str, dict[str, Any]] | None = None,
        search_constraints: SearchConstraints | None = None,
        tokenizer: Any | None = None,
        rollout_retries: int = 0,
        force_final_answer: bool = True,
    ) -> None:
        self.config = config
        self._tokenizer = tokenizer or _load_tokenizer(config.tokenizer_path)
        self._tools = _build_tools(tool_options or {}, search_constraints)
        # Same semantics as the global config: 0 = one rollout, 1 = one retry.
        self._rollout_retries = max(int(rollout_retries or 0), 0)
        self._force_final_answer = force_final_answer

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        # Keep the established prompt as default so existing evaluations are unchanged.
        prompt = (
            BROWSECOMP_SIMPLIFIED_SYSTEM_PROMPT
            if self.config.use_simplified_system_prompt
            else BROWSECOMP_SYSTEM_PROMPT
        )
        return prompt.format(tool_names=", ".join(_MODEL_TOOL_NAMES))

    def get_model_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the model-facing schemas so task-specific variants can refine them."""
        return FOLD_SEARCH_TOOL_SCHEMAS

    def get_finish_only_tool_schemas(self) -> list[dict[str, Any]]:
        return _FINISH_ONLY_TOOL_SCHEMAS

    def get_final_hint(self) -> str:
        return FOLD_SEARCH_FINAL_HINT

    def get_no_tool_call_prompt(self) -> str:
        return NO_TOOL_CALL_PROMPT

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def create_loop(self, context: AgentContext) -> AgentLoop:
        return FoldSearchLoop(self, context)

    async def step(self, context: AgentContext) -> Action:
        raise NotImplementedError("FoldSearchAgent uses FoldSearchLoop directly")


class FoldSearchLoop(AgentLoop):
    """QwenSearch-style loop with finish(answer=...) termination."""

    agent: FoldSearchAgent

    def _count_agent_prompt_tokens(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ) -> int:
        config = getattr(self.ctx.llm, "config", None)
        params = getattr(config, "params", {})
        template_kwargs = params.get("extra_body", {}).get("chat_template_kwargs", {})
        return count_messages_tokens(
            self.agent._tokenizer, messages, tools=tools,
            chat_template_kwargs=template_kwargs,
        )

    def _response_to_action(self, response: LLMResponse) -> Action:
        """Convert the provider response into the action consumed by this loop."""
        tool_calls = [_tool_call_dict(tool_call) for tool_call in response.tool_calls]
        return Action(
            type="tool_call" if tool_calls else "message",
            content=response.content or "",
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

    async def _generate_response(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int,
        stats: RunStats,
    ) -> LLMResponse:
        """Generate an action and record model usage."""
        started = time.monotonic()
        response = await self.ctx.llm.chat(
            messages=messages, tools=tools, max_tokens=max_tokens,
        )
        usage = response.usage
        stats.record_llm_call(
            time.monotonic() - started,
            (getattr(usage, "prompt_tokens", 0) or 0) if usage else self._count_agent_prompt_tokens(messages, tools),
            (getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
        )
        return response

    async def run(self, task_prompt: str) -> AgentResult:
        max_attempts = self.agent._rollout_retries + 1
        result = await self._run_once(task_prompt)
        used_attempts = 1

        while used_attempts < max_attempts and result.finish_reason == "max_steps":
            logger.info(
                "FoldSearch rollout attempt %d/%d reached max_steps; retrying",
                used_attempts,
                max_attempts,
            )
            used_attempts += 1
            result = await self._run_once(task_prompt)

        if used_attempts > 1:
            result.metadata["rollout_attempts"] = used_attempts

        if (
            self.agent._rollout_retries > 0
            and result.finish_reason == "max_steps"
            and self.agent._force_final_answer
        ):
            result = await self._force_final_answer(result)
        return result

    def _context_guard_details(self, prompt_tokens: int) -> dict[str, Any] | None:
        threshold = self.agent.config.force_final_answer_remaining_tokens
        remaining_tokens = self.agent.config.max_context_tokens - prompt_tokens
        if (
            not self.agent._force_final_answer
            or threshold == 0
            or remaining_tokens >= threshold
        ):
            return None
        return {
            "context_guard_triggered": True,
            "context_guard_step": self.ctx.current_step,
            "context_guard_prompt_tokens": prompt_tokens,
            "context_guard_remaining_tokens": remaining_tokens,
            "context_guard_threshold_tokens": threshold,
        }

    async def _run_once(self, task_prompt: str) -> AgentResult:
        self.ctx.messages = [
            Message(role="system", content=self.agent.get_system_prompt(self.ctx.task_info)),
            Message(role="user", content=task_prompt),
        ]
        self.ctx.trajectory = Trajectory()

        stats = RunStats()
        stats.start()
        finish_reason = "max_steps"
        final_answer = ""
        answer_source = "fold_search_text"
        fold_triggered_step: int | None = None
        rewriter = RuntimeTokenContextRewriter(
            self.agent._tokenizer,
            max_context_length=self.agent.config.tool_context_max_tokens,
            target_context_length=self.agent.config.tool_context_target_tokens,
        )

        for step in range(self.ctx.max_steps):
            self.ctx.current_step = step
            if step + 1 == self.ctx.max_steps:
                self._append_final_hint_once()

            inference_messages = rewriter.process(self.ctx.messages)
            if fold_triggered_step is None and any(
                message.role == "tool" and str(message.content or "").startswith("Content folded")
                for message in inference_messages
            ):
                fold_triggered_step = step

            prompt_tokens = self._count_agent_prompt_tokens(
                inference_messages,
                self.agent.get_model_tool_schemas(),
            )
            context_guard = self._context_guard_details(prompt_tokens)
            if context_guard is not None:
                logger.info(
                    "FoldSearch remaining context %d is below force-answer threshold %d "
                    "at step %d; submitting final answer",
                    context_guard["context_guard_remaining_tokens"],
                    context_guard["context_guard_threshold_tokens"],
                    step,
                )
                stats.finish()
                result = AgentResult(
                    trajectory=self.ctx.trajectory,
                    messages=list(self.ctx.messages),
                    finish_reason="context_length",
                    metadata={
                        "stats": stats.to_dict(),
                        "final_answer": final_answer,
                        "answer_provenance": {
                            "source": answer_source,
                            "finish_tool_used": False,
                            "forced_final_answer": False,
                            "context_fold_trigger_step": fold_triggered_step,
                        },
                    },
                )
                return await self._force_final_answer(
                    result,
                    reason="remaining_context_below_threshold",
                    extra_provenance=context_guard,
                )

            dynamic_max_tokens = max(
                1,
                min(
                    self.agent.config.max_context_tokens - prompt_tokens,
                    self.agent.config.max_output_tokens_cap,
                ),
            )

            try:
                response = await self._generate_response(
                    inference_messages,
                    self.agent.get_model_tool_schemas(),
                    dynamic_max_tokens,
                    stats,
                )
            except Exception as exc:
                logger.error("FoldSearch step %d failed: %s", step, exc, exc_info=True)
                return AgentResult(
                    trajectory=self.ctx.trajectory,
                    messages=list(self.ctx.messages),
                    finish_reason="error",
                    error=str(exc),
                    metadata={"stats": stats.to_dict()},
                )

            action = self._response_to_action(response)
            tool_calls = action.tool_calls
            self.ctx.trajectory.add_step(
                step=step,
                action=action,
                reasoning_text=action.reasoning_text,
                llm_response_raw=action.llm_response_raw,
            )

            finish_answer = _finish_answer_from_tool_calls(tool_calls)
            if finish_answer is not None:
                self.ctx.messages.append(Message(
                    role="assistant",
                    content=action.content or "",
                    tool_calls=[
                        ToolCall.from_dict(tool_call) for tool_call in action.tool_calls
                    ],
                    reasoning_raw=action.reasoning_raw,
                ))
                final_answer = finish_answer
                answer_source = "fold_search_finish_tool"
                finish_reason = "finish"
                stats.end_step()
                break

            if tool_calls:
                observations = await self._execute_search_tools(action)
                self.ctx.trajectory.steps[-1].observations = observations
                for tool_call in tool_calls:
                    stats.record_tool_call(
                        tool_call.get("function", {}).get("name", ""),
                        0.0,
                    )
            else:
                self.ctx.messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    reasoning_raw=response.reasoning_raw,
                ))
                no_tool_call_prompt = self.agent.get_no_tool_call_prompt()
                if no_tool_call_prompt:
                    # Keep the draft and remind the model to continue with a tool.
                    self.ctx.messages.append(Message(
                        role="user",
                        content=no_tool_call_prompt,
                    ))
                    self.ctx.trajectory.steps[-1].metadata.setdefault(
                        "invalid_message_retry",
                        {},
                    ).update({
                        "reason": "no_tool_call",
                        "prompt": no_tool_call_prompt,
                    })
                    stats.end_step()
                    continue
                final_answer = _extract_text_answer(response.content)
                finish_reason = "finish"
                stats.end_step()
                break

            stats.end_step()

        stats.finish()
        if finish_reason == "max_steps":
            final_answer = _extract_text_answer(self._terminal_assistant_content())

        return AgentResult(
            trajectory=self.ctx.trajectory,
            messages=list(self.ctx.messages),
            finish_reason=finish_reason,
            metadata={
                "stats": stats.to_dict(),
                "final_answer": final_answer,
                "answer_provenance": {
                    "source": answer_source,
                    "finish_tool_used": answer_source == "fold_search_finish_tool",
                    "forced_final_answer": False,
                    "context_fold_trigger_step": fold_triggered_step,
                },
            },
        )

    def _append_final_hint_once(self, *, at_tail: bool = False) -> None:
        if at_tail:
            if (
                self.ctx.messages
                and self.ctx.messages[-1].role == "tool"
                and self.ctx.messages[-1].name == "hint"
            ):
                return
            self.ctx.messages.append(Message(
                role="tool",
                name="hint",
                content=self.agent.get_final_hint(),
            ))
            return
        if any(message.role == "tool" and message.name == "hint" for message in self.ctx.messages):
            return
        self.ctx.messages.append(
            Message(role="tool", name="hint", content=self.agent.get_final_hint())
        )

    def _terminal_assistant_content(self) -> str:
        if self.ctx.messages and self.ctx.messages[-1].role == "assistant":
            return str(self.ctx.messages[-1].content or "")
        return ""

    async def _force_final_answer(
        self,
        result: AgentResult,
        *,
        reason: str = "max_steps_after_retries",
        extra_provenance: dict[str, Any] | None = None,
    ) -> AgentResult:
        self._append_final_hint_once(at_tail=True)
        provenance = result.metadata.setdefault("answer_provenance", {})
        provenance.update(extra_provenance or {})
        provenance.update({
            "forced_final_answer": True,
            "forced_final_answer_reason": reason,
        })
        rewriter = RuntimeTokenContextRewriter(
            self.agent._tokenizer,
            max_context_length=self.agent.config.tool_context_max_tokens,
            target_context_length=self.agent.config.tool_context_target_tokens,
        )
        inference_messages = rewriter.process(self.ctx.messages)
        prompt_tokens = self._count_agent_prompt_tokens(
            inference_messages,
            self.agent.get_finish_only_tool_schemas(),
        )
        dynamic_max_tokens = max(
            1,
            min(
                self.agent.config.max_context_tokens - prompt_tokens,
                self.agent.config.max_output_tokens_cap,
            ),
        )
        try:
            response = await self.ctx.llm.chat(
                messages=inference_messages,
                tools=self.agent.get_finish_only_tool_schemas(),
                max_tokens=dynamic_max_tokens,
            )
        except Exception as exc:
            logger.warning("FoldSearch forced final answer failed: %s", exc, exc_info=True)
            result.metadata.setdefault("answer_provenance", {})[
                "forced_final_answer_error"
            ] = str(exc)
            return result

        action = self._response_to_action(response)
        tool_calls = action.tool_calls
        self.ctx.trajectory.add_step(
            step=len(self.ctx.trajectory.steps),
            action=action,
            reasoning_text=action.reasoning_text,
            llm_response_raw=action.llm_response_raw,
        )
        self.ctx.messages.append(Message(
            role="assistant",
            content=response.content or "",
            tool_calls=[ToolCall.from_dict(tool_call) for tool_call in tool_calls]
            if tool_calls else None,
            reasoning_raw=response.reasoning_raw,
        ))

        answer = _finish_answer_from_tool_calls(tool_calls)
        if answer is None:
            answer = _extract_text_answer(response.content)
        result.messages = list(self.ctx.messages)
        result.metadata["final_answer"] = answer
        provenance["finish_tool_used"] = (
            _finish_answer_from_tool_calls(tool_calls) is not None
        )
        if answer:
            result.finish_reason = "finish"
        return result

    async def _execute_search_tools(self, action: Action) -> list[str]:
        self.ctx.messages.append(Message(
            role="assistant",
            content=action.content or "",
            tool_calls=[ToolCall.from_dict(tool_call) for tool_call in action.tool_calls],
            reasoning_raw=action.reasoning_raw,
        ))

        observations: list[str] = []
        for tool_call in action.tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name", "")
            call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"
            args = _parse_arguments(name, func.get("arguments", "{}"))
            tool_start = time.monotonic()
            obs = await self._execute_one_search_tool(name, args)
            logger.info("FoldSearch tool %s took %.2fs", name, time.monotonic() - tool_start)
            observations.append(obs)
            self.ctx.messages.append(Message(
                role="tool",
                name=name,
                content=obs,
                tool_call_id=call_id,
            ))
        return observations

    async def _execute_one_search_tool(self, name: str, args: dict[str, Any]) -> str:
        if name in {"search_api", "web_search"}:
            raw_query = args.get("query", args.get("queries", []))
            queries = raw_query if isinstance(raw_query, list) else [raw_query]
            queries = [query for query in queries if isinstance(query, str) and query.strip()]
            if not queries:
                return "Error: empty search query."
            tool = self.ctx.get_tool("web_search")
            if tool is None:
                return "Error: tool 'web_search' not found."
            return await tool.execute(
                {"query": queries if len(queries) > 1 else queries[0]},
                session=self.ctx.session,
            )

        if name in {"link_summary_tool", "web_extractor"}:
            raw_url = args.get("url", args.get("urls", []))
            urls = raw_url if isinstance(raw_url, list) else [raw_url]
            urls = [url for url in urls if isinstance(url, str) and url.strip()]
            if not urls:
                return "Error: empty URL."
            tool = self.ctx.get_tool("web_fetch")
            if tool is None:
                return "Error: tool 'web_fetch' not found."
            prompt = str(args.get("prompt", args.get("goal", "")))
            parts = []
            for url in urls:
                fetched = await tool.execute(
                    {"url": url, "prompt": prompt},
                    session=self.ctx.session,
                )
                parts.append(f"Extracted content for: {url}\n\n{fetched}")
            return "\n\n".join(parts)

        if name == "finish":
            return "Final answer submitted."
        return f"Error: Unknown tool {name}"


__all__ = [
    "BROWSECOMP_SIMPLIFIED_SYSTEM_PROMPT",
    "BROWSECOMP_SYSTEM_PROMPT",
    "FOLD_SEARCH_FINAL_HINT",
    "FOLD_SEARCH_TOOL_SCHEMAS",
    "FoldSearchAgent",
    "FoldSearchConfig",
    "FoldSearchLoop",
]
