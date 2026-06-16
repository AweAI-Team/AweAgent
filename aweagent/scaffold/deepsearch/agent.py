"""DeepSearchAgent — iterative web-search agent for question answering."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action
from aweagent.core.llm.format import get_tool_format
from aweagent.core.llm.format.protocol import ToolCallFormat
from aweagent.core.tool.code import FinishWithTextTool
from aweagent.core.tool.protocol import Tool
from aweagent.core.tool.registry import tool_registry
from aweagent.core.tool.search import (
    SearchConstraints,
    WebFetchRawTool,
    WebFetchTool,
    WebSearchTool,
)
from aweagent.scaffold.deepsearch.policy import RetryThenForceAnswerPolicy
from aweagent.scaffold.deepsearch.prompts import (
    NO_TOOL_CALL_PROMPT,
    get_system_prompt,
    resolve_from_task_info,
)

if TYPE_CHECKING:
    from aweagent.core.config.schema import AweAgentConfig

logger = logging.getLogger(__name__)

_FINISH_TOOL_NAME = "finish"
_DEEPSEARCH_DEFAULT_TOOLS = ["web_search", "web_fetch", "finish"]
_DEEPSEARCH_TOOLSETS = {
    "default": _DEEPSEARCH_DEFAULT_TOOLS,
}


def _format_current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_tool_names(config: AweAgentConfig) -> list[str]:
    """Resolve DeepSearch tools from an explicit list or a named toolset preset."""
    agent_config = config.agent
    if agent_config.tools is not None:
        return list(agent_config.tools)

    toolset = agent_config.toolset or "default"
    if toolset not in _DEEPSEARCH_TOOLSETS:
        available = ", ".join(sorted(_DEEPSEARCH_TOOLSETS))
        raise ValueError(f"Unknown DeepSearch toolset {toolset!r}. Available: {available}")
    return list(_DEEPSEARCH_TOOLSETS[toolset])


def _build_tools(
    tool_names: list[str],
    tool_options: dict[str, dict[str, Any]],
    search_constraints: SearchConstraints | None,
) -> list[Tool]:
    return [
        _build_tool(name, tool_options.get(name, {}), search_constraints)
        for name in tool_names
    ]


def _build_tool(
    name: str,
    options: dict[str, Any],
    search_constraints: SearchConstraints | None,
) -> Tool:
    if name == "web_search":
        return WebSearchTool(
            engine=options.get("engine"),
            constraints=search_constraints,
            max_attempts=options.get("max_attempts", 1),
            result_scheme=options.get("result_scheme"),
            backend=options.get("backend"),
        )
    if name == "web_fetch":
        reader = None
        reader_backend = options.get("reader_backend")
        if reader_backend:
            reader = WebFetchRawTool(
                constraints=search_constraints,
                max_content_tokens=options.get("max_content_tokens", 100000),
                max_attempts=options.get("reader_max_attempts", 3),
                backend=reader_backend,
            )
        return WebFetchTool(
            constraints=search_constraints,
            max_content_tokens=options.get("max_content_tokens", 25000),
            max_attempts=options.get("max_attempts", 3),
            system_prompt=options.get("system_prompt"),
            llm_config_path=options.get("llm_config_path"),
            llm_config=options.get("llm") or options.get("llm_config"),
            llm_params=options.get("llm_params"),
            reader=reader,
        )
    if name == "web_fetch_raw":
        return WebFetchRawTool(
            constraints=search_constraints,
            max_content_tokens=options.get("max_content_tokens", 100000),
            max_attempts=options.get("max_attempts", 3),
            backend=options.get("backend"),
        )
    if name == "finish":
        return FinishWithTextTool()

    tool_cls = tool_registry.get(name)
    return tool_cls(**options)


class DeepSearchAgent(Agent):
    """Search-focused agent with text-answer submission."""

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> DeepSearchAgent:
        search_constraints: SearchConstraints | None = None
        if config.security.blocked_search_patterns:
            search_constraints = SearchConstraints(
                blocked_patterns=config.security.blocked_search_patterns,
            )
        return cls(
            search_constraints=search_constraints,
            tool_call_format=config.agent.tool_call_format,
            tool_names=_resolve_tool_names(config),
            tool_options=config.agent.tool_options,
            rollout_retries=config.agent.rollout_retries,
            force_final_answer=config.agent.force_final_answer,
        )

    @classmethod
    def from_config_with_constraints(
        cls,
        config: AweAgentConfig,
        instance_constraints: SearchConstraints,
    ) -> DeepSearchAgent:
        search_constraints: SearchConstraints | None = None
        if config.security.blocked_search_patterns:
            search_constraints = SearchConstraints(
                blocked_patterns=config.security.blocked_search_patterns,
            )
        if search_constraints is not None:
            search_constraints = search_constraints.merge(instance_constraints)
        else:
            search_constraints = instance_constraints
        return cls(
            search_constraints=search_constraints,
            tool_call_format=config.agent.tool_call_format,
            tool_names=_resolve_tool_names(config),
            tool_options=config.agent.tool_options,
            rollout_retries=config.agent.rollout_retries,
            force_final_answer=config.agent.force_final_answer,
        )

    def __init__(
        self,
        search_constraints: SearchConstraints | None = None,
        max_empty_retries: int = 3,
        tool_call_format: str = "openai_function",
        tool_names: list[str] | None = None,
        tool_options: dict[str, dict[str, Any]] | None = None,
        rollout_retries: int = 0,
        force_final_answer: bool = True,
    ) -> None:
        self._format = get_tool_format(tool_call_format)
        self._max_empty_retries = max_empty_retries
        self._rollout_retries = max(rollout_retries, 0)
        self._force_final_answer = force_final_answer
        self._tools = _build_tools(
            tool_names or list(_DEEPSEARCH_DEFAULT_TOOLS),
            tool_options or {},
            search_constraints,
        )

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        system_key, _, _ = resolve_from_task_info(task_info)
        prompt = get_system_prompt(system_key).format(
            tool_names=", ".join(tool.name for tool in self._tools)
        )
        prompt = f"{prompt}\nCurrent time: {_format_current_time()}"
        tool_schemas = [tool.schema for tool in self._tools]
        suffix = self._format.get_system_prompt_suffix(tool_schemas)
        if suffix:
            prompt = prompt + "\n" + suffix
        return prompt

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def get_loop_policy(self, context: AgentContext) -> RetryThenForceAnswerPolicy:
        return RetryThenForceAnswerPolicy(
            rollout_retries=self._rollout_retries,
            force_final_answer=self._force_final_answer,
        )

    def get_tool_call_format(self) -> ToolCallFormat | None:
        return self._format

    def get_no_tool_call_prompt(self) -> str | None:
        return NO_TOOL_CALL_PROMPT

    @staticmethod
    def _is_valid_response(response: Any) -> bool:
        has_content = bool(response.content) or bool(response.tool_calls)
        truncated = getattr(response, "finish_reason", None) == "length"
        return has_content and not truncated

    async def step(self, context: AgentContext) -> Action:
        messages = context.messages
        if context.condenser is not None:
            messages = await context.condenser.condense(messages)

        api_tools = self._format.prepare_tools(context.get_tool_schemas())

        # RL training mode: pass input_ids for token-level continuation.
        llm_overrides: dict[str, Any] = {}
        if context.training is not None:
            llm_overrides["input_ids"] = context.training.get_input_ids()

        response = None
        for attempt in range(1, self._max_empty_retries + 1):
            response = await context.llm.chat(
                messages=messages, tools=api_tools, **llm_overrides,
            )
            if self._is_valid_response(response):
                break
            # In training mode, "length" means the token budget is exhausted —
            # a valid terminal state, not a transient empty response to retry.
            if context.training is not None and response.finish_status == "length":
                break
            logger.warning(
                "Invalid DeepSearch response (attempt %d/%d): empty=%s, truncated=%s",
                attempt,
                self._max_empty_retries,
                not response.content and not response.tool_calls,
                response.finish_reason == "length",
            )

        tool_calls = self._format.parse_response(response)
        if tool_calls:
            tool_call_dicts = [tc.to_dict() for tc in tool_calls]
            is_finish = any(tc.name == _FINISH_TOOL_NAME for tc in tool_calls)
            return Action(
                type="finish" if is_finish else "tool_call",
                content=response.content,
                reasoning_text=response.reasoning_text,
                reasoning_raw=response.reasoning_raw,
                tool_calls=tool_call_dicts,
                token_ids=response.completion_token_ids,
                logprobs=response.logprobs,
                weight_version=response.weight_version,
                finish_status=response.finish_status,
                usage=response.usage,
                llm_response_raw=response.raw,
            )

        return Action(
            type="message",
            content=response.content,
            reasoning_text=response.reasoning_text,
            reasoning_raw=response.reasoning_raw,
            token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
            weight_version=response.weight_version,
            finish_status=response.finish_status,
            usage=response.usage,
            llm_response_raw=response.raw,
        )
