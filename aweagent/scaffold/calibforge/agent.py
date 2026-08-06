"""Independent CalibForge-Eval code-agent scaffold based on the DeepSeek-V4 paper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action
from aweagent.core.llm.format import get_tool_format
from aweagent.core.llm.types import Message
from aweagent.core.tool.code import ExecuteBashTool, FinishTool, StrReplaceEditorTool
from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.calibforge.prompts import (
    NO_TOOL_CALL_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

if TYPE_CHECKING:
    from aweagent.core.config.schema import AweAgentConfig
    from aweagent.core.llm.format.protocol import ToolCallFormat

logger = logging.getLogger(__name__)

_FINISH_TOOL_NAME = "finish"


class CalibForgeAgent(Agent):
    """Tool-calling terminal agent used by the CalibForge-Eval scaffold."""

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> CalibForgeAgent:
        if config.agent.tool_call_format != "openai_function":
            raise ValueError(
                "calibforge requires agent.tool_call_format='openai_function'"
            )
        return cls(
            bash_timeout=config.agent.bash_timeout,
            bash_max_timeout=config.agent.bash_max_timeout,
            max_output_length=config.agent.max_output_length,
            bash_blocklist=config.security.bash_blocklist or None,
        )

    @classmethod
    def from_config_with_constraints(
        cls,
        config: AweAgentConfig,
        instance_constraints: Any,
    ) -> CalibForgeAgent:
        return cls.from_config(config)

    def __init__(
        self,
        bash_timeout: int = 180,
        bash_max_timeout: int = 600,
        max_output_length: int = 32000,
        bash_blocklist: list[str] | None = None,
        max_empty_retries: int = 3,
    ) -> None:
        self._format: ToolCallFormat = get_tool_format("openai_function")
        self._max_empty_retries = max_empty_retries
        self._initialized = False

        self._tools: list[Tool] = [
            ExecuteBashTool(
                timeout=bash_timeout,
                max_output_length=max_output_length,
                blocklist=bash_blocklist,
                max_timeout=bash_max_timeout,
            ),
            StrReplaceEditorTool(),
            FinishTool(),
        ]

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return SYSTEM_PROMPT

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def get_tool_call_format(self) -> ToolCallFormat | None:
        return self._format

    def get_no_tool_call_prompt(self) -> str | None:
        return NO_TOOL_CALL_PROMPT

    async def step(self, context: AgentContext) -> Action:
        if not self._initialized:
            self._initialize_prompt(context)

        messages = context.messages
        if context.condenser is not None:
            messages = await context.condenser.condense(messages)

        api_tools = self._format.prepare_tools(context.get_tool_schemas())
        llm_overrides: dict[str, Any] = {}
        if context.training is not None:
            llm_overrides["input_ids"] = context.training.get_input_ids()

        response = None
        for attempt in range(1, self._max_empty_retries + 1):
            response = await context.llm.chat(
                messages=messages,
                tools=api_tools,
                **llm_overrides,
            )
            if response.content or response.tool_calls:
                break
            if context.training is not None and response.finish_status == "length":
                break
            logger.warning(
                "Empty CalibForge response (attempt %d/%d)",
                attempt,
                self._max_empty_retries,
            )

        assert response is not None
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

    def _initialize_prompt(self, context: AgentContext) -> None:
        prompt = USER_PROMPT_TEMPLATE.format(
            instruction=context.task_info.get("instruction", "").strip(),
            workdir=context.task_info.get("workdir", "/workspace"),
        )

        for i, msg in enumerate(context.messages):
            if msg.role == "user":
                context.messages[i] = Message(role="user", content=prompt)
                break

        if context.training is not None:
            msg_dicts = [m.to_dict() for m in context.messages]
            tool_schemas = context.get_tool_schemas() or None
            context.training.init_prompt(msg_dicts, tools=tool_schemas)

        self._initialized = True
