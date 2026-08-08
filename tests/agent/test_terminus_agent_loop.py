"""Agent-loop integration points required by Terminus 2 JSON."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action
from aweagent.core.condenser.terminus_2 import Terminus2Condenser
from aweagent.core.llm.client import LLMClient
from aweagent.core.llm.format.terminus_json import TerminusJSONFormat
from aweagent.core.llm.types import Message, TokenUsage
from aweagent.core.tool.code import ThinkTool
from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.terminus_2.agent import Terminus2Agent


class HighUsageThenFinishAgent(Agent):
    def __init__(self) -> None:
        self.calls = 0

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return ""

    def get_tools(self) -> list[Tool]:
        return [ThinkTool()]

    async def step(self, context: AgentContext) -> Action:
        self.calls += 1
        if self.calls == 1:
            return Action(
                type="tool_call",
                content="thinking",
                tool_calls=[
                    {"id": "tc1", "name": "think", "arguments": '{"content":"x"}'}
                ],
                usage=TokenUsage(prompt_tokens=90, completion_tokens=20),
            )
        return Action(type="finish", content="done")


class ContextManagingHighUsageThenFinishAgent(HighUsageThenFinishAgent):
    def manages_context_limits(self, condenser: object) -> bool:
        return isinstance(condenser, Terminus2Condenser)


class PlainAgent(Agent):
    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return ""

    def get_tools(self) -> list[Tool]:
        return [ThinkTool()]

    async def step(self, context: AgentContext) -> Action:
        return Action(type="finish")


@pytest.fixture
def mock_llm() -> AsyncMock:
    return AsyncMock(spec=LLMClient)


@pytest.mark.asyncio
async def test_context_managing_condenser_bypasses_legacy_hard_stop(
    mock_llm: AsyncMock,
) -> None:
    agent = ContextManagingHighUsageThenFinishAgent()
    context = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=3,
        max_context_length=100,
        condenser=Terminus2Condenser(
            enable_summarize=False,
            token_counter=lambda model, messages: 1,
        ),
    )

    result = await AgentLoop(agent, context).run("task")

    assert result.finish_reason == "finish"
    assert agent.calls == 2


@pytest.mark.asyncio
async def test_unmarked_agent_cannot_bypass_context_guard(
    mock_llm: AsyncMock,
) -> None:
    agent = HighUsageThenFinishAgent()
    context = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=3,
        max_context_length=100,
        condenser=Terminus2Condenser(
            enable_summarize=False,
            token_counter=lambda model, messages: 1,
        ),
    )

    result = await AgentLoop(agent, context).run("task")

    assert result.finish_reason == "context_length"
    assert agent.calls == 1


def test_terminus2_agent_only_manages_terminus2_condenser() -> None:
    agent = object.__new__(Terminus2Agent)
    condenser = Terminus2Condenser(
        enable_summarize=False,
        token_counter=lambda model, messages: 1,
    )

    assert agent.manages_context_limits(condenser) is True
    assert agent.manages_context_limits(None) is False


@pytest.mark.asyncio
async def test_execute_tools_uses_text_format_observation_formatter(
    mock_llm: AsyncMock,
) -> None:
    agent = PlainAgent()
    context = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        tool_call_format=TerminusJSONFormat(),
    )
    action = Action(
        type="tool_call",
        content="assistant text",
        tool_calls=[
            {"id": "tc1", "name": "think", "arguments": '{"content":"ok"}'}
        ],
    )

    observations = await AgentLoop(agent, context)._execute_tools(action)

    assert len(observations) == 1
    assert context.messages[-1] == Message(role="user", content=observations[0])
