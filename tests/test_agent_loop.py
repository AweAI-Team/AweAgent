"""Tests for the agent loop execution engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentLoop, AgentResult
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from awe_agent.core.tool.builtin.bash import BashTool
from awe_agent.core.tool.builtin.think import ThinkTool
from awe_agent.scaffold.search_swe.agent import SearchSWEAgent
from tests.conftest import MockRuntimeSession


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    llm = AsyncMock(spec=LLMClient)
    return llm


@pytest.fixture
def agent_context(mock_llm, mock_session):
    """Create an agent context with mock dependencies."""
    agent = SearchSWEAgent()
    return AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"workdir": "/testbed", "instance_id": "test-1"},
        max_steps=10,
    )


@pytest.mark.asyncio
async def test_agent_loop_finish_immediately(mock_llm, mock_session):
    """Agent finishes on first step with no tool calls."""
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content="The issue is already fixed.",
        tool_calls=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
    ))

    agent = SearchSWEAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"workdir": "/testbed"},
        max_steps=10,
    )
    loop = AgentLoop(agent, ctx)
    result = await loop.run("Fix the bug")

    assert result.finish_reason == "finish"
    assert len(result.trajectory.steps) == 1
    assert result.trajectory.steps[0].action.type == "finish"


@pytest.mark.asyncio
async def test_agent_loop_tool_then_finish(mock_llm, mock_session):
    """Agent uses a tool, then finishes."""
    call_count = 0

    async def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content="Let me check the code.",
                tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command": "ls /testbed"}')],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            )
        else:
            return LLMResponse(
                content="The fix is complete.",
                tool_calls=None,
                usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
            )

    mock_llm.chat = mock_chat

    agent = SearchSWEAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"workdir": "/testbed"},
        max_steps=10,
    )
    loop = AgentLoop(agent, ctx)
    result = await loop.run("Fix the bug")

    assert result.finish_reason == "finish"
    assert len(result.trajectory.steps) == 2
    assert result.trajectory.steps[0].action.type == "tool_call"
    assert result.trajectory.steps[1].action.type == "finish"
    assert "ls /testbed" in mock_session.commands


@pytest.mark.asyncio
async def test_agent_loop_max_steps(mock_llm, mock_session):
    """Agent hits max_steps limit."""
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(
            content="Trying again...",
            tool_calls=[ToolCall(id="tc1", name="think", arguments='{"thought": "hmm"}')],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        )

    mock_llm.chat = mock_chat

    agent = SearchSWEAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"workdir": "/testbed"},
        max_steps=3,
    )
    loop = AgentLoop(agent, ctx)
    result = await loop.run("Fix the bug")

    assert result.finish_reason == "max_steps"
    assert len(result.trajectory.steps) == 3


@pytest.mark.asyncio
async def test_agent_loop_error_handling(mock_llm, mock_session):
    """Agent handles errors gracefully."""
    async def mock_chat(messages, tools=None, **kwargs):
        raise RuntimeError("API connection failed")

    mock_llm.chat = mock_chat

    agent = SearchSWEAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"workdir": "/testbed"},
        max_steps=10,
    )
    loop = AgentLoop(agent, ctx)
    result = await loop.run("Fix the bug")

    assert result.finish_reason == "error"
    assert "API connection failed" in result.error


@pytest.mark.asyncio
async def test_agent_loop_step_callbacks(mock_llm, mock_session):
    """Step callbacks are invoked after each step."""
    call_count = 0
    callback_steps = []

    async def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return LLMResponse(
                content=f"Step {call_count}",
                tool_calls=[ToolCall(id=f"tc{call_count}", name="think", arguments='{"thought": "ok"}')],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
            )
        return LLMResponse(
            content="Done",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
        )

    mock_llm.chat = mock_chat

    async def callback(step, action, observations):
        callback_steps.append(step)

    agent = SearchSWEAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"workdir": "/testbed"},
        max_steps=10,
        step_callbacks=[callback],
    )
    loop = AgentLoop(agent, ctx)
    await loop.run("Fix the bug")

    assert callback_steps == [0, 1]


@pytest.mark.asyncio
async def test_single_step_for_rl(mock_llm, mock_session):
    """run_single_step works for RL integration."""
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content="Thinking...",
        tool_calls=[ToolCall(id="tc1", name="think", arguments='{"thought": "analyzing"}')],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
    ))

    agent = SearchSWEAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={},
        max_steps=1,
    )
    loop = AgentLoop(agent, ctx)

    messages = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Fix the bug."),
    ]
    action, observations = await loop.run_single_step(messages)

    assert action.type == "tool_call"
    assert len(observations) == 1
