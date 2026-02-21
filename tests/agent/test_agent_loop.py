"""Tests for the agent loop execution engine."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentLoop, AgentResult
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from awe_agent.core.tool.code import ExecuteBashTool, ThinkTool
from awe_agent.core.tool.protocol import Tool
from awe_agent.scaffold.search_swe.agent import SearchSWEAgent
from tests.conftest import MockRuntimeSession


# ── Helpers ──────────────────────────────────────────────────────────────


class _PlainAgent(Agent):
    """Minimal agent that does NOT provide a no-tool-call prompt.

    Used to test the backward-compatible "no tool call → immediate finish"
    behaviour of AgentLoop.
    """

    def __init__(self, llm_responses: list[LLMResponse]) -> None:
        self._responses = iter(llm_responses)

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return "You are a test agent."

    def get_tools(self) -> list[Tool]:
        return [ThinkTool()]

    async def step(self, context: AgentContext) -> Action:
        response = next(self._responses)
        if response.tool_calls:
            return Action(
                type="tool_call",
                content=response.content,
                tool_calls=[tc.to_dict() for tc in response.tool_calls],
            )
        return Action(type="message", content=response.content)


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


# ── Backward-compatible: agent without no_tool_call_prompt ───────────────


@pytest.mark.asyncio
async def test_agent_loop_finish_immediately_no_prompt(mock_llm, mock_session):
    """Agent without get_no_tool_call_prompt: no tool calls → immediate finish."""
    agent = _PlainAgent([
        LLMResponse(
            content="The issue is already fixed.",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        ),
    ])
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
    assert result.trajectory.steps[0].action.type == "message"


@pytest.mark.asyncio
async def test_agent_loop_tool_then_finish_no_prompt(mock_llm, mock_session):
    """Agent without prompt: tool call → no tool call → finish."""
    agent = _PlainAgent([
        LLMResponse(
            content="Let me check.",
            tool_calls=[ToolCall(id="tc1", name="think", arguments='{"content": "ok"}')],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        ),
        LLMResponse(
            content="The fix is complete.",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
        ),
    ])
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
    assert result.trajectory.steps[1].action.type == "message"


# ── SearchSWEAgent: reminder behavior (agent provides prompt) ───────────


@pytest.mark.asyncio
async def test_no_tool_call_prompt_sends_reminder(mock_llm, mock_session):
    """SearchSWEAgent provides no-tool-call prompt → reminder → continue."""
    call_count = 0

    async def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: no tool call → should trigger reminder
            return LLMResponse(
                content="I think I know what's wrong.",
                tool_calls=None,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            )
        elif call_count == 2:
            # Second call: after reminder, agent calls finish tool
            return LLMResponse(
                content="Task complete.",
                tool_calls=[ToolCall(id="tc1", name="finish", arguments="{}")],
                usage=TokenUsage(prompt_tokens=50, completion_tokens=30),
            )
        # Should not reach here
        return LLMResponse(
            content="unexpected",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
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
    # Step 0: message (no tool call → reminder sent)
    assert result.trajectory.steps[0].action.type == "message"
    # Step 1: finish (agent called finish tool after reminder)
    assert result.trajectory.steps[1].action.type == "finish"
    # Verify the reminder was actually inserted into messages
    user_messages = [m for m in ctx.messages if m.role == "user"]
    reminder_messages = [
        m for m in user_messages
        if "finish" in m.content.lower() and "CRITICAL" in m.content
    ]
    assert len(reminder_messages) >= 1


@pytest.mark.asyncio
async def test_no_tool_call_prompt_bounded_by_max_steps(mock_llm, mock_session):
    """Reminders don't loop forever — bounded by max_steps."""
    async def mock_chat(messages, tools=None, **kwargs):
        # Always return text without tool calls
        return LLMResponse(
            content="Still thinking...",
            tool_calls=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
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


# ── Explicit finish via finish tool ──────────────────────────────────────


@pytest.mark.asyncio
async def test_finish_tool_terminates_loop(mock_llm, mock_session):
    """Calling the finish tool terminates the loop with finish_reason='finish'."""
    call_count = 0

    async def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return LLMResponse(
                content="Let me check.",
                tool_calls=[ToolCall(id="tc1", name="execute_bash", arguments='{"command": "ls /testbed"}')],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
            )
        else:
            return LLMResponse(
                content="Done, submitting.",
                tool_calls=[ToolCall(id="tc2", name="finish", arguments="{}")],
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
    # Verify finish tool was executed (its response is in observations)
    assert len(result.trajectory.steps[1].observations) >= 1


# ── Other existing tests ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_loop_max_steps(mock_llm, mock_session):
    """Agent hits max_steps limit."""
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(
            content="Trying again...",
            tool_calls=[ToolCall(id="tc1", name="think", arguments='{"content": "hmm"}')],
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
    """Step callbacks are invoked after each tool-call step."""
    call_count = 0
    callback_steps = []

    async def mock_chat(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return LLMResponse(
                content=f"Step {call_count}",
                tool_calls=[ToolCall(id=f"tc{call_count}", name="think", arguments='{"content": "ok"}')],
                usage=TokenUsage(prompt_tokens=10, completion_tokens=10),
            )
        # Explicit finish via finish tool
        return LLMResponse(
            content="Done",
            tool_calls=[ToolCall(id="tc_fin", name="finish", arguments="{}")],
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

    # Callbacks fire for tool_call steps (0, 1) — not for the finish step
    assert callback_steps == [0, 1]


@pytest.mark.asyncio
async def test_single_step_for_rl(mock_llm, mock_session):
    """run_single_step works for RL integration."""
    mock_llm.chat = AsyncMock(return_value=LLMResponse(
        content="Thinking...",
        tool_calls=[ToolCall(id="tc1", name="think", arguments='{"content": "analyzing"}')],
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
