from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.config.schema import AweAgentConfig
from awe_agent.core.llm.types import LLMResponse, TokenUsage, ToolCall
from awe_agent.core.tool.code import FinishWithTextTool, ThinkTool
from awe_agent.core.tool.protocol import Tool
from awe_agent.scaffold.deepsearch.agent import DeepSearchAgent
from awe_agent.scaffold.deepsearch.loop import DeepSearchLoop
from awe_agent.scaffold.deepsearch.prompts import (
    NO_FINAL_ANSWER_FALLBACK,
    OMITTED_TOOL_RESULT,
    get_final_answer_prompts,
    get_system_prompt,
    get_user_prompt,
    resolve_prompt_keys,
)


@pytest.fixture
def mock_llm():
    return AsyncMock()


def test_deepsearch_default_tools_are_search_focused():
    agent = DeepSearchAgent()

    assert [tool.name for tool in agent.get_tools()] == [
        "web_search",
        "web_fetch",
        "think",
        "finish",
    ]
    finish = agent.get_tools()[-1]
    assert finish.parameters["required"] == ["answer"]


def test_deepsearch_create_loop_returns_deepsearch_loop(mock_llm, mock_session):
    agent = DeepSearchAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
    )

    assert isinstance(agent.create_loop(ctx), DeepSearchLoop)


def test_deepsearch_prompt_registry_default_route():
    assert resolve_prompt_keys("browsecomp", "browsecomp") == (
        "browsecomp",
        "raw",
        "default",
    )
    assert "{tool_names}" in get_system_prompt("default")
    browsecomp_prompt = get_system_prompt("browsecomp")
    assert "{tool_names}" in browsecomp_prompt
    assert "finish(answer=...)" in browsecomp_prompt
    assert "shortest final answer string" in browsecomp_prompt
    assert "reasoning, evidence, explanations, candidate lists" in browsecomp_prompt
    assert "apologies" in browsecomp_prompt
    assert get_user_prompt("raw") == "{question}"
    assert get_final_answer_prompts("default").current_history


def test_deepsearch_prompt_registry_unknown_key_errors():
    with pytest.raises(KeyError):
        get_system_prompt("missing")
    with pytest.raises(KeyError):
        get_user_prompt("missing")
    with pytest.raises(KeyError):
        get_final_answer_prompts("missing")


def test_deepsearch_system_prompt_includes_current_time():
    prompt = DeepSearchAgent().get_system_prompt(
        {"dataset_id": "browsecomp", "task_type": "browsecomp"}
    )

    assert "{tool_names}" not in prompt
    assert "web_search, web_fetch, think, finish" in prompt
    assert re.search(
        r"Current time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        prompt,
    )


def test_deepsearch_explicit_tools_override_toolset():
    config = AweAgentConfig(
        agent={
            "type": "deepsearch",
            "toolset": "default",
            "tools": ["web_search", "web_fetch_raw", "think", "finish"],
        }
    )

    agent = DeepSearchAgent.from_config(config)

    assert [tool.name for tool in agent.get_tools()] == [
        "web_search",
        "web_fetch_raw",
        "think",
        "finish",
    ]


def test_deepsearch_web_tools_accept_custom_backend_options(monkeypatch):
    from awe_agent.core.tool.search.backends.reader import reader_backend_registry
    from awe_agent.core.tool.search.backends.search import search_backend_registry

    class FakePrivateSearchBackend:
        async def search(self, **kwargs: Any) -> list[dict[str, Any]]:
            return []

    class FakePrivateReaderBackend:
        async def read_link(self, url: str) -> str:
            return ""

    monkeypatch.setitem(
        search_backend_registry._items,  # noqa: SLF001
        "private_search",
        FakePrivateSearchBackend,
    )
    monkeypatch.setitem(
        reader_backend_registry._items,  # noqa: SLF001
        "private_reader",
        FakePrivateReaderBackend,
    )
    config = AweAgentConfig(
        agent={
            "type": "deepsearch",
            "tools": ["web_search", "web_fetch", "think", "finish"],
            "tool_options": {
                "web_search": {"backend": "private_search", "engine": "google"},
                "web_fetch": {"reader_backend": "private_reader"},
            },
        }
    )

    agent = DeepSearchAgent.from_config(config)

    assert [tool.name for tool in agent.get_tools()] == [
        "web_search",
        "web_fetch",
        "think",
        "finish",
    ]


@pytest.mark.asyncio
async def test_deepsearch_finish_records_final_answer(mock_llm, mock_session):
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(
            content="Submitting.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="finish",
                    arguments='{"answer": "Paris"}',
                )
            ],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

    mock_llm.chat = mock_chat
    agent = DeepSearchAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=3,
    )

    result = await agent.create_loop(ctx).run("What is the capital of France?")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "Paris"
    assert result.metadata["agent_submitted_final_answer"] is True
    assert result.metadata["forced_final_answer"] is False


class _RetryAgent(Agent):
    def __init__(self) -> None:
        self.user_message_counts: list[int] = []
        self.calls = 0

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return "test"

    def get_tools(self) -> list[Tool]:
        return [ThinkTool(), FinishWithTextTool()]

    async def step(self, context: AgentContext) -> Action:
        self.calls += 1
        self.user_message_counts.append(
            len([m for m in context.messages if m.role == "user"])
        )
        if self.calls == 1:
            return Action(
                type="tool_call",
                content="thinking",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "think",
                        "arguments": '{"content": "not enough"}',
                    }
                ],
            )
        return Action(
            type="finish",
            content="done",
            tool_calls=[
                {
                    "id": "tc2",
                    "name": "finish",
                    "arguments": '{"answer": "retry answer"}',
                }
            ],
        )


@pytest.mark.asyncio
async def test_rollout_retry_restarts_from_original_prompt(mock_llm, mock_session):
    agent = _RetryAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await DeepSearchLoop(
        agent,
        ctx,
        rollout_retries=1,
        force_final_answer=False,
    ).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "retry answer"
    assert agent.user_message_counts == [1, 1]


class _AlwaysSearchingAgent(_RetryAgent):
    async def step(self, context: AgentContext) -> Action:
        self.calls += 1
        return Action(
            type="tool_call",
            content="thinking",
            tool_calls=[
                {
                    "id": f"tc{self.calls}",
                    "name": "think",
                    "arguments": '{"content": "still searching"}',
                }
            ],
        )


class _MalformedFinishAgent(_RetryAgent):
    async def step(self, context: AgentContext) -> Action:
        return Action(
            type="finish",
            content="malformed finish",
            tool_calls=[
                {
                    "id": "bad_finish",
                    "name": "finish",
                    "arguments": '{"not_answer": "missing answer field"}',
                }
            ],
        )


@pytest.mark.asyncio
async def test_force_final_answer_after_all_retries_fail(mock_llm, mock_session):
    chat_calls: list[dict[str, Any]] = []

    async def mock_chat(messages, tools=None, **kwargs):
        chat_calls.append({"messages": messages, "tools": tools})
        return LLMResponse(content="best guess")

    mock_llm.chat = mock_chat
    agent = _AlwaysSearchingAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await DeepSearchLoop(
        agent,
        ctx,
        rollout_retries=1,
        force_final_answer=True,
    ).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "best guess"
    assert result.metadata["forced_final_answer"] is True
    assert result.metadata["agent_submitted_final_answer"] is False
    assert result.metadata["forced_final_answer_stage"] == "current_history"
    assert chat_calls[-1]["tools"] is None
    assert len(result.trajectory.steps) == 2


@pytest.mark.asyncio
async def test_force_final_answer_when_finish_has_no_answer(mock_llm, mock_session):
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(content="extracted from malformed finish")

    mock_llm.chat = mock_chat
    agent = _MalformedFinishAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await DeepSearchLoop(
        agent,
        ctx,
        rollout_retries=0,
        force_final_answer=True,
    ).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "extracted from malformed finish"
    assert result.metadata["agent_submitted_final_answer"] is False
    assert result.metadata["forced_final_answer"] is True
    assert result.metadata["forced_final_answer_reason"] == "finish"


@pytest.mark.asyncio
async def test_force_final_answer_falls_back_to_folded_history(
    mock_llm,
    mock_session,
):
    async def mock_chat(messages, tools=None, **kwargs):
        if len(mock_chat.calls) == 0:
            mock_chat.calls.append(messages)
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="bad_tool_call",
                        name="think",
                        arguments='{"content": "more search"}',
                    )
                ]
            )
        mock_chat.calls.append(messages)
        return LLMResponse(content="folded history answer")

    mock_chat.calls = []
    mock_llm.chat = mock_chat
    agent = _AlwaysSearchingAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await DeepSearchLoop(
        agent,
        ctx,
        rollout_retries=0,
        force_final_answer=True,
    ).run("question")

    assert result.metadata["final_answer"] == "folded history answer"
    assert result.metadata["forced_final_answer"] is True
    assert result.metadata["forced_final_answer_stage"] == "folded_full_history"
    assert len(mock_chat.calls) == 2
    folded_messages = mock_chat.calls[1]
    assert any(
        message.content == OMITTED_TOOL_RESULT
        for message in folded_messages
        if message.role == "tool"
    )


@pytest.mark.asyncio
async def test_force_final_answer_records_fallback_when_extraction_empty(
    mock_llm,
    mock_session,
):
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(content="")

    mock_llm.chat = mock_chat
    agent = _AlwaysSearchingAgent()
    ctx = AgentContext(
        llm=mock_llm,
        session=mock_session,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await DeepSearchLoop(
        agent,
        ctx,
        rollout_retries=0,
        force_final_answer=True,
    ).run("question")

    assert result.metadata["final_answer"] == NO_FINAL_ANSWER_FALLBACK
    assert result.metadata["forced_final_answer"] is True
    assert result.metadata["forced_final_answer_stage"] == "no_answer_fallback"
