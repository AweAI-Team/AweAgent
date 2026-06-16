from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.training import TrainingState
from aweagent.core.agent.trajectory import Action
from aweagent.core.config.schema import AweAgentConfig
from aweagent.core.llm.types import LLMResponse, TokenUsage, ToolCall
from aweagent.core.tool.code import FinishWithTextTool, ThinkTool
from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.deepsearch.agent import DeepSearchAgent, _resolve_tool_names
from aweagent.scaffold.deepsearch.policy import RetryThenForceAnswerPolicy
from aweagent.scaffold.deepsearch.prompts import (
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


def _force_answer_loop(
    agent: Agent,
    ctx: AgentContext,
    *,
    rollout_retries: int = 0,
    force_final_answer: bool = True,
) -> AgentLoop:
    """An AgentLoop carrying the DeepSearch retry/force-answer policy."""
    return AgentLoop(
        agent,
        ctx,
        policy=RetryThenForceAnswerPolicy(
            rollout_retries=rollout_retries,
            force_final_answer=force_final_answer,
        ),
    )


def test_deepsearch_default_tools_are_search_focused():
    agent = DeepSearchAgent()

    assert [tool.name for tool in agent.get_tools()] == [
        "web_search",
        "web_fetch",
        "finish",
    ]
    finish = agent.get_tools()[-1]
    assert finish.parameters["required"] == ["answer"]


def test_deepsearch_create_loop_carries_force_answer_policy(mock_llm):
    agent = DeepSearchAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
    )

    loop = agent.create_loop(ctx)
    assert isinstance(loop, AgentLoop)
    assert isinstance(loop.policy, RetryThenForceAnswerPolicy)


def test_deepsearch_prompt_registry_default_route():
    assert resolve_prompt_keys("browsecomp") == (
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
    assert "think to organize" not in browsecomp_prompt
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
    assert "web_search, web_fetch, finish" in prompt
    assert "think" not in prompt.lower()
    assert re.search(
        r"Current time: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        prompt,
    )


def test_resolve_tool_names_defaults_to_toolset_when_unset():
    names = _resolve_tool_names(AweAgentConfig(agent={"type": "deepsearch"}))
    assert names == ["web_search", "web_fetch", "finish"]


def test_resolve_tool_names_honors_explicit_list():
    # Any explicit list is honored, including one identical to another
    # scaffold's default tools.
    config = AweAgentConfig(
        agent={
            "type": "deepsearch",
            "tools": ["execute_bash", "str_replace_editor", "think"],
        }
    )
    assert _resolve_tool_names(config) == ["execute_bash", "str_replace_editor", "think"]


def test_deepsearch_explicit_tools_override_toolset():
    config = AweAgentConfig(
        agent={
            "type": "deepsearch",
            "toolset": "default",
            "tools": ["web_search", "web_fetch_raw", "finish"],
        }
    )

    agent = DeepSearchAgent.from_config(config)

    assert [tool.name for tool in agent.get_tools()] == [
        "web_search",
        "web_fetch_raw",
        "finish",
    ]


def test_deepsearch_web_tools_accept_custom_backend_options(monkeypatch):
    from aweagent.core.tool.search.backends.reader import reader_backend_registry
    from aweagent.core.tool.search.backends.search import search_backend_registry

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
            "tools": ["web_search", "web_fetch", "finish"],
            "tool_options": {
                "web_search": {"backend": "private_search", "engine": "google"},
                "web_fetch": {
                    "reader_backend": "private_reader",
                    "llm": {
                        "backend": "openai",
                        "model": "web-fetch-model",
                        "api_key": "web-fetch-key",
                    },
                },
            },
        }
    )

    agent = DeepSearchAgent.from_config(config)
    web_search, web_fetch, _finish = agent.get_tools()

    # The configured backend/engine/reader are actually wired into the tools.
    assert isinstance(web_search._backend, FakePrivateSearchBackend)  # noqa: SLF001
    assert web_search._engine == "google"  # noqa: SLF001
    assert isinstance(web_fetch._reader._backend, FakePrivateReaderBackend)  # noqa: SLF001
    assert web_fetch._llm_config_inline["model"] == "web-fetch-model"  # noqa: SLF001


@pytest.mark.asyncio
async def test_deepsearch_finish_records_final_answer(mock_llm):
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
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=3,
    )

    result = await agent.create_loop(ctx).run("What is the capital of France?")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "Paris"
    provenance = result.metadata["answer_provenance"]
    assert provenance["agent_submitted_final_answer"] is True
    assert provenance["forced_final_answer"] is False


@pytest.mark.asyncio
async def test_deepsearch_step_passes_training_input_ids(mock_llm):
    seen: dict[str, Any] = {}

    async def chat(messages, tools=None, **kwargs):
        seen["input_ids"] = kwargs.get("input_ids")
        return LLMResponse(
            content="done",
            tool_calls=[ToolCall(id="t", name="finish", arguments='{"answer": "a"}')],
        )

    mock_llm.chat = chat
    agent = DeepSearchAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        training=TrainingState(tokenizer=None, prompt_token_ids=[1, 2, 3]),
    )

    await agent.step(ctx)

    assert seen["input_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_deepsearch_step_stops_on_length_in_training(mock_llm):
    calls = 0

    async def chat(messages, tools=None, **kwargs):
        nonlocal calls
        calls += 1
        return LLMResponse(content="", finish_reason="length", finish_status="length")

    mock_llm.chat = chat
    agent = DeepSearchAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        training=TrainingState(tokenizer=None, prompt_token_ids=[1]),
    )

    await agent.step(ctx)

    # An exhausted token budget is terminal in training, so the empty-response
    # retry loop must not spin.
    assert calls == 1


@pytest.mark.asyncio
async def test_deepsearch_step_omits_input_ids_without_training(mock_llm):
    seen: dict[str, Any] = {}

    async def chat(messages, tools=None, **kwargs):
        seen["kwargs"] = kwargs
        return LLMResponse(content="hi")

    mock_llm.chat = chat
    agent = DeepSearchAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
    )

    await agent.step(ctx)

    assert "input_ids" not in seen["kwargs"]


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
async def test_rollout_retry_restarts_from_original_prompt(mock_llm):
    agent = _RetryAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await _force_answer_loop(
        agent, ctx, rollout_retries=1, force_final_answer=False,
    ).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "retry answer"
    assert result.metadata["answer_provenance"]["agent_submitted_final_answer"] is True
    # Each rollout starts fresh: a single user message at the first step.
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


class _EmptyAnswerFinishAgent(_RetryAgent):
    async def step(self, context: AgentContext) -> Action:
        return Action(
            type="finish",
            content="empty answer",
            tool_calls=[
                {
                    "id": "empty_finish",
                    "name": "finish",
                    "arguments": '{"answer": "   "}',
                }
            ],
        )


@pytest.mark.asyncio
async def test_force_final_answer_after_all_retries_fail(mock_llm):
    chat_calls: list[dict[str, Any]] = []

    async def mock_chat(messages, tools=None, **kwargs):
        chat_calls.append({"messages": messages, "tools": tools})
        return LLMResponse(content="best guess")

    mock_llm.chat = mock_chat
    agent = _AlwaysSearchingAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await _force_answer_loop(
        agent, ctx, rollout_retries=1, force_final_answer=True,
    ).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "best guess"
    provenance = result.metadata["answer_provenance"]
    assert provenance["forced_final_answer"] is True
    assert provenance["agent_submitted_final_answer"] is False
    assert provenance["forced_final_answer_stage"] == "current_history"
    assert chat_calls[-1]["tools"] is None
    assert len(result.trajectory.steps) == 2


@pytest.mark.asyncio
async def test_force_final_answer_when_finish_has_no_answer(mock_llm):
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(content="extracted from malformed finish")

    mock_llm.chat = mock_chat
    agent = _MalformedFinishAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await _force_answer_loop(agent, ctx).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "extracted from malformed finish"
    provenance = result.metadata["answer_provenance"]
    assert provenance["agent_submitted_final_answer"] is False
    assert provenance["forced_final_answer"] is True
    assert provenance["forced_final_answer_reason"] == "finish"


@pytest.mark.asyncio
async def test_empty_submitted_answer_triggers_forced_extraction(mock_llm):
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(content="recovered answer")

    mock_llm.chat = mock_chat
    agent = _EmptyAnswerFinishAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await _force_answer_loop(agent, ctx).run("question")

    # An empty submission is not a usable answer; extraction recovers one.
    assert result.metadata["final_answer"] == "recovered answer"
    provenance = result.metadata["answer_provenance"]
    assert provenance["agent_submitted_final_answer"] is False
    assert provenance["forced_final_answer"] is True


@pytest.mark.asyncio
async def test_force_final_answer_falls_back_to_folded_history(mock_llm):
    async def mock_chat(messages, tools=None, **kwargs):
        mock_chat.calls.append(messages)
        if len(mock_chat.calls) == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="bad_tool_call",
                        name="think",
                        arguments='{"content": "more search"}',
                    )
                ]
            )
        return LLMResponse(content="folded history answer")

    mock_chat.calls = []
    mock_llm.chat = mock_chat
    agent = _AlwaysSearchingAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await _force_answer_loop(agent, ctx).run("question")

    assert result.metadata["final_answer"] == "folded history answer"
    provenance = result.metadata["answer_provenance"]
    assert provenance["forced_final_answer"] is True
    assert provenance["forced_final_answer_stage"] == "folded_full_history"
    assert len(mock_chat.calls) == 2

    folded_messages = mock_chat.calls[1]
    assert any(
        message.content == OMITTED_TOOL_RESULT
        for message in folded_messages
        if message.role == "tool"
    )
    # The folded view is built from the pre-extraction baseline, so the
    # first-pass extraction instruction never leaks into it.
    current_prompt = get_final_answer_prompts("default").current_history
    assert all(message.content != current_prompt for message in folded_messages)


@pytest.mark.asyncio
async def test_force_final_answer_records_fallback_when_extraction_empty(mock_llm):
    async def mock_chat(messages, tools=None, **kwargs):
        return LLMResponse(content="")

    mock_llm.chat = mock_chat
    agent = _AlwaysSearchingAgent()
    ctx = AgentContext(
        llm=mock_llm,
        tools=agent.get_tools(),
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await _force_answer_loop(agent, ctx).run("question")

    assert result.metadata["final_answer"] == NO_FINAL_ANSWER_FALLBACK
    provenance = result.metadata["answer_provenance"]
    assert provenance["forced_final_answer"] is True
    assert provenance["forced_final_answer_stage"] == "no_answer_fallback"
