from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.config.schema import AweAgentConfig
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from aweagent.core.runtime.protocol import RuntimeSession
from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.fold_search.agent import (
    BROWSECOMP_SIMPLIFIED_SYSTEM_PROMPT,
    BROWSECOMP_SYSTEM_PROMPT,
    FOLD_SEARCH_FINAL_HINT,
    FOLD_SEARCH_TOOL_SCHEMAS,
    FoldSearchAgent,
    FoldSearchConfig,
    FoldSearchLoop,
)
from aweagent.scaffold.fold_search.agent import (
    _resolve_config as _resolve_fold_search_config,
)
from aweagent.scaffold.fold_search.context import FOLD_TEXT, RuntimeTokenContextRewriter
from aweagent.scaffold.fold_search.prompts import NO_TOOL_CALL_PROMPT
from aweagent.scaffold.registry import agent_registry


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        # A deliberately small template for loop-control tests; the rendered
        # payload contract is checked separately with reasoning/tool messages.
        return list("".join(str(message.get("content") or "") for message in messages))

    def encode(self, text: str, *args, **kwargs):
        return list(str(text))


def test_context_rewriter_folds_old_tools_without_mutating_history():
    rewriter = RuntimeTokenContextRewriter(
        FakeTokenizer(), max_context_length=60, target_context_length=20,
    )
    contents = [letter * 50 for letter in "abcd"]
    messages = [
        Message(role="system", content="instructions"),
        Message(role="assistant", content="plan", reasoning_raw="reasoning"),
        *[
            Message(role="tool", content=content, tool_call_id=f"tc{i}", name="search_api")
            for i, content in enumerate(contents)
        ],
    ]

    processed = rewriter.process(messages)

    assert [m.content for m in processed[2:]] == [FOLD_TEXT, FOLD_TEXT, *contents[2:]]
    assert [m.content for m in messages[2:]] == contents
    assert processed[1].reasoning_raw == "reasoning"
    assert [m.tool_call_id for m in processed[2:]] == [f"tc{i}" for i in range(4)]


def test_context_rewriter_leaves_small_history_intact():
    rewriter = RuntimeTokenContextRewriter(
        FakeTokenizer(), max_context_length=200, target_context_length=20,
    )
    messages = [Message(role="tool", content="a" * 50) for _ in range(4)]
    processed = rewriter.process(messages)
    assert [m.content for m in processed] == ["a" * 50] * 4
    assert all(before is not after for before, after in zip(messages, processed, strict=True))


class FakeLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, "kwargs": kwargs})
        return self._responses.pop(0)


class FakeTool(Tool):
    def __init__(self, name: str, result: str) -> None:
        self._name = name
        self.result = result
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"fake {self._name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        self.calls.append(params)
        return self.result


def _agent(
    *,
    rollout_retries: int = 0,
    force_final_answer: bool = True,
    force_final_answer_remaining_tokens: int = 0,
    use_simplified_system_prompt: bool = False,
) -> FoldSearchAgent:
    return FoldSearchAgent(
        config=FoldSearchConfig(
            tokenizer_path="/fake/tokenizer",
            max_context_tokens=100,
            max_output_tokens_cap=30,
            force_final_answer_remaining_tokens=force_final_answer_remaining_tokens,
            tool_context_max_tokens=20,
            tool_context_target_tokens=5,
            use_simplified_system_prompt=use_simplified_system_prompt,
        ),
        tokenizer=FakeTokenizer(),
        rollout_retries=rollout_retries,
        force_final_answer=force_final_answer,
    )


@pytest.mark.asyncio
async def test_fold_search_tolerates_missing_usage_counters():
    llm = FakeLLM([
        LLMResponse(
            tool_calls=[ToolCall(id="done", name="finish", arguments='{"answer": "final"}')],
            usage=SimpleNamespace(prompt_tokens=None, completion_tokens=None),
        ),
    ])
    ctx = AgentContext(llm=llm, tools=[], max_steps=3)
    result = await _agent().create_loop(ctx).run("question")
    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "final"


def test_fold_search_registered():
    assert agent_registry.get("fold_search") is FoldSearchAgent


def test_fold_search_from_config_requires_tokenizer_path():
    with pytest.raises(ValueError, match="tokenizer_path"):
        FoldSearchAgent.from_config(AweAgentConfig(agent={"type": "fold_search"}))


def test_fold_search_force_answer_remaining_tokens_is_configurable():
    assert _resolve_fold_search_config({
        "tokenizer_path": "/fake/tokenizer",
    }).force_final_answer_remaining_tokens == 8192
    assert _resolve_fold_search_config({
        "tokenizer_path": "/fake/tokenizer",
        "force_final_answer_remaining_tokens": 4096,
    }).force_final_answer_remaining_tokens == 4096
    assert _resolve_fold_search_config({
        "tokenizer_path": "/fake/tokenizer",
        "force_final_answer_remaining_tokens": 0,
    }).force_final_answer_remaining_tokens == 0
    with pytest.raises(ValueError, match="non-negative integer"):
        _resolve_fold_search_config({
            "tokenizer_path": "/fake/tokenizer",
            "force_final_answer_remaining_tokens": -1,
        })


def test_fold_search_system_prompt_uses_deepsearch_prompt():
    prompt = _agent().get_system_prompt({})

    assert BROWSECOMP_SYSTEM_PROMPT.splitlines()[0] in prompt
    assert "search_api, link_summary_tool, finish" in prompt


def test_fold_search_can_use_simplified_system_prompt():
    prompt = _agent(use_simplified_system_prompt=True).get_system_prompt({})

    assert prompt == BROWSECOMP_SIMPLIFIED_SYSTEM_PROMPT.format(
        tool_names="search_api, link_summary_tool, finish"
    )
    assert "answer value must contain only the shortest" not in prompt


@pytest.mark.asyncio
async def test_fold_search_maps_search_api_and_link_summary_tool():
    search_tool = FakeTool("web_search", "search output")
    fetch_tool = FakeTool("web_fetch", "fetch output")
    llm = FakeLLM([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="s1",
                    name="search_api",
                    arguments='{"query": "\\"62.2\\" km² commune Romania"}',
                ),
                ToolCall(
                    id="f1",
                    name="link_summary_tool",
                    arguments=(
                        '{"url": "https://en.wikipedia.org/wiki/Morunglav", '
                        '"prompt": "What is the area?"}'
                    ),
                ),
            ],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
            finish_reason="tool_calls",
        ),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="done", name="finish", arguments='{"answer": "Morunglav"}')
            ],
            usage=TokenUsage(prompt_tokens=20, completion_tokens=3),
        ),
    ])
    ctx = AgentContext(
        llm=llm,
        tools=[search_tool, fetch_tool],
        task_info={"skip_patch_extraction": True},
        max_steps=3,
    )

    result = await FoldSearchLoop(_agent(), ctx).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "Morunglav"
    assert result.metadata["answer_provenance"]["finish_tool_used"] is True
    assert search_tool.calls == [{"query": '"62.2" km² commune Romania'}]
    assert fetch_tool.calls == [{
        "url": "https://en.wikipedia.org/wiki/Morunglav",
        "prompt": "What is the area?",
    }]
    assert llm.calls[0]["tools"] == FOLD_SEARCH_TOOL_SCHEMAS


@pytest.mark.asyncio
async def test_fold_search_reminds_after_message_without_tool_call():
    llm = FakeLLM([
        LLMResponse(
            content="draft answer without a tool call",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(id="done", name="finish", arguments='{"answer": "final"}')
            ],
            usage=TokenUsage(prompt_tokens=20, completion_tokens=2),
        ),
    ])
    ctx = AgentContext(
        llm=llm,
        tools=[],
        task_info={"skip_patch_extraction": True},
        max_steps=3,
    )

    result = await FoldSearchLoop(_agent(), ctx).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "final"
    assert len(llm.calls) == 2
    assert any(
        message.role == "user" and message.content == NO_TOOL_CALL_PROMPT
        for message in llm.calls[1]["messages"]
    )
    assert result.trajectory.steps[0].metadata["invalid_message_retry"] == {
        "reason": "no_tool_call",
        "prompt": NO_TOOL_CALL_PROMPT,
    }


@pytest.mark.asyncio
async def test_fold_search_retries_and_forces_finish_after_budget():
    llm = FakeLLM([
        LLMResponse(
            tool_calls=[
                ToolCall(id="s1", name="search_api", arguments='{"query": "first"}')
            ],
            usage=TokenUsage(prompt_tokens=1),
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(id="s2", name="search_api", arguments='{"query": "second"}')
            ],
            usage=TokenUsage(prompt_tokens=1),
        ),
        LLMResponse(
            tool_calls=[
                ToolCall(id="done", name="finish", arguments='{"answer": "forced"}')
            ],
            usage=TokenUsage(prompt_tokens=1),
        ),
    ])
    ctx = AgentContext(
        llm=llm,
        tools=[],
        task_info={"skip_patch_extraction": True},
        max_steps=1,
    )

    result = await FoldSearchLoop(_agent(rollout_retries=1), ctx).run("question")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "forced"
    assert result.metadata["rollout_attempts"] == 2
    assert result.metadata["answer_provenance"]["forced_final_answer"] is True
    assert llm.calls[-1]["tools"] == [FOLD_SEARCH_TOOL_SCHEMAS[-1]]
    assert FOLD_SEARCH_FINAL_HINT in str(llm.calls[-1]["messages"][-1].content)


@pytest.mark.asyncio
async def test_fold_search_forces_finish_when_remaining_context_is_below_threshold():
    llm = FakeLLM([
        LLMResponse(
            tool_calls=[
                ToolCall(id="done", name="finish", arguments='{"answer": "guarded"}')
            ],
            usage=TokenUsage(prompt_tokens=90, completion_tokens=2),
        ),
    ])
    ctx = AgentContext(
        llm=llm,
        tools=[],
        task_info={"skip_patch_extraction": True},
        max_steps=3,
    )

    result = await FoldSearchLoop(
        _agent(force_final_answer_remaining_tokens=1),
        ctx,
    ).run("question")

    provenance = result.metadata["answer_provenance"]
    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "guarded"
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] == [FOLD_SEARCH_TOOL_SCHEMAS[-1]]
    assert FOLD_SEARCH_FINAL_HINT in str(llm.calls[0]["messages"][-1].content)
    assert provenance["forced_final_answer"] is True
    assert provenance["forced_final_answer_reason"] == "remaining_context_below_threshold"
    assert provenance["context_guard_triggered"] is True
    assert provenance["context_guard_step"] == 0
    assert provenance["context_guard_threshold_tokens"] == 1
    assert provenance["context_guard_remaining_tokens"] < 1


def test_tokenizer_loading_does_not_execute_remote_repository_code(monkeypatch):
    import sys
    from types import SimpleNamespace

    from aweagent.scaffold.fold_search.context import _load_tokenizer

    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return FakeTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
    ))
    _load_tokenizer.cache_clear()
    try:
        _load_tokenizer("org/public-qwen-checkpoint")
        assert calls[0][0] == "org/public-qwen-checkpoint"
        assert calls[0][1].get("trust_remote_code", False) is False
    finally:
        _load_tokenizer.cache_clear()

@pytest.mark.parametrize("legacy_key", ["debug_search", "qwen_search"])
def test_fold_search_does_not_read_unrelated_scaffold_options(legacy_key):
    from aweagent.scaffold.fold_search.agent import _options

    config = AweAgentConfig(agent={
        "type": "fold_search",
        "tool_options": {legacy_key: {"tokenizer_path": "unrelated-model"}},
    })
    assert _options(config) == {}


def test_context_count_uses_rendered_reasoning_tools_and_template_options():
    from aweagent.scaffold.fold_search.context import count_messages_tokens

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[1]["reasoning_content"] == "check evidence"
            assert messages[1]["tool_calls"][0]["function"]["name"] == "search_api"
            assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"query": "source"}
            assert kwargs["tools"] == FOLD_SEARCH_TOOL_SCHEMAS
            assert kwargs["enable_history_reasoning"] is True
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["tokenize"] is True
            assert kwargs["return_dict"] is False
            return [1] * 123

    messages = [Message(role="user", content="question"), Message(
        role="assistant", content="", reasoning_raw="check evidence",
        tool_calls=[ToolCall(id="s1", name="search_api", arguments='{"query":"source"}')],
    )]
    assert count_messages_tokens(
        Tokenizer(), messages, tools=FOLD_SEARCH_TOOL_SCHEMAS,
        chat_template_kwargs={"enable_history_reasoning": True},
    ) == 123
