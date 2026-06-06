"""Tests for LLM backend enhancements.

Covers:
- Anthropic: full content block round-trip (order preservation)
- Anthropic: thinking config (effort → adaptive, budget → manual, default)
- OpenAI Response API: previous_response_id continuation
- OpenAI Response API: incomplete_details.reason normalization
- Trajectory: llm_response_raw flows through Action → TrajectoryStep
- Message: reasoning_text vs reasoning_raw semantic split
- create_async_client rejects unsupported backends
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall


# ═══════════════════════════════════════════════════════════════════════
# Anthropic: full content block order round-trip
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_block(type: str, **kwargs: Any) -> Any:
    """Create a mock Anthropic content block object."""

    @dataclass
    class MockBlock:
        type: str

    block = MockBlock(type=type)
    for k, v in kwargs.items():
        setattr(block, k, v)
    return block


def _make_mock_anthropic_response(
    content_blocks: list[Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Any:
    """Create a mock Anthropic response object."""

    @dataclass
    class MockUsage:
        input_tokens: int
        output_tokens: int
        cache_read_input_tokens: int = 0

    class MockResponse:
        def __init__(self, content, stop_reason, usage):
            self.content = content
            self.stop_reason = stop_reason
            self.usage = usage

    return MockResponse(
        content=content_blocks,
        stop_reason=stop_reason,
        usage=MockUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_anthropic_parse_preserves_all_blocks_in_order():
    """_parse_response stores ALL content blocks in reasoning_raw, preserving order."""
    from aweagent.core.llm.backends.anthropic import AnthropicBackend

    # Simulate interleaved response: thinking → text → thinking → tool_use → text
    blocks = [
        _make_mock_block("thinking", thinking="Let me analyze...", signature="sig_1"),
        _make_mock_block("text", text="First part"),
        _make_mock_block("thinking", thinking="Now for the tool...", signature="sig_2"),
        _make_mock_block(
            "tool_use", id="call_1", name="search", input={"q": "test"}
        ),
        _make_mock_block("text", text="Second part"),
    ]
    response = _make_mock_anthropic_response(blocks)

    backend = object.__new__(AnthropicBackend)
    result = backend._parse_response(response)

    # reasoning_raw should have ALL 5 blocks in original order
    assert result.reasoning_raw is not None
    assert len(result.reasoning_raw) == 5
    types = [b["type"] for b in result.reasoning_raw]
    assert types == ["thinking", "text", "thinking", "tool_use", "text"]

    # Verify block details
    assert result.reasoning_raw[0]["signature"] == "sig_1"
    assert result.reasoning_raw[1]["text"] == "First part"
    assert result.reasoning_raw[3]["name"] == "search"
    assert result.reasoning_raw[4]["text"] == "Second part"

    # reasoning_text is the plain-text summary, reasoning_raw is provider payload
    assert result.content == "First part\nSecond part"
    assert result.reasoning_text == "Let me analyze...\nNow for the tool..."


def test_anthropic_parse_preserves_redacted_thinking():
    """_parse_response stores redacted_thinking blocks in reasoning_raw."""
    from aweagent.core.llm.backends.anthropic import AnthropicBackend

    blocks = [
        _make_mock_block("thinking", thinking="Visible thought", signature="sig_a"),
        _make_mock_block("redacted_thinking", data="encrypted_data_here"),
        _make_mock_block("text", text="Output"),
    ]
    response = _make_mock_anthropic_response(blocks)

    backend = object.__new__(AnthropicBackend)
    result = backend._parse_response(response)

    assert len(result.reasoning_raw) == 3
    assert result.reasoning_raw[1] == {
        "type": "redacted_thinking",
        "data": "encrypted_data_here",
    }


def test_anthropic_serialize_replays_full_blocks():
    """_serialize_messages replays full blocks directly when available."""
    from aweagent.core.llm.backends.anthropic import AnthropicBackend

    full_blocks = [
        {"type": "thinking", "thinking": "thought 1", "signature": "sig_1"},
        {"type": "text", "text": "First response"},
        {"type": "thinking", "thinking": "thought 2", "signature": "sig_2"},
        {"type": "tool_use", "id": "call_1", "name": "bash", "input": {"cmd": "ls"}},
    ]

    messages = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="First response",
            tool_calls=[ToolCall(id="call_1", name="bash", arguments='{"cmd":"ls"}')],
            reasoning_raw=full_blocks,
        ),
        Message(role="tool", content="output", tool_call_id="call_1"),
    ]

    backend = object.__new__(AnthropicBackend)
    from aweagent.core.llm.config import LLMConfig

    backend.config = LLMConfig(backend="anthropic", model="claude-sonnet-4-6")
    system, anthropic_msgs = backend._serialize_messages(messages)

    assert len(anthropic_msgs) == 3
    assistant_msg = anthropic_msgs[1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["content"]) == 4
    types = [b["type"] for b in assistant_msg["content"]]
    assert types == ["thinking", "text", "thinking", "tool_use"]
    assert assistant_msg["content"][0]["signature"] == "sig_1"


def test_anthropic_serialize_legacy_thinking_only():
    """_serialize_messages falls back to reconstruction for thinking-only blocks."""
    from aweagent.core.llm.backends.anthropic import AnthropicBackend

    thinking_blocks = [
        {"type": "thinking", "thinking": "I think...", "signature": "sig"},
    ]

    messages = [
        Message(
            role="assistant",
            content="Some text",
            reasoning_raw=thinking_blocks,
        ),
    ]

    backend = object.__new__(AnthropicBackend)
    from aweagent.core.llm.config import LLMConfig

    backend.config = LLMConfig(backend="anthropic", model="claude-sonnet-4-6")
    system, anthropic_msgs = backend._serialize_messages(messages)

    content = anthropic_msgs[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "thinking"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "Some text"


def test_anthropic_serialize_no_reasoning():
    """_serialize_messages works without reasoning_raw."""
    from aweagent.core.llm.backends.anthropic import AnthropicBackend

    messages = [
        Message(role="assistant", content="Just text"),
    ]

    backend = object.__new__(AnthropicBackend)
    from aweagent.core.llm.config import LLMConfig

    backend.config = LLMConfig(backend="anthropic", model="claude-sonnet-4-6")
    system, anthropic_msgs = backend._serialize_messages(messages)

    content = anthropic_msgs[0]["content"]
    assert len(content) == 1
    assert content[0] == {"type": "text", "text": "Just text"}


# ═══════════════════════════════════════════════════════════════════════
# Anthropic: thinking config (thinking_type, budget, model validation)
# ═══════════════════════════════════════════════════════════════════════


def _make_anthropic_backend(
    thinking_budget: int | None = None,
    thinking_type: str | None = None,
    **config_kw: Any,
):
    """Create an AnthropicBackend with config, skipping SDK import."""
    from aweagent.core.llm.backends.anthropic import AnthropicBackend
    from aweagent.core.llm.config import LLMConfig

    config = LLMConfig(
        backend="anthropic", model=config_kw.pop("model", "claude-sonnet-4-6"),
        thinking=True, thinking_type=thinking_type,
        thinking_budget=thinking_budget,
        params={"max_tokens": 16384},
        **config_kw,
    )
    backend = object.__new__(AnthropicBackend)
    backend.config = config
    return backend


def test_adaptive_on_claude_sonnet_46():
    """thinking_type=adaptive + Claude Sonnet 4.6 → adaptive."""
    backend = _make_anthropic_backend(thinking_type="adaptive")
    assert backend._resolve_thinking({}) == {"type": "adaptive"}


def test_adaptive_on_claude_opus_46():
    """thinking_type=adaptive + Claude Opus 4.6 → adaptive."""
    backend = _make_anthropic_backend(
        thinking_type="adaptive", model="claude-opus-4-6-20260301",
    )
    assert backend._resolve_thinking({}) == {"type": "adaptive"}


def test_adaptive_on_non_46_model_raises():
    """thinking_type=adaptive on non-4.6 model → ValueError, not silent fallback."""
    backend = _make_anthropic_backend(
        thinking_type="adaptive", model="claude-3-5-sonnet-20241022",
    )
    with pytest.raises(ValueError, match="only supported by Claude 4.6"):
        backend._resolve_thinking({})


def test_manual_with_explicit_budget():
    """Explicit thinking_budget → manual mode."""
    backend = _make_anthropic_backend(thinking_budget=12000)
    assert backend._resolve_thinking({}) == {"type": "enabled", "budget_tokens": 12000}


def test_manual_with_runtime_budget_override():
    """Runtime budget_tokens in thinking_config overrides config."""
    backend = _make_anthropic_backend(thinking_budget=5000)
    result = backend._resolve_thinking({"budget_tokens": 20000})
    assert result == {"type": "enabled", "budget_tokens": 20000}


def test_no_thinking_type_no_budget_raises():
    """thinking=True but neither thinking_type nor thinking_budget → ValueError."""
    backend = _make_anthropic_backend()
    with pytest.raises(ValueError, match="requires either thinking_type or thinking_budget"):
        backend._resolve_thinking({})


def test_no_thinking_type_no_budget_bool_config_raises():
    """thinking_config is True (bool) + no budget → ValueError."""
    backend = _make_anthropic_backend()
    with pytest.raises(ValueError, match="requires either thinking_type or thinking_budget"):
        backend._resolve_thinking(True)


def test_runtime_override_type_adaptive():
    """Runtime thinking={"type": "adaptive"} overrides config."""
    backend = _make_anthropic_backend(thinking_budget=8000)  # config says manual
    result = backend._resolve_thinking({"type": "adaptive"})
    assert result == {"type": "adaptive"}


def test_runtime_override_type_on_non_46_raises():
    """Runtime adaptive override on non-4.6 model still raises."""
    backend = _make_anthropic_backend(
        thinking_budget=8000, model="claude-3-5-sonnet-20241022",
    )
    with pytest.raises(ValueError, match="only supported by Claude 4.6"):
        backend._resolve_thinking({"type": "adaptive"})


def test_thinking_budget_zero_raises():
    """thinking_budget=0 → ValueError at runtime."""
    backend = _make_anthropic_backend(thinking_budget=1)  # pass config validation
    # Simulate runtime budget of 0 via thinking_config
    with pytest.raises(ValueError, match="positive integer"):
        backend._resolve_thinking({"budget_tokens": 0})


def test_thinking_budget_negative_raises():
    """thinking_budget=-1 → ValueError at config validation."""
    from pydantic import ValidationError
    from aweagent.core.llm.config import LLMConfig

    with pytest.raises(ValidationError):
        LLMConfig(backend="anthropic", model="claude-sonnet-4-6", thinking_budget=-1)


def test_thinking_budget_zero_rejected_by_config():
    """thinking_budget=0 → rejected by Pydantic (gt=0)."""
    from pydantic import ValidationError
    from aweagent.core.llm.config import LLMConfig

    with pytest.raises(ValidationError):
        LLMConfig(backend="anthropic", model="claude-sonnet-4-6", thinking_budget=0)


def test_invalid_thinking_type_rejected_by_config():
    """Misspelled thinking_type → rejected by Pydantic Literal validation."""
    from pydantic import ValidationError
    from aweagent.core.llm.config import LLMConfig

    with pytest.raises(ValidationError):
        LLMConfig(backend="anthropic", model="claude-sonnet-4-6", thinking_type="adpative")


def test_content_block_roundtrip_is_provider_agnostic():
    """Content block serialization works for any Anthropic-compatible provider.

    MiniMax via Anthropic-compatible API uses the same content block format.
    No provider detection needed — the parsing/serialization is protocol-level.
    """
    full_blocks = [
        {"type": "thinking", "thinking": "reasoning here", "signature": "sig_1"},
        {"type": "text", "text": "answer"},
        {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "x"}},
    ]

    messages = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant", content="answer",
            tool_calls=[ToolCall(id="call_1", name="search", arguments='{"q":"x"}')],
            reasoning_raw=full_blocks,
        ),
        Message(role="tool", content="result", tool_call_id="call_1"),
    ]

    # Use MiniMax model — no thinking config, no profile detection
    backend = _make_anthropic_backend(
        model="MiniMax-M2.5", thinking_budget=8000,  # needed to pass _resolve_thinking
    )
    _, anthropic_msgs = backend._serialize_messages(messages)

    assistant_content = anthropic_msgs[1]["content"]
    assert len(assistant_content) == 3
    assert [b["type"] for b in assistant_content] == ["thinking", "text", "tool_use"]
    assert assistant_content[0]["signature"] == "sig_1"


def test_minimax_anthropic_preset_no_thinking():
    """MiniMax anthropic preset doesn't set thinking — no thinking param sent.

    This verifies the design: MiniMax doesn't get Claude thinking control
    because the preset simply doesn't configure thinking: true.
    """
    from aweagent.core.llm.config import LLMConfig

    config = LLMConfig(
        backend="anthropic",
        base_url="https://api.minimax.io/anthropic",
        model="MiniMax-M2.5",
        params={"max_tokens": 8192, "temperature": 1.0},
    )
    # thinking defaults to False — no thinking param will be sent
    assert config.thinking is False
    assert config.thinking_type is None
    assert config.thinking_budget is None


# ═══════════════════════════════════════════════════════════════════════
# Response API: previous_response_id continuation
# ═══════════════════════════════════════════════════════════════════════


def test_response_api_extracts_previous_response_id():
    """_serialize_input extracts response_id from reasoning_raw."""
    from aweagent.core.llm.config import LLMConfig
    from aweagent.core.llm.backends.openai_response import OpenAIResponseBackend

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="Hi there",
            reasoning_raw={
                "response_id": "resp_abc123",
                "reasoning_items": None,
            },
        ),
        Message(role="user", content="Follow up"),
    ]

    backend = object.__new__(OpenAIResponseBackend)
    backend.config = LLMConfig()
    instructions, items, prev_id = backend._serialize_input(messages)

    assert prev_id == "resp_abc123"
    assert instructions == "You are helpful."
    # Continuation mode: only new input after last response
    assert len(items) == 1
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Follow up"


def test_response_api_continuation_sends_only_new_input():
    """With previous_response_id, only new items after last response are sent."""
    from aweagent.core.llm.config import LLMConfig
    from aweagent.core.llm.backends.openai_response import OpenAIResponseBackend

    messages = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="Hi there",
            reasoning_raw={
                "response_id": "resp_abc",
                "reasoning_items": [
                    {"type": "reasoning", "id": "r_1", "encrypted_content": "enc"},
                ],
            },
        ),
        Message(role="user", content="Follow up"),
    ]

    backend = object.__new__(OpenAIResponseBackend)
    backend.config = LLMConfig()
    _, items, prev_id = backend._serialize_input(messages)

    assert prev_id == "resp_abc"
    # Only the new user message should be in items — no old reasoning/assistant
    assert len(items) == 1
    assert items[0]["role"] == "user"
    assert items[0]["content"] == "Follow up"


def test_response_api_manual_mode_replays_reasoning():
    """Without response_id, reasoning items are replayed in full."""
    from aweagent.core.llm.config import LLMConfig
    from aweagent.core.llm.backends.openai_response import OpenAIResponseBackend

    messages = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant",
            content="Hi there",
            reasoning_raw={
                "response_id": None,  # No response_id → manual mode
                "reasoning_items": [
                    {"type": "reasoning", "id": "r_1", "encrypted_content": "enc"},
                ],
            },
        ),
        Message(role="user", content="Follow up"),
    ]

    backend = object.__new__(OpenAIResponseBackend)
    backend.config = LLMConfig()
    _, items, prev_id = backend._serialize_input(messages)

    assert prev_id is None
    # All items should be present: user + reasoning + assistant + user
    reasoning_in_items = [i for i in items if i.get("type") == "reasoning"]
    assert len(reasoning_in_items) == 1
    assert reasoning_in_items[0]["encrypted_content"] == "enc"
    user_items = [i for i in items if i.get("role") == "user"]
    assert len(user_items) == 2


def test_response_api_no_previous_response_id():
    """_serialize_input returns None when no response_id available."""
    from aweagent.core.llm.config import LLMConfig
    from aweagent.core.llm.backends.openai_response import OpenAIResponseBackend

    messages = [
        Message(role="user", content="Hello"),
    ]

    backend = object.__new__(OpenAIResponseBackend)
    backend.config = LLMConfig()
    _, _, prev_id = backend._serialize_input(messages)
    assert prev_id is None


# ═══════════════════════════════════════════════════════════════════════
# Response API: incomplete_details.reason normalization
# ═══════════════════════════════════════════════════════════════════════


def test_normalize_response_status_completed():
    from aweagent.core.llm.backends.openai_response import (
        _normalize_response_status,
    )

    assert _normalize_response_status("completed") == "stop"


def test_normalize_response_status_failed():
    from aweagent.core.llm.backends.openai_response import (
        _normalize_response_status,
    )

    assert _normalize_response_status("failed") == "error"


def test_normalize_response_status_incomplete_max_tokens():
    """incomplete with max_output_tokens reason → 'length'."""
    from aweagent.core.llm.backends.openai_response import (
        _normalize_response_status,
    )

    @dataclass
    class IncompleteDetails:
        reason: str = "max_output_tokens"

    result = _normalize_response_status("incomplete", IncompleteDetails())
    assert result == "length"


def test_normalize_response_status_incomplete_content_filter():
    """incomplete with content_filter reason → preserves reason, NOT 'length'."""
    from aweagent.core.llm.backends.openai_response import (
        _normalize_response_status,
    )

    @dataclass
    class IncompleteDetails:
        reason: str = "content_filter"

    result = _normalize_response_status("incomplete", IncompleteDetails())
    assert result == "content_filter"
    assert result != "length"


def test_normalize_response_status_incomplete_no_details():
    """incomplete without details → 'incomplete' (not 'length')."""
    from aweagent.core.llm.backends.openai_response import (
        _normalize_response_status,
    )

    result = _normalize_response_status("incomplete", None)
    assert result == "incomplete"
    assert result != "length"


def test_normalize_response_status_none():
    from aweagent.core.llm.backends.openai_response import (
        _normalize_response_status,
    )

    assert _normalize_response_status(None) is None


# ═══════════════════════════════════════════════════════════════════════
# Model preset → request body mapping
# ═══════════════════════════════════════════════════════════════════════


def test_glm5_clear_thinking_nested_in_thinking_dict():
    """GLM-5: clear_thinking must be nested inside the thinking dict, not a sibling."""
    from aweagent.core.llm.backends.openai import OpenAIBackend
    from aweagent.core.llm.config import LLMConfig

    config = LLMConfig(
        backend="openai",
        model="glm-5",
        thinking=True,
        extra={"clear_thinking": False},
        params={"max_tokens": 20480, "temperature": 1.0},
    )
    backend = OpenAIBackend(config)

    messages = [Message(role="user", content="Hello")]
    # Simulate what LLMClient passes when thinking=True
    params = backend._build_params(messages, thinking={"type": "enabled"})

    extra_body = params.get("extra_body", {})
    # clear_thinking must be INSIDE thinking, not alongside it
    assert "clear_thinking" not in extra_body, (
        "clear_thinking should be nested inside thinking dict, not a sibling key"
    )
    assert isinstance(extra_body.get("thinking"), dict)
    assert extra_body["thinking"]["clear_thinking"] is False
    assert extra_body["thinking"]["type"] == "enabled"


def test_minimax_reasoning_split_stays_separate():
    """MiniMax: reasoning_split stays as a sibling key (not merged into thinking)."""
    from aweagent.core.llm.backends.openai import OpenAIBackend
    from aweagent.core.llm.config import LLMConfig

    config = LLMConfig(
        backend="openai",
        model="MiniMax-M2.5",
        extra={"reasoning_split": True},
        params={"temperature": 1.0, "max_tokens": 8192},
    )
    backend = OpenAIBackend(config)

    messages = [Message(role="user", content="Hello")]
    params = backend._build_params(messages)

    extra_body = params.get("extra_body", {})
    assert extra_body.get("reasoning_split") is True


def test_qwen_enable_thinking_in_extra_body():
    """Qwen: enable_thinking goes into extra_body."""
    from aweagent.core.llm.backends.openai import OpenAIBackend
    from aweagent.core.llm.config import LLMConfig

    config = LLMConfig(
        backend="openai",
        model="qwen3.5-plus",
        extra={"enable_thinking": True},
        params={"temperature": 0.6, "max_tokens": 16384},
    )
    backend = OpenAIBackend(config)

    messages = [Message(role="user", content="Hello")]
    params = backend._build_params(messages)

    extra_body = params.get("extra_body", {})
    assert extra_body.get("enable_thinking") is True


# ═══════════════════════════════════════════════════════════════════════
# Trajectory: llm_response_raw flows through data chain
# ═══════════════════════════════════════════════════════════════════════


def test_action_carries_llm_response_raw():
    """Action dataclass includes llm_response_raw field."""
    raw_obj = {"model": "gpt-4o", "id": "resp_123"}
    action = Action(type="tool_call", content="hello", llm_response_raw=raw_obj)
    assert action.llm_response_raw is raw_obj


def test_trajectory_step_receives_llm_response_raw():
    """TrajectoryStep.llm_response_raw is set via add_step kwargs."""
    traj = Trajectory()
    raw_obj = {"id": "resp_456"}
    action = Action(type="message", content="test", llm_response_raw=raw_obj)

    traj.add_step(
        step=0,
        action=action,
        reasoning_text="some thought",
        llm_response_raw=raw_obj,
    )

    assert traj.steps[0].llm_response_raw is raw_obj
    assert traj.steps[0].reasoning_text == "some thought"


# ═══════════════════════════════════════════════════════════════════════
# reasoning_text vs reasoning_raw semantic split
# ═══════════════════════════════════════════════════════════════════════


def test_llm_response_text_vs_raw_are_separate():
    """LLMResponse.reasoning_text and reasoning_raw are distinct fields."""
    resp = LLMResponse(
        content="output",
        reasoning_text="I thought carefully",
        reasoning_raw={"type": "thinking", "data": "opaque"},
    )
    assert resp.reasoning_text == "I thought carefully"
    assert resp.reasoning_raw == {"type": "thinking", "data": "opaque"}
    # They must be different objects, not aliased
    assert resp.reasoning_text != resp.reasoning_raw


def test_action_text_vs_raw_are_separate():
    """Action.reasoning_text and reasoning_raw are distinct fields."""
    action = Action(
        type="tool_call",
        reasoning_text="plain summary",
        reasoning_raw=[{"type": "thinking", "thinking": "details"}],
    )
    assert action.reasoning_text == "plain summary"
    assert isinstance(action.reasoning_raw, list)


def test_message_reasoning_raw_round_trip():
    """Message.reasoning_raw round-trips through to_full_dict / from_dict."""
    raw_payload = [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "text", "text": "output"},
    ]
    msg = Message(role="assistant", content="output", reasoning_raw=raw_payload)

    full = msg.to_full_dict()
    assert "reasoning_raw" in full
    assert full["reasoning_raw"] == raw_payload

    # to_dict must NOT include reasoning_raw
    standard = msg.to_dict()
    assert "reasoning_raw" not in standard

    # from_dict restores it
    restored = Message.from_dict(full)
    assert restored.reasoning_raw == raw_payload


def test_message_reasoning_raw_omitted_when_none():
    """to_full_dict omits reasoning_raw when it's None."""
    msg = Message(role="assistant", content="No reasoning")
    full = msg.to_full_dict()
    assert "reasoning_raw" not in full


def test_trajectory_to_messages_exports_reasoning_raw():
    """Trajectory.to_messages() exports both reasoning_text and reasoning_raw."""
    traj = Trajectory()
    raw_payload = [{"type": "thinking", "data": "opaque"}]
    action = Action(
        type="message", content="answer",
        reasoning_text="my reasoning",
        reasoning_raw=raw_payload,
    )
    traj.add_step(step=0, action=action, reasoning_text="my reasoning")

    msgs = traj.to_messages()
    assert msgs[0]["reasoning_text"] == "my reasoning"
    assert msgs[0]["reasoning_raw"] == raw_payload


def test_trajectory_to_training_format_includes_reasoning():
    """to_training_format() includes reasoning_text and reasoning_raw per step."""
    traj = Trajectory()
    traj.add_step(
        step=0,
        action=Action(
            type="message", content="answer",
            reasoning_text="I thought about this",
            reasoning_raw={"type": "thinking", "encrypted": "data"},
        ),
    )
    traj.add_step(
        step=1,
        action=Action(type="message", content="no reasoning here"),
    )

    result = traj.to_training_format()
    assert "reasoning" in result
    assert len(result["reasoning"]) == 1  # only step 0 has reasoning
    assert result["reasoning"][0]["step"] == 0
    assert result["reasoning"][0]["reasoning_text"] == "I thought about this"
    assert result["reasoning"][0]["reasoning_raw"]["encrypted"] == "data"


def test_trajectory_to_messages_exports_both_reasoning_fields():
    """Trajectory.to_messages() exports both reasoning_text and reasoning_raw."""
    traj = Trajectory()
    action = Action(
        type="message", content="answer",
        reasoning_text="my reasoning",
        reasoning_raw={"opaque": "data"},
    )
    traj.add_step(step=0, action=action, reasoning_text="my reasoning")

    msgs = traj.to_messages()
    assert msgs[0]["reasoning_text"] == "my reasoning"
    assert msgs[0]["reasoning_raw"] == {"opaque": "data"}


# ═══════════════════════════════════════════════════════════════════════
# create_async_client rejects unsupported backends
# ═══════════════════════════════════════════════════════════════════════


def test_ark_serializes_reasoning_content():
    """Ark backend preserves reasoning_raw as reasoning_content in messages."""
    from aweagent.core.llm.backends.ark import ArkBackend
    from aweagent.core.llm.config import LLMConfig

    config = LLMConfig(backend="ark", model="ep-xxx", api_key="test")
    backend = object.__new__(ArkBackend)
    backend.config = config

    messages = [
        Message(role="user", content="Hello"),
        Message(
            role="assistant", content="Answer",
            reasoning_raw="I thought about this carefully.",
        ),
    ]
    serialized = backend._serialize_messages(messages)

    assert serialized[1]["role"] == "assistant"
    assert serialized[1]["reasoning_content"] == "I thought about this carefully."
    # to_dict() should NOT have reasoning
    assert "reasoning_raw" not in serialized[0]


def test_create_async_client_rejects_anthropic():
    """create_async_client raises ValueError for 'anthropic' backend."""
    from aweagent.core.llm.client import create_async_client

    with pytest.raises(ValueError, match="anthropic"):
        create_async_client(backend="anthropic", api_key="test")


def test_create_async_client_rejects_openai_response():
    """create_async_client raises ValueError for 'openai_response' backend."""
    from aweagent.core.llm.client import create_async_client

    with pytest.raises(ValueError, match="openai_response"):
        create_async_client(backend="openai_response", api_key="test")


def test_create_async_client_accepts_openai():
    """create_async_client works for 'openai' backend."""
    from aweagent.core.llm.client import create_async_client

    client = create_async_client(backend="openai", api_key="test-key")
    assert client is not None
