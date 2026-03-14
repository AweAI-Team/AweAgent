"""Core types for the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # Raw reasoning payload for multi-turn round-trip back to provider.
    # Format depends on backend:
    #   - str: reasoning_content field value (Kimi, DeepSeek, GLM-5, Ark)
    #   - list[dict]: Anthropic content blocks (thinking/text/tool_use)
    #   - dict: OpenAI Response API (response_id + reasoning items)
    #   - None: no reasoning or backend doesn't need to preserve it
    reasoning_raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to standard OpenAI format (no reasoning — backward compatible)."""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    def to_full_dict(self) -> dict[str, Any]:
        """Serialize including reasoning_raw — for trajectory/training export."""
        d = self.to_dict()
        if self.reasoning_raw is not None:
            d["reasoning_raw"] = self.reasoning_raw
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        tool_calls = None
        if "tool_calls" in d and d["tool_calls"]:
            tool_calls = [ToolCall.from_dict(tc) for tc in d["tool_calls"]]
        return cls(
            role=d["role"],
            content=d.get("content"),
            tool_calls=tool_calls,
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            reasoning_raw=d.get("reasoning_raw"),
        )


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: str  # JSON string

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        func = d.get("function", d)
        return cls(
            id=d.get("id", ""),
            name=func["name"],
            arguments=func.get("arguments", "{}"),
        )


@dataclass
class TokenUsage:
    """Token usage statistics from an LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0  # Reasoning/thinking token count
    cached_tokens: int = 0  # Prompt cache hit tokens


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_text: str | None = None  # Plain-text reasoning for logs/training
    reasoning_raw: Any = None  # Raw provider payload for multi-turn round-trip
    usage: TokenUsage | None = None
    finish_reason: str | None = None  # "stop" | "length" | "tool_calls" | etc.
    raw: Any = None  # Raw response for debugging

    # RL-related fields (populated by SGLang backend when return_tokens/logprobs enabled)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    weight_version: str | None = None
    finish_status: str | None = None  # "stop" | "length"
