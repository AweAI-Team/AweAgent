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

    def to_dict(self) -> dict[str, Any]:
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


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None  # Extended thinking content
    usage: TokenUsage | None = None
    raw: Any = None  # Raw response for debugging

    # RL-related fields (populated when return_tokens/return_logprobs enabled)
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
