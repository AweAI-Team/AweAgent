"""Anthropic Messages API backend.

Supports Claude models with native thinking blocks, tool use, and
strict user/assistant message alternation. Also works with any
Anthropic-compatible provider (e.g. MiniMax) — the content block
parsing and round-trip logic is provider-agnostic.

Install: pip install anthropic
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)

# Models that support adaptive thinking (official Anthropic docs, 2026-03-14)
_ADAPTIVE_THINKING_PREFIXES = ("claude-opus-4-6", "claude-sonnet-4-6")


def _normalize_stop_reason(stop_reason: str | None) -> str | None:
    """Map Anthropic stop_reason to framework-standard finish_reason."""
    if stop_reason is None:
        return None
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }.get(stop_reason, stop_reason)


class AnthropicBackend:
    """Backend for the Anthropic Messages API."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "Anthropic backend requires the anthropic SDK. "
                "Install with: pip install anthropic"
            )
        self.config = config
        self._client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_content, anthropic_msgs = self._serialize_messages(messages)

        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": anthropic_msgs,
            "max_tokens": kwargs.pop(
                "max_tokens",
                self.config.params.get("max_tokens", 4096),
            ),
        }

        if system_content:
            params["system"] = system_content

        # Merge remaining config params (temperature, top_p, etc.)
        for k, v in self.config.params.items():
            if k not in ("max_tokens",) and k not in params:
                params[k] = v
        params.update(kwargs)

        # Thinking/extended thinking support
        thinking_config = params.pop("thinking", None)
        if thinking_config or self.config.thinking:
            params["thinking"] = self._resolve_thinking(thinking_config)
            # Anthropic requires unsetting temperature when thinking is enabled
            params.pop("temperature", None)

        # Tool schemas
        if tools:
            params["tools"] = self._convert_tools(tools)

        # Stop sequences
        stop = params.pop("stop", None) or self.config.stop
        if stop:
            params["stop_sequences"] = stop

        # Remove keys not accepted by Anthropic API
        params.pop("response_format", None)

        response = await self._client.messages.create(**params)
        return self._parse_response(response)

    def _resolve_thinking(self, thinking_config: Any) -> dict[str, Any]:
        """Build the ``thinking`` API parameter from explicit config.

        This method does NOT provide silent defaults.  If ``thinking: true``
        is set but neither ``thinking_type`` nor ``thinking_budget`` is
        configured, it raises ``ValueError``.  Presets are responsible for
        providing complete, runnable defaults.

        Rules:
            1. ``thinking_type: adaptive`` → ``{"type": "adaptive"}``.
               Raises ``ValueError`` if the model is not Claude 4.6.
            2. ``thinking_budget: N`` → ``{"type": "enabled", "budget_tokens": N}``.
            3. Neither configured → ``ValueError`` (no silent fallback).
        """
        # Runtime override takes precedence over config
        thinking_type = (
            thinking_config.get("type", self.config.thinking_type)
            if isinstance(thinking_config, dict)
            else self.config.thinking_type
        )

        if thinking_type == "adaptive":
            model = self.config.model.lower()
            if not any(model.startswith(p) for p in _ADAPTIVE_THINKING_PREFIXES):
                raise ValueError(
                    f"thinking_type='adaptive' is only supported by Claude 4.6 "
                    f"models ({', '.join(_ADAPTIVE_THINKING_PREFIXES)}*), "
                    f"but got model={self.config.model!r}. "
                    f"Use thinking_budget instead, or switch to a supported model."
                )
            return {"type": "adaptive"}

        # Manual mode: config budget or runtime budget
        budget = (
            thinking_config.get("budget_tokens", self.config.thinking_budget)
            if isinstance(thinking_config, dict)
            else self.config.thinking_budget
        )
        if budget is not None:
            if budget <= 0:
                raise ValueError(
                    f"thinking_budget must be a positive integer, got {budget}. "
                    f"For adaptive thinking, set thinking_type='adaptive' instead."
                )
            return {"type": "enabled", "budget_tokens": budget}

        # Neither thinking_type nor thinking_budget — refuse to guess
        raise ValueError(
            "thinking=True requires either thinking_type or thinking_budget. "
            "Set thinking_type='adaptive' (Claude 4.6 only) or "
            "thinking_budget=<N> in your LLM config."
        )

    def _serialize_messages(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert framework Messages to Anthropic format.

        Returns:
            (system_text, anthropic_messages)
        """
        system_parts: list[str] = []
        anthropic_msgs: list[dict[str, Any]] = []

        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                continue

            if m.role == "assistant":
                content_blocks: list[dict[str, Any]] = []

                # Check if reasoning_raw carries full content blocks
                # (lossless round-trip from _parse_response).  Detected by the
                # presence of non-thinking block types (text / tool_use).
                if m.reasoning_raw and isinstance(m.reasoning_raw, list):
                    has_full_blocks = any(
                        isinstance(b, dict) and b.get("type") in ("text", "tool_use")
                        for b in m.reasoning_raw
                    )
                    if has_full_blocks:
                        # Replay ALL blocks in original order — lossless
                        content_blocks = list(m.reasoning_raw)
                    else:
                        # Legacy: reasoning_raw only has thinking blocks
                        for block in m.reasoning_raw:
                            content_blocks.append(block)
                        if m.content:
                            content_blocks.append(
                                {"type": "text", "text": m.content}
                            )
                        if m.tool_calls:
                            for tc in m.tool_calls:
                                content_blocks.append({
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.name,
                                    "input": json.loads(tc.arguments)
                                    if isinstance(tc.arguments, str)
                                    else tc.arguments,
                                })
                else:
                    # No reasoning_raw — reconstruct from fields
                    if m.content:
                        content_blocks.append(
                            {"type": "text", "text": m.content}
                        )
                    if m.tool_calls:
                        for tc in m.tool_calls:
                            content_blocks.append({
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": json.loads(tc.arguments)
                                if isinstance(tc.arguments, str)
                                else tc.arguments,
                            })

                if content_blocks:
                    anthropic_msgs.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })

            elif m.role == "tool":
                # Anthropic: tool_result is a content block inside a user message.
                # Merge consecutive tool results into the same user message.
                tool_result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content or "",
                }
                if (
                    anthropic_msgs
                    and anthropic_msgs[-1]["role"] == "user"
                    and isinstance(anthropic_msgs[-1]["content"], list)
                ):
                    anthropic_msgs[-1]["content"].append(tool_result_block)
                else:
                    anthropic_msgs.append({
                        "role": "user",
                        "content": [tool_result_block],
                    })

            else:  # user
                anthropic_msgs.append({
                    "role": m.role,
                    "content": m.content or "",
                })

        system = "\n\n".join(system_parts) if system_parts else ""
        return system, anthropic_msgs

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Anthropic response, extracting content, tool_use, and thinking blocks.

        Stores ALL content blocks in original order in reasoning_raw for
        lossless round-trip serialization.  The ``reasoning_text`` and
        ``content`` fields carry plain-text summaries for logging / training.
        """
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        # Preserve ALL blocks in original order for lossless round-trip
        all_content_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
                all_content_blocks.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input)
                    if not isinstance(block.input, str)
                    else block.input,
                ))
                all_content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "thinking":
                thinking_parts.append(block.thinking)
                thinking_block: dict[str, Any] = {
                    "type": "thinking",
                    "thinking": block.thinking,
                }
                # Preserve signature for round-trip (Anthropic requires it)
                if hasattr(block, "signature") and block.signature:
                    thinking_block["signature"] = block.signature
                all_content_blocks.append(thinking_block)
            elif block.type == "redacted_thinking":
                # Redacted thinking must be preserved unchanged for round-trip
                all_content_blocks.append(
                    {"type": "redacted_thinking", "data": block.data}
                )

        content_text = "\n".join(content_parts) if content_parts else None
        thinking_text = "\n".join(thinking_parts) if thinking_parts else None

        # Parse usage
        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0),
                completion_tokens=getattr(response.usage, "output_tokens", 0),
                total_tokens=(
                    getattr(response.usage, "input_tokens", 0)
                    + getattr(response.usage, "output_tokens", 0)
                ),
                cached_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0
                ) or 0,
            )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            reasoning_text=thinking_text,
            reasoning_raw=all_content_blocks or None,
            usage=usage,
            finish_reason=_normalize_stop_reason(getattr(response, "stop_reason", None)),
            raw=response,
        )

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool schema to Anthropic tool schema."""
        anthropic_tools: list[dict[str, Any]] = []
        for t in tools:
            func = t.get("function", t)
            anthropic_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            })
        return anthropic_tools

    async def close(self) -> None:
        """Clean up client resources."""
        if hasattr(self._client, "close"):
            await self._client.close()
