"""OpenAI-compatible LLM backend.

Works with OpenAI, vLLM, and any OpenAI-compatible API.
Supports reasoning models: Kimi, DeepSeek, GLM-5, MiniMax, Qwen, etc.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from openai import AsyncOpenAI

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)

# Keys that should be sent via extra_body rather than top-level params
_EXTRA_BODY_KEYS = {"reasoning_split", "thinking", "clear_thinking", "enable_thinking"}


def _extract_think_tags(content: str) -> tuple[str, str]:
    """Extract <think>...</think> blocks from content.

    Returns:
        (thinking_text, clean_content) where clean_content has think blocks removed.
    """
    think_parts: list[str] = []
    clean = content
    for match in re.finditer(r"<think>(.*?)</think>", content, re.DOTALL):
        think_parts.append(match.group(1).strip())
    clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return "\n".join(think_parts), clean


class OpenAIBackend:
    """Backend for OpenAI and OpenAI-compatible APIs.

    Handles reasoning content in multiple formats:
    - reasoning_content field (Kimi, DeepSeek, GLM-5, Ark)
    - reasoning_details field (MiniMax with reasoning_split=True)
    - <think> tags (MiniMax default, Qwen)
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Don't pass extra_body keys to the client constructor
        client_extra = {
            k: v for k, v in config.extra.items() if k not in _EXTRA_BODY_KEYS
        }
        self._client = AsyncOpenAI(
            api_key=config.api_key or "dummy",
            base_url=config.base_url,
            timeout=config.timeout,
            **client_extra,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        request_params = self._build_params(messages, tools, **kwargs)
        response = await self._client.chat.completions.create(**request_params)
        return self._parse_response(response)

    def _should_preserve_reasoning(self) -> bool:
        """Determine whether to include reasoning_content in serialized messages."""
        preserve = self.config.reasoning.preserve
        if preserve is not None:
            return preserve
        # Auto mode: default to not preserving (safest — DeepSeek errors if sent)
        return False

    def _get_reasoning_field_name(self) -> str:
        """Return the API field name for reasoning based on config format."""
        fmt = self.config.reasoning.format
        if fmt == "reasoning_details":
            return "reasoning_details"
        # "auto", "reasoning_content", "think_tags" → all use reasoning_content
        # (think_tags are embedded in content, not a separate field, and only
        #  sent back as reasoning_content by models that need it)
        return "reasoning_content"

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Serialize Message list for the API, handling reasoning preservation."""
        result = []
        field_name = self._get_reasoning_field_name()
        for m in messages:
            d = m.to_dict()
            if m.role == "assistant" and m.reasoning_raw is not None:
                if self._should_preserve_reasoning():
                    d[field_name] = m.reasoning_raw
                # else: strip reasoning (DeepSeek etc.)
            result.append(d)
        return result

    def _build_params(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": self._serialize_messages(messages),
        }

        # Merge config params with runtime overrides — pass everything through.
        params.update({**self.config.params, **kwargs})

        stop = params.pop("stop", None) or self.config.stop
        if stop:
            params["stop"] = stop

        response_format = params.pop("response_format", None) or self.config.response_format
        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools

        # Move special keys from config.extra and params into extra_body
        extra_body: dict[str, Any] = {}
        for key in list(_EXTRA_BODY_KEYS):
            if key in params:
                extra_body[key] = params.pop(key)
            elif key in self.config.extra:
                extra_body[key] = self.config.extra[key]

        # GLM-5: clear_thinking must be nested inside the thinking dict,
        # not a sibling key.  Merge it in when both are present.
        if (
            "clear_thinking" in extra_body
            and "thinking" in extra_body
            and isinstance(extra_body["thinking"], dict)
        ):
            extra_body["thinking"]["clear_thinking"] = extra_body.pop(
                "clear_thinking"
            )

        if extra_body:
            params["extra_body"] = extra_body

        return params

    def _parse_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        # Extract reasoning content (multiple format support)
        thinking = None
        reasoning_raw = None
        content = msg.content

        # 1. reasoning_content field (Kimi, DeepSeek, GLM-5, Ark)
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            thinking = msg.reasoning_content
            reasoning_raw = msg.reasoning_content  # str

        # 2. reasoning_details field (MiniMax with reasoning_split=True)
        elif hasattr(msg, "reasoning_details") and msg.reasoning_details:
            reasoning_raw = msg.reasoning_details  # list[dict]
            thinking = "\n".join(
                item.get("text", "")
                for item in msg.reasoning_details
                if isinstance(item, dict) and item.get("type") == "reasoning.text"
            )
            if not thinking:
                # Fallback: concatenate all text entries
                thinking = "\n".join(
                    item.get("text", "")
                    for item in msg.reasoning_details
                    if isinstance(item, dict) and item.get("text")
                )

        # 3. <think> tags (MiniMax default, Qwen)
        elif content and "<think>" in content:
            thinking, content = _extract_think_tags(content)
            reasoning_raw = thinking

        # Parse usage with reasoning/cached tokens
        usage = None
        if response.usage:
            reasoning_tokens = 0
            cached_tokens = 0
            if hasattr(response.usage, "completion_tokens_details"):
                details = response.usage.completion_tokens_details
                if details and hasattr(details, "reasoning_tokens"):
                    reasoning_tokens = details.reasoning_tokens or 0
            if hasattr(response.usage, "prompt_tokens_details"):
                details = response.usage.prompt_tokens_details
                if details and hasattr(details, "cached_tokens"):
                    cached_tokens = details.cached_tokens or 0
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                reasoning_tokens=reasoning_tokens,
                cached_tokens=cached_tokens,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_text=thinking,
            reasoning_raw=reasoning_raw,
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )
