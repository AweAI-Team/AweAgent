"""Volcengine Ark LLM backend. Supports extended thinking mode.

Install: pip install -U 'volcengine-python-sdk[ark]'
"""

from __future__ import annotations

import logging
from typing import Any

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


class ArkBackend:
    """Backend for Volcengine Ark runtime. Supports extended thinking."""

    def __init__(self, config: LLMConfig) -> None:
        try:
            from volcenginesdkarkruntime import AsyncArk
        except ImportError:
            raise ImportError(
                "Ark backend requires volcengine SDK. "
                "Install with: pip install -U 'volcengine-python-sdk[ark]'"
            )
        self.config = config
        self._client = AsyncArk(
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
        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
        }

        # Merge config params with runtime overrides — pass everything through.
        # If the YAML config has invalid params, let the API error out directly.
        params.update({**self.config.params, **kwargs})

        stop = params.pop("stop", None) or self.config.stop
        if stop:
            params["stop"] = stop

        if tools:
            params["tools"] = tools

        # Extended thinking support (special handling: convert to Ark format)
        thinking_config = params.pop("thinking", None)
        if thinking_config or self.config.thinking:
            budget = (
                thinking_config.get("budget_tokens", self.config.thinking_budget)
                if isinstance(thinking_config, dict)
                else self.config.thinking_budget
            ) or 10000
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}

        response = await self._client.chat.completions.create(**params)
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                ))

        # Extract thinking content if present
        thinking = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            thinking = msg.reasoning_content

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            thinking=thinking,
            usage=usage,
            finish_reason=getattr(choice, "finish_reason", None),
            raw=response,
        )
