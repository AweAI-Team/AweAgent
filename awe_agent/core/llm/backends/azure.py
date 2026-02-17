"""Azure OpenAI LLM backend."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncAzureOpenAI

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


class AzureOpenAIBackend:
    """Backend for Azure OpenAI Service."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.base_url or "",
            api_version=config.extra.get("api_version", "2024-06-01"),
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
            usage=usage,
            raw=response,
        )
