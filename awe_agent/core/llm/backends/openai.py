"""OpenAI-compatible LLM backend. Works with OpenAI, vLLM, and any OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """Backend for OpenAI and OpenAI-compatible APIs."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "dummy",
            base_url=config.base_url,
            timeout=config.timeout,
            **config.extra,
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

    def _build_params(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

        response_format = params.pop("response_format", None) or self.config.response_format
        if response_format:
            params["response_format"] = response_format

        if tools:
            params["tools"] = tools

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

        # Parse usage
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
