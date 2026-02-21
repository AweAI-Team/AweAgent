"""SGLang LLM backend for RL training integration.

This backend sends requests to an SGLang inference server.
It returns token_ids and logprobs for RL training.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage

logger = logging.getLogger(__name__)


class SGLangBackend:
    """Backend for SGLang inference engine. Returns token-level data for RL."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._base_url = config.base_url or "http://localhost:30000"
        # SGLang has two modes: /generate (raw) and /v1/chat/completions (OpenAI-compat)
        # For RL training we use /generate for token-level control
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._session

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        merged = {**self.config.params, **kwargs}

        # Check if token_ids are provided directly (RL multi-turn continuation)
        input_ids = merged.pop("input_ids", None)

        if input_ids is not None:
            return await self._generate_from_ids(input_ids, merged)

        # Otherwise use OpenAI-compatible chat endpoint
        return await self._chat_completions(messages, tools, merged)

    async def _generate_from_ids(
        self, input_ids: list[int], params: dict[str, Any]
    ) -> LLMResponse:
        """Direct /generate endpoint with token IDs for RL training."""
        session = await self._get_session()
        url = f"{self._base_url}/generate"

        sampling_params: dict[str, Any] = {}
        for key in ("temperature", "max_new_tokens", "max_tokens", "top_p", "top_k"):
            if key in params:
                val = params.pop(key)
                # SGLang uses max_new_tokens
                if key == "max_tokens":
                    sampling_params["max_new_tokens"] = val
                else:
                    sampling_params[key] = val

        stop = params.pop("stop", None) or self.config.stop
        if stop:
            sampling_params["stop"] = stop

        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": self.config.return_logprobs,
        }

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return LLMResponse(
            content=data.get("text", ""),
            tool_calls=[],
            completion_token_ids=data.get("meta_info", {}).get("output_ids"),
            logprobs=data.get("meta_info", {}).get("output_logprobs"),
            usage=TokenUsage(
                prompt_tokens=data.get("meta_info", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("meta_info", {}).get("completion_tokens", 0),
                total_tokens=data.get("meta_info", {}).get("prompt_tokens", 0)
                + data.get("meta_info", {}).get("completion_tokens", 0),
            ),
            raw=data,
        )

    async def _chat_completions(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> LLMResponse:
        """OpenAI-compatible /v1/chat/completions endpoint."""
        session = await self._get_session()
        url = f"{self._base_url}/v1/chat/completions"

        payload: dict[str, Any] = {
            "model": params.pop("model", self.config.model),
            "messages": [m.to_dict() for m in messages],
        }

        for key in ("temperature", "max_tokens", "top_p"):
            if key in params:
                payload[key] = params.pop(key)

        stop = params.pop("stop", None) or self.config.stop
        if stop:
            payload["stop"] = stop

        if tools:
            payload["tools"] = tools

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        choice = data["choices"][0]
        msg = choice["message"]
        from awe_agent.core.llm.types import ToolCall

        tool_calls = []
        if msg.get("tool_calls"):
            tool_calls = [ToolCall.from_dict(tc) for tc in msg["tool_calls"]]

        usage_data = data.get("usage", {})
        return LLMResponse(
            content=msg.get("content"),
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
