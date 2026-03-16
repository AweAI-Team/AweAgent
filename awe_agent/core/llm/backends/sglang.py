"""SGLang LLM backend for RL training integration.

This backend communicates with an SGLang inference server via two endpoints:

- ``/generate`` — raw token-level interface used during RL training.  Accepts
  ``input_ids`` and returns ``output_token_logprobs`` (list of
  ``(logprob, token_id)`` tuples) plus ``weight_version`` for staleness
  detection.
- ``/v1/chat/completions`` — OpenAI-compatible interface used during inference.

The backend automatically selects the endpoint based on whether ``input_ids``
are provided in the call kwargs.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage

logger = logging.getLogger(__name__)

# Keys forwarded from the merged params dict into SGLang sampling_params.
_SAMPLING_KEYS = ("temperature", "max_new_tokens", "max_tokens", "top_p", "top_k")


class SGLangBackend:
    """Backend for SGLang inference engine.

    In **RL training mode** (``input_ids`` provided), uses the ``/generate``
    endpoint and returns token-level data (token IDs, log-probabilities,
    weight version).  In **inference mode**, falls back to the standard
    ``/v1/chat/completions`` endpoint.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._base_url = (config.base_url or "http://localhost:30000").rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
        return self._session

    # ── Public interface ──────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        merged = {**self.config.params, **kwargs}

        # RL continuation mode: caller provides pre-built input_ids.
        input_ids = merged.pop("input_ids", None)
        if input_ids is not None:
            return await self._generate_from_ids(input_ids, merged)

        # Standard inference mode.
        return await self._chat_completions(messages, tools, merged)

    # ── /generate endpoint (RL training) ──────────────────────────────

    async def _generate_from_ids(
        self, input_ids: list[int], params: dict[str, Any],
    ) -> LLMResponse:
        """Call SGLang ``/generate`` with raw token IDs.

        Handles the **max_new_tokens overflow** case: when the accumulated
        input sequence is so long that the remaining budget is negative,
        we issue a minimal request solely to obtain the current
        ``weight_version`` and return an empty ``"length"`` response.
        """
        sampling_params = self._extract_sampling_params(params)

        # ── Overflow guard ─────────────────────────────────────────────
        # The model cannot generate if the remaining budget is exhausted.
        max_budget = sampling_params.get("max_new_tokens", 32768)
        remaining = max_budget - len(input_ids) - 128
        if remaining <= 0:
            logger.info(
                "Token budget exhausted (input=%d, budget=%d), fetching weight_version only",
                len(input_ids), max_budget,
            )
            weight_version = await self._fetch_weight_version(sampling_params)
            return LLMResponse(
                content="",
                finish_status="length",
                weight_version=weight_version,
            )

        sampling_params["max_new_tokens"] = remaining

        # ── Normal generation ─────────────────────────────────────────
        session = await self._get_session()
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        async with session.post(f"{self._base_url}/generate", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        meta = data.get("meta_info", {})

        # Parse output_token_logprobs: list of (logprob, token_id) tuples.
        token_logprobs = meta.get("output_token_logprobs", [])
        completion_ids = [item[1] for item in token_logprobs]
        logprobs = [item[0] for item in token_logprobs]

        # Finish reason: meta_info.finish_reason is {"type": "stop"|"length"}.
        finish_reason_obj = meta.get("finish_reason", {})
        finish_status = (
            finish_reason_obj.get("type", "stop")
            if isinstance(finish_reason_obj, dict)
            else str(finish_reason_obj)
        )

        return LLMResponse(
            content=data.get("text", ""),
            completion_token_ids=completion_ids,
            logprobs=logprobs,
            weight_version=meta.get("weight_version"),
            finish_status=finish_status,
            usage=TokenUsage(
                prompt_tokens=meta.get("prompt_tokens", 0),
                completion_tokens=meta.get("completion_tokens", 0),
                total_tokens=(
                    meta.get("prompt_tokens", 0)
                    + meta.get("completion_tokens", 0)
                ),
            ),
            finish_reason=finish_status,
            raw=data,
        )

    # ── /v1/chat/completions endpoint (inference) ─────────────────────

    async def _chat_completions(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> LLMResponse:
        """OpenAI-compatible chat completions endpoint for inference."""
        session = await self._get_session()

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

        async with session.post(
            f"{self._base_url}/v1/chat/completions", json=payload,
        ) as resp:
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

    # ── Helpers ───────────────────────────────────────────────────────

    def _extract_sampling_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Extract SGLang sampling_params from the merged param dict.

        Pops recognised keys and normalises ``max_tokens`` →
        ``max_new_tokens`` (SGLang naming convention).
        """
        sampling_params: dict[str, Any] = {}
        for key in _SAMPLING_KEYS:
            if key in params:
                val = params.pop(key)
                if key == "max_tokens":
                    sampling_params["max_new_tokens"] = val
                else:
                    sampling_params[key] = val

        stop = params.pop("stop", None) or self.config.stop
        if stop:
            sampling_params["stop"] = stop

        # Forward any extra sampling params provided via config.
        extra_sampling = params.pop("sampling_params", None)
        if isinstance(extra_sampling, dict):
            sampling_params.update(extra_sampling)

        return sampling_params

    async def _fetch_weight_version(
        self, sampling_params: dict[str, Any],
    ) -> str | None:
        """Obtain the current ``weight_version`` via a minimal /generate call.

        When the token budget is exhausted, we still need the weight version
        to mark the sample correctly.  This sends a short dummy request
        (``max_new_tokens=12``) purely to read ``meta_info.weight_version``.
        """
        session = await self._get_session()
        # Use a trivially short prompt — the content doesn't matter.
        dummy_ids = [1]
        mini_params = {
            k: v for k, v in sampling_params.items()
            if k not in ("max_new_tokens",)
        }
        mini_params["max_new_tokens"] = 12

        payload = {
            "input_ids": dummy_ids,
            "sampling_params": mini_params,
            "return_logprob": True,
        }
        try:
            async with session.post(
                f"{self._base_url}/generate", json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return data.get("meta_info", {}).get("weight_version")
        except Exception:
            logger.warning("Failed to fetch weight_version via dummy request", exc_info=True)
            return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
