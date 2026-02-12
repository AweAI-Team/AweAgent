"""Trace middleware for LLM calls — structured logging of requests/responses."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Coroutine

from awe_agent.core.llm.types import LLMResponse, Message

logger = logging.getLogger(__name__)

ChatFn = Callable[..., Coroutine[Any, Any, LLMResponse]]


def with_trace(fn: ChatFn) -> ChatFn:
    """Log LLM call timing and token usage."""

    async def wrapper(
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.monotonic()
        model = kwargs.get("model", "unknown")
        logger.debug("LLM call start: model=%s, messages=%d", model, len(messages))

        response = await fn(messages, tools, **kwargs)

        elapsed = time.monotonic() - start
        usage_str = ""
        if response.usage:
            usage_str = (
                f", tokens={response.usage.prompt_tokens}+{response.usage.completion_tokens}"
            )
        logger.info("LLM call done: model=%s, %.2fs%s", model, elapsed, usage_str)
        return response

    return wrapper
