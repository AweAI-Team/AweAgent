"""Retry middleware for LLM calls."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine

from awe_agent.core.llm.config import RetryConfig
from awe_agent.core.llm.types import LLMResponse, Message

logger = logging.getLogger(__name__)

ChatFn = Callable[..., Coroutine[Any, Any, LLMResponse]]


def with_retry(config: RetryConfig) -> Callable[[ChatFn], ChatFn]:
    """Wrap an async chat function with retry logic."""

    def decorator(fn: ChatFn) -> ChatFn:
        async def wrapper(
            messages: list[Message],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            last_exc: Exception | None = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await fn(messages, tools, **kwargs)
                except Exception as e:
                    last_exc = e
                    exc_name = type(e).__name__
                    should_retry = any(r in exc_name for r in config.retry_on)
                    if not should_retry or attempt == config.max_attempts:
                        raise

                    delay = _compute_delay(config, attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt,
                        config.max_attempts,
                        exc_name,
                        delay,
                    )
                    await asyncio.sleep(delay)

            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def _compute_delay(config: RetryConfig, attempt: int) -> float:
    if config.backoff == "exponential":
        delay = config.base_delay * (2 ** (attempt - 1))
    elif config.backoff == "linear":
        delay = config.base_delay * attempt
    else:
        delay = config.base_delay
    # Add jitter (±25%)
    delay *= 0.75 + random.random() * 0.5
    return min(delay, config.max_delay)
