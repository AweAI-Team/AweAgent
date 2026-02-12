"""LLM configuration models. All LLM behavior is controlled via config, no code changes needed."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RetryConfig(BaseModel):
    """Retry behavior for LLM calls."""

    max_attempts: int = 5
    backoff: Literal["exponential", "linear", "fixed"] = "exponential"
    base_delay: float = 1.0
    max_delay: float = 60.0
    retry_on: list[str] = Field(
        default_factory=lambda: ["RateLimitError", "APIConnectionError", "Timeout"]
    )


class CacheConfig(BaseModel):
    """Response caching for LLM calls."""

    enabled: bool = False
    ttl: int = 3600


class LLMConfig(BaseModel):
    """Complete LLM configuration.

    All behavior is driven by this config. Switch backends, enable thinking,
    set stop strings — all without touching code.

    Example YAML:
        backend: openai
        model: gpt-4o
        base_url: https://api.openai.com/v1
        api_key: ${OPENAI_API_KEY}
        params:
          temperature: 0.0
          max_tokens: 4096
        thinking: true
        thinking_budget: 10000
        stop: ["<DONE>"]
    """

    backend: str = "openai"
    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o"

    # Generation parameters — all config-driven
    params: dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0.0,
        "max_tokens": 4096,
    })

    # Advanced features
    thinking: bool = False
    thinking_budget: int | None = None
    stop: list[str] | None = None
    response_format: dict[str, Any] | None = None

    # Middleware
    retry: RetryConfig = Field(default_factory=RetryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    timeout: float = 120.0

    # RL mode: whether to return token ids and logprobs
    return_tokens: bool = False
    return_logprobs: bool = False

    # Extra backend-specific args (passed directly to backend constructor)
    extra: dict[str, Any] = Field(default_factory=dict)
