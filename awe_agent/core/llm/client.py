"""LLMClient — the unified entry point for all LLM interactions.

Handles backend dispatch, middleware chain (retry, trace), and config merging.
Users interact with this class, not backends directly.
"""

from __future__ import annotations

import logging
from typing import Any

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.middleware.retry import with_retry
from awe_agent.core.llm.middleware.trace import with_trace
from awe_agent.core.llm.protocol import LLMBackend
from awe_agent.core.llm.types import LLMResponse, Message
from awe_agent.plugins.registry import Registry

logger = logging.getLogger(__name__)

# Global registry for LLM backends
llm_registry: Registry[type] = Registry("awe_agent.llm_backend")


def create_async_client(
    backend: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a raw async LLM client (OpenAI-compatible).

    A lightweight factory that creates just the underlying async client
    without middleware or response parsing. Useful for tools that need
    direct access to the ``chat.completions.create`` API.

    Args:
        backend: One of ``"openai"``, ``"azure"``, ``"ark"``.
        api_key: API key for the service.
        base_url: Base URL / endpoint for the service.
        **kwargs: Backend-specific arguments:
            - ``azure_endpoint``: Azure endpoint (falls back to *base_url*).
            - ``api_version``: Azure API version (default ``"2024-02-01"``).
            - ``timeout``: Request timeout in seconds.

    Returns:
        An ``AsyncOpenAI``, ``AsyncAzureOpenAI``, or ``AsyncArk`` client.
    """
    timeout = kwargs.get("timeout")

    if backend == "azure":
        from openai import AsyncAzureOpenAI
        return AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=kwargs.get("azure_endpoint") or base_url or "",
            api_version=kwargs.get("api_version", "2024-02-01"),
            **({"timeout": timeout} if timeout else {}),
        )

    if backend == "ark":
        try:
            from volcenginesdkarkruntime import AsyncArk  # type: ignore[import-untyped]
            return AsyncArk(
                api_key=api_key,
                base_url=base_url,
                **({"timeout": timeout} if timeout else {}),
            )
        except ImportError:
            logger.warning(
                "volcenginesdkarkruntime not installed, falling back to openai"
            )

    if backend not in ("openai", "ark"):
        # ark falls through here when SDK import fails; other backends
        # (anthropic, openai_response, etc.) have fundamentally different
        # client interfaces and cannot be used with chat.completions.create.
        # Raise explicitly rather than returning a silently-wrong client.
        raise ValueError(
            f"create_async_client does not support backend={backend!r}. "
            f"Only 'openai', 'azure', and 'ark' are supported. "
            f"Use LLMClient for full backend support (anthropic, openai_response, etc.)."
        )

    # Default: openai (also fallback for failed ark import)
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        **({"timeout": timeout} if timeout else {}),
    )


class LLMClient:
    """Unified LLM client.

    Creates the appropriate backend based on config, applies middleware,
    and provides a single `chat()` method for all LLM interactions.

    Example:
        config = LLMConfig(backend="openai", model="gpt-4o")
        client = LLMClient(config)
        response = await client.chat([Message(role="user", content="Hello")])
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._backend = self._create_backend(config)
        logger.info("LLMClient: backend=%s, model=%s", config.backend, config.model)

    def _create_backend(self, config: LLMConfig) -> LLMBackend:
        backend_cls = llm_registry.get(config.backend)
        return backend_cls(config)

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> LLMResponse:
        """Send a chat request through the middleware chain.

        Args:
            messages: Conversation history.
            tools: OpenAI-format tool schemas.
            **overrides: Runtime overrides for generation params.
        """
        # Build the call chain: backend.chat → trace → retry
        fn = self._backend.chat
        fn = with_trace(fn)
        fn = with_retry(self.config.retry)(fn)

        # Merge: config.params < overrides
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            **self.config.params,
            **overrides,
        }

        # Config-level stop strings (overridable at call time)
        if "stop" not in overrides and self.config.stop:
            kwargs["stop"] = self.config.stop

        # Thinking mode — pass config to backend for profile-aware resolution
        if self.config.thinking and "thinking" not in kwargs:
            thinking_dict: dict[str, Any] = {}
            if self.config.thinking_budget is not None:
                thinking_dict["budget_tokens"] = self.config.thinking_budget
            kwargs["thinking"] = thinking_dict

        return await fn(messages, tools, **kwargs)

    async def close(self) -> None:
        """Clean up backend resources."""
        if hasattr(self._backend, "close"):
            await self._backend.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
