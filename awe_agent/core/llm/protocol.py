"""LLM Backend Protocol — the interface all backends must satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from awe_agent.core.llm.types import LLMResponse, Message


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends.

    Each backend (OpenAI, Azure, Ark, SGLang, etc.) implements this single method.
    Backends are kept simple — retry, caching, tracing are handled by middleware.
    """

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat request, optionally with tool schemas.

        Args:
            messages: Conversation history.
            tools: OpenAI-format tool/function schemas.
            **kwargs: Merged generation params (temperature, max_tokens, stop, etc.)

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        ...
