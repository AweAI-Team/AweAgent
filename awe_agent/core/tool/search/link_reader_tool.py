"""LinkReaderTool — fetch raw content from URLs with pluggable reader backends."""

from __future__ import annotations

import logging
from typing import Any, Callable

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.search.constraints import SearchConstraints

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_TOKENS = 100000


class LinkReaderTool(Tool):
    """Fetch raw content from URLs (web pages or PDFs).

    Checks URLs against :class:`SearchConstraints` before fetching.

    Supports pluggable reader backends discovered through entry-points
    (``awe_agent.reader_backend`` group) or injected directly.

    Backend resolution order:
        1. Explicit ``backend`` argument (instance or registry name).
        2. Explicit ``reader_fn`` callable (legacy / testing).
        3. Auto-discover from ``READER_BACKEND`` env var or registry.

    Args:
        constraints: Optional constraints for URL blocking.
        max_content_tokens: Maximum tokens for content truncation.
        max_attempts: Number of retry attempts for fetch calls.
        backend: A reader backend instance (must have an async ``read_link`` method),
            or a registered backend name (e.g. ``"jina"``).
            Auto-discovered if not provided.
        reader_fn: Async callable for fetching URL content (legacy / testing).
            Signature: ``async (url) -> str``.
    """

    def __init__(
        self,
        constraints: SearchConstraints | None = None,
        max_content_tokens: int = _DEFAULT_MAX_CONTENT_TOKENS,
        max_attempts: int = 3,
        backend: str | Any | None = None,
        reader_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._constraints = constraints or SearchConstraints()
        self._max_content_tokens = max_content_tokens
        self._max_attempts = max_attempts

        # Resolve backend
        if isinstance(backend, str):
            from awe_agent.core.tool.search.backends.reader import reader_backend_registry

            backend_cls = reader_backend_registry.get(backend)
            self._backend: Any | None = backend_cls()
        elif backend is not None:
            self._backend = backend
        else:
            self._backend = None  # lazy-discovered on first use

        # Legacy: direct callable injection (takes priority when set)
        self._reader_fn: Callable[..., Any] | None = reader_fn
        # Lazy-loaded tiktoken encoding (avoid re-creating per call)
        self._tiktoken_enc: Any = None

    @property
    def name(self) -> str:
        return "link_reader"

    @property
    def description(self) -> str:
        return (
            "Fetch raw content from a URL. Returns the full text content of "
            "a web page or PDF. Use 'link_summary' instead if you need a "
            "concise summary of the content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
            },
            "required": ["url"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        url = params.get("url", "").strip()
        if not url:
            return "Error: empty URL."

        # Check URL against constraints
        if self._constraints.is_url_blocked(url):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked by security constraints. "
                "This URL may point to the target repository and accessing it is "
                "not allowed during evaluation."
            )

        return await self._fetch(url)

    # ── Internal ─────────────────────────────────────────────────────

    async def _fetch(self, url: str) -> str:
        """Fetch URL content via reader_fn or backend."""
        # Priority 1: explicit reader_fn (legacy / testing)
        if self._reader_fn is not None:
            return await self._call_reader_fn(url)

        # Priority 2: backend (explicit or auto-discovered)
        backend = self._ensure_backend()
        if backend is None:
            return (
                "Error: no reader backend configured. Set READER_BACKEND env var "
                "or install a reader backend plugin (e.g. pip install awe-agent[search])."
            )

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                content = await backend.read_link(url)
                if not content:
                    return f"No content returned from {url}"
                return self._truncate_content(content, self._max_content_tokens)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "read_link attempt %d/%d failed for %r: %s",
                    attempt + 1, self._max_attempts, url, exc,
                )

        return f"Error: failed to fetch {url}: {last_error}"

    async def _call_reader_fn(self, url: str) -> str:
        """Call a raw reader_fn callable (legacy path)."""
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                result = await self._reader_fn(url=url)
                content = result if isinstance(result, str) else str(result)
                if not content:
                    return f"No content returned from {url}"
                return self._truncate_content(content, self._max_content_tokens)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "reader_fn attempt %d/%d failed for %r: %s",
                    attempt + 1, self._max_attempts, url, exc,
                )
        return f"Error: failed to fetch {url}: {last_error}"

    def _ensure_backend(self) -> Any:
        """Lazy-discover a reader backend if not yet resolved."""
        if self._backend is not None:
            return self._backend

        from awe_agent.core.tool.search.backends.reader import get_reader_backend

        self._backend = get_reader_backend()
        if self._backend is None:
            logger.warning(
                "No reader backend available. Set READER_BACKEND env var or "
                "install a reader backend plugin."
            )
        return self._backend

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Token-aware truncation using tiktoken (with char-based fallback)."""
        try:
            if self._tiktoken_enc is None:
                import tiktoken

                self._tiktoken_enc = tiktoken.get_encoding("o200k_base")
            tokens = self._tiktoken_enc.encode(content)
            if len(tokens) <= max_tokens:
                return content
            truncated = self._tiktoken_enc.decode(tokens[:max_tokens])
            return truncated + "\n\n... [content truncated]"
        except (ImportError, Exception):
            # Fallback: rough char-based estimate (~4 chars per token)
            max_chars = max_tokens * 4
            if len(content) <= max_chars:
                return content
            return content[:max_chars] + "\n\n... [content truncated]"
