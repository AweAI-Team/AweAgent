"""SearchTool — web search with pluggable backends and anti-hack constraint filtering."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.search.constraints import SearchConstraints

logger = logging.getLogger(__name__)

# Default fields to include in formatted results
_DEFAULT_RESULT_SCHEME = ["position", "title", "description", "snippets", "url"]


class SearchTool(Tool):
    """Web search with anti-hack constraint filtering.

    Supports pluggable search backends discovered through entry-points
    (``awe_agent.search_backend`` group) or injected directly.

    Backend resolution order:
        1. Explicit ``backend`` argument (instance or registry name).
        2. Explicit ``search_fn`` callable (legacy / testing).
        3. Auto-discover from ``SEARCH_BACKEND`` env var or registry.

    Args:
        engine: Search engine name passed to the backend. Defaults to
            env ``ENGINE`` or ``"google"``.
        constraints: Optional constraints for result filtering.
        max_attempts: Number of retry attempts for search calls.
        result_scheme: Fields to include in formatted output.
        backend: A search backend instance (must have an async ``search`` method),
            or a registered backend name (e.g. ``"serpapi"``).
            Auto-discovered if not provided.
        search_fn: Optional raw callable for search (legacy / testing).
            Signature: ``async (query, num, start, engine) -> list[dict]``.
    """

    def __init__(
        self,
        engine: str | None = None,
        constraints: SearchConstraints | None = None,
        max_attempts: int = 1,
        result_scheme: list[str] | None = None,
        backend: str | Any | None = None,
        search_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._engine = engine or os.environ.get("ENGINE", "google")
        self._constraints = constraints or SearchConstraints()
        self._max_attempts = max_attempts
        self._result_scheme = result_scheme or list(_DEFAULT_RESULT_SCHEME)

        # Resolve backend
        if isinstance(backend, str):
            from awe_agent.core.tool.search.backends.search import search_backend_registry

            backend_cls = search_backend_registry.get(backend)
            self._backend: Any | None = backend_cls()
        elif backend is not None:
            self._backend = backend
        else:
            self._backend = None  # lazy-discovered on first use

        # Legacy: direct callable injection (takes priority when set)
        self._search_fn: Callable[..., Any] | None = search_fn

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Use this when you need to look up "
            "external library documentation, debug unfamiliar errors, or research "
            "best practices. Do NOT use this for information that should be in the "
            "local codebase. Supports single or batch queries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {
                            "type": "string",
                            "description": "A single search query.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple search queries (batch).",
                        },
                    ],
                    "description": "The search query (string or list of strings).",
                },
                "num": {
                    "type": "integer",
                    "description": "Number of results to return (default 10).",
                },
                "start": {
                    "type": "integer",
                    "description": "Starting offset for results (pagination).",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        query = params.get("query", "")
        num = params.get("num", 10)
        start = params.get("start", 0)

        if not query:
            return "Error: empty search query."

        # Normalize to list of queries
        queries = query if isinstance(query, list) else [query]
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            return "Error: empty search query."

        parts: list[str] = []
        for q in queries:
            results = await self._search_single(q, num=num, start=start)
            filtered, filtered_count = self._constraints.filter_search_results(results)
            parts.append(self._format_results(q, filtered, filtered_count))

        return "\n\n".join(parts)

    # ── Internal ─────────────────────────────────────────────────────

    async def _search_single(
        self,
        query: str,
        num: int,
        start: int,
    ) -> list[dict]:
        """Execute a single search query with retries."""
        # Priority 1: explicit search_fn (legacy / testing)
        if self._search_fn is not None:
            return await self._call_search_fn(query, num, start)

        # Priority 2: backend (explicit or auto-discovered)
        backend = self._ensure_backend()
        if backend is None:
            return []

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return await backend.search(
                    query, num=num, start=start, engine=self._engine,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Search attempt %d/%d failed for query %r: %s",
                    attempt + 1, self._max_attempts, query, exc,
                )

        logger.error("All search attempts failed for query %r: %s", query, last_error)
        return []

    async def _call_search_fn(
        self, query: str, num: int, start: int,
    ) -> list[dict]:
        """Call a raw search_fn callable (legacy path)."""
        import json

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                result = await self._search_fn(
                    query=query, num=num, start=start, engine=self._engine,
                )
                # Parse JSON string if returned
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        return [{"description": result}]
                if isinstance(result, list):
                    return result
                if isinstance(result, dict):
                    return result.get("results", result.get("organic", [result]))
                return []
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "search_fn attempt %d/%d failed for query %r: %s",
                    attempt + 1, self._max_attempts, query, exc,
                )
        logger.error("All search_fn attempts failed for query %r: %s", query, last_error)
        return []

    def _ensure_backend(self) -> Any:
        """Lazy-discover a search backend if not yet resolved."""
        if self._backend is not None:
            return self._backend

        from awe_agent.core.tool.search.backends.search import get_search_backend

        self._backend = get_search_backend()
        if self._backend is None:
            logger.warning(
                "No search backend available. Set SEARCH_BACKEND env var or "
                "install a search backend plugin (e.g. pip install awe-agent[search])."
            )
        return self._backend

    def _format_results(
        self, query: str, results: list[dict], filtered_count: int,
    ) -> str:
        """Format results with optional filtered-count warning."""
        lines = [f"Search results for: {query}"]

        if filtered_count > 0:
            lines.append(
                f"WARNING: {filtered_count} result(s) filtered by security constraints."
            )

        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        lines.append("")
        for i, item in enumerate(results, 1):
            entry_parts: list[str] = []
            for field_name in self._result_scheme:
                value = item.get(field_name)
                if value is not None:
                    if field_name == "position":
                        continue  # use our own numbering
                    if isinstance(value, list):
                        value = " ".join(str(v) for v in value)
                    entry_parts.append(f"  {field_name}: {value}")
            if entry_parts:
                lines.append(f"[{i}]")
                lines.extend(entry_parts)
                lines.append("")

        return "\n".join(lines)
