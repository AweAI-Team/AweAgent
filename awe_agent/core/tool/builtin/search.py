"""Search tool — web search for external information during coding tasks.

Provides a pluggable search interface. Concrete implementations can use
different search providers (Bing, Google, Tavily, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool

logger = logging.getLogger(__name__)


class SearchTool(Tool):
    """Search the web for relevant information.

    This tool allows the agent to research external libraries, documentation,
    error messages, and best practices during coding tasks.

    The actual search is performed by a pluggable backend. By default,
    it delegates to a runtime shell command using `curl` with a search API.
    Override ``_search()`` for custom providers.
    """

    def __init__(
        self,
        max_results: int = 10,
        blocked_domains: list[str] | None = None,
    ) -> None:
        self._max_results = max_results
        self._blocked_domains = blocked_domains or []

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Use this when you need to look up "
            "external library documentation, debug unfamiliar errors, or research "
            "best practices. Do NOT use this for information that should be in the "
            "local codebase."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and technical.",
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
        if not query.strip():
            return "Error: empty search query."

        return await self._search(query, session)

    async def _search(
        self,
        query: str,
        session: RuntimeSession | None = None,
    ) -> str:
        """Perform the search. Override in subclasses for custom providers.

        Default implementation returns a placeholder instructing the user
        to configure a search backend.
        """
        return (
            f"Search for: {query}\n\n"
            "No search backend configured. To enable web search, subclass "
            "SearchTool and implement _search(), or install a search provider "
            "plugin."
        )


class LinkSummaryTool(Tool):
    """Fetch and summarize a web page.

    Allows the agent to read documentation pages, blog posts, or
    other web content relevant to the coding task.
    """

    def __init__(self, max_length: int = 8000) -> None:
        self._max_length = max_length

    @property
    def name(self) -> str:
        return "link_summary"

    @property
    def description(self) -> str:
        return (
            "Fetch and summarize a web page. Use this to read documentation, "
            "API references, or relevant technical content from a URL."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and summarize.",
                },
            },
            "required": ["url"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        url = params.get("url", "")
        if not url.strip():
            return "Error: empty URL."

        return await self._fetch_and_summarize(url, session)

    async def _fetch_and_summarize(
        self,
        url: str,
        session: RuntimeSession | None = None,
    ) -> str:
        """Fetch URL content. Override in subclasses for custom implementations."""
        if session is None:
            return "Error: LinkSummaryTool requires a runtime session."

        result = await session.execute(
            f"curl -sL --max-time 30 '{url}' | head -c {self._max_length * 2}",
            timeout=45,
        )
        content = result.stdout
        if not content:
            return f"Failed to fetch content from {url}"

        if len(content) > self._max_length:
            content = content[:self._max_length] + "\n\n... [content truncated]"

        return f"Content from {url}:\n\n{content}"
