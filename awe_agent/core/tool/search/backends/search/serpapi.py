"""SerpAPI search backend — Google search via SerpAPI."""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_SERPAPI_ENDPOINT = "https://serpapi.com/search"


class SerpAPIBackend:
    """Search backend using SerpAPI.

    Requires the ``SERPAPI_API_KEY`` environment variable (or explicit ``api_key``).

    Args:
        api_key: SerpAPI key. Falls back to ``SERPAPI_API_KEY`` env var.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.environ.get("SERPAPI_API_KEY", "")
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def search(
        self,
        query: str,
        *,
        num: int = 10,
        start: int = 0,
        engine: str = "google",
    ) -> list[dict[str, Any]]:
        """Execute a search query via SerpAPI.

        Args:
            query: The search query string.
            num: Number of results to return.
            start: Starting offset for pagination.
            engine: Search engine name (default ``"google"``).

        Returns:
            A list of result dicts with keys: position, title, url, description, snippets.
        """
        if not self._api_key:
            logger.error("SERPAPI_API_KEY not set. Cannot perform search.")
            return []

        params = {
            "api_key": self._api_key,
            "engine": engine,
            "q": query,
            "num": num,
            "start": start,
        }

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.get(_SERPAPI_ENDPOINT, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[dict[str, Any]] = []
        for item in data.get("organic_results", []):
            snippet = item.get("snippet", "")
            results.append({
                "position": item.get("position", 0),
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": snippet,
                "snippets": [snippet] if snippet else [],
            })
        return results
