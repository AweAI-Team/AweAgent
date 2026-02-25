"""Jina Reader backend — convert URLs to LLM-friendly content via r.jina.ai.

Jina Reader converts any URL into clean, LLM-friendly markdown text.
Supports web pages (including JS-rendered SPAs), PDFs, and image captioning.

API reference: https://jina.ai/reader/
GitHub: https://github.com/jina-ai/reader

Environment variables:
    JINA_API_KEY: Optional API key for higher rate limits.
        Without key: 20 RPM.
        With free key: 500 RPM (new keys get 10M free tokens).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

_JINA_READER_ENDPOINT = "https://r.jina.ai/"


class JinaReaderBackend:
    """Link reader backend using Jina Reader API.

    Fetches URL content as clean markdown via ``https://r.jina.ai/``.
    Works without an API key (20 RPM), or with one for higher limits (500 RPM).

    Args:
        api_key: Jina API key. Falls back to ``JINA_API_KEY`` env var.
            Optional — the API works without a key at lower rate limits.
        timeout: HTTP request timeout in seconds. Default 300s to handle
            large pages and JS-rendered content.
        return_format: Content format. One of ``"markdown"`` (default),
            ``"html"``, ``"text"``, ``"screenshot"``, ``"pageshot"``.
        target_selector: CSS selector to extract specific page elements.
            E.g. ``"article"`` or ``".main-content"``.
        wait_for_selector: CSS selector to wait for before extraction.
            Useful for JS-rendered SPAs.
        remove_selector: CSS selector for elements to remove from output.
        with_links_summary: Include a summary of all links found on the page.
        with_images_summary: Include a summary of all images found on the page.
        with_generated_alt: Enable AI-generated image captions via VLM.
            Adds latency but useful for multimodal reasoning.
        no_cache: Bypass Jina's server-side cache (default 3600s TTL).
        engine: Rendering engine. One of ``"browser"`` (default, headless Chrome),
            ``"direct"`` (fast, no JS), ``"cf-browser-rendering"``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 300,
        return_format: str = "markdown",
        target_selector: str | None = None,
        wait_for_selector: str | None = None,
        remove_selector: str | None = None,
        with_links_summary: bool = False,
        with_images_summary: bool = False,
        with_generated_alt: bool = False,
        no_cache: bool = False,
        engine: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._return_format = return_format
        self._target_selector = target_selector
        self._wait_for_selector = wait_for_selector
        self._remove_selector = remove_selector
        self._with_links_summary = with_links_summary
        self._with_images_summary = with_images_summary
        self._with_generated_alt = with_generated_alt
        self._no_cache = no_cache
        self._engine = engine

    async def read_link(self, url: str) -> str:
        """Fetch URL content as LLM-friendly markdown via Jina Reader.

        Args:
            url: The URL to fetch.

        Returns:
            Extracted content as markdown string.

        Raises:
            RuntimeError: On HTTP errors (4xx/5xx) with Jina's error message.
            aiohttp.ClientError: On connection/timeout errors.
        """
        headers = self._build_headers()

        # POST with JSON body — handles hash-based routes and special URLs
        payload: dict[str, Any] = {"url": url}

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            async with session.post(
                _JINA_READER_ENDPOINT,
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "Jina Reader returned HTTP %d for %r: %s",
                        resp.status, url, body[:200],
                    )
                    # Extract Jina's error message for a more informative exception
                    detail = ""
                    try:
                        import json
                        err_data = json.loads(body)
                        detail = err_data.get("message", "")
                    except (json.JSONDecodeError, AttributeError):
                        detail = body[:200]
                    raise RuntimeError(
                        f"Jina Reader HTTP {resp.status} for '{url}': {detail}"
                    )

                data = await resp.json()

        # Extract content from JSON response
        content = self._extract_content(data)
        if not content:
            logger.warning("Jina Reader returned empty content for %r", url)

        return content

    def _build_headers(self) -> dict[str, str]:
        """Build request headers from configuration."""
        headers: dict[str, str] = {
            "Accept": "application/json",
            "X-Return-Format": self._return_format,
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        if self._target_selector:
            headers["X-Target-Selector"] = self._target_selector
        if self._wait_for_selector:
            headers["X-Wait-For-Selector"] = self._wait_for_selector
        if self._remove_selector:
            headers["X-Remove-Selector"] = self._remove_selector
        if self._with_links_summary:
            headers["X-With-Links-Summary"] = "true"
        if self._with_images_summary:
            headers["X-With-Images-Summary"] = "true"
        if self._with_generated_alt:
            headers["X-With-Generated-Alt"] = "true"
        if self._no_cache:
            headers["X-No-Cache"] = "true"
        if self._engine:
            headers["X-Engine"] = self._engine

        return headers

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        """Extract content from Jina Reader JSON response.

        Response structure::

            {
                "code": 200,
                "status": 20000,
                "data": {
                    "url": "...",
                    "title": "...",
                    "content": "...",         # main content (markdown)
                    "description": "...",
                    "links": {"...": "..."},  # if X-With-Links-Summary
                    "images": {"...": "..."}  # if X-With-Images-Summary
                }
            }
        """
        # Handle nested data structure
        inner = data.get("data", data)

        title = inner.get("title", "")
        content = inner.get("content", "")
        description = inner.get("description", "")

        if not content and not title:
            return ""

        parts: list[str] = []
        if title:
            parts.append(f"# {title}")
            parts.append("")
        if description and description not in (content[:200] if content else ""):
            parts.append(f"> {description}")
            parts.append("")
        if content:
            parts.append(content)

        # Append links summary if present
        links = inner.get("links")
        if links and isinstance(links, dict):
            parts.append("")
            parts.append("## Links")
            for text, href in links.items():
                parts.append(f"- [{text}]({href})")

        # Append images summary if present
        images = inner.get("images")
        if images and isinstance(images, dict):
            parts.append("")
            parts.append("## Images")
            for alt, src in images.items():
                parts.append(f"- ![{alt}]({src})")

        return "\n".join(parts)
