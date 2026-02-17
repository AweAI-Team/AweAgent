"""LinkSummaryTool — fetch and summarize web content with anti-hack URL blocking.

LLM Configuration
=================

Three ways to configure the LLM backend (can be combined):

1. **YAML config only** (recommended for production)::

       export LINK_SUMMARY_CONFIG_PATH=/path/to/configs/llm/link_summary/azure.yaml

   The YAML file provides all settings: backend, api_key, model, params, etc.

2. **Environment variables only** (quick setup, no YAML needed)::

       export LINK_SUMMARY_MODEL=gpt-4o
       export OPENAI_API_KEY=sk-...
       export OPENAI_BASE_URL=https://api.openai.com/v1   # optional

   Uses the OpenAI backend by default. ``OPENAI_BASE_URL`` is optional
   (defaults to the official OpenAI endpoint).

3. **Both** (YAML config + model override)::

       export LINK_SUMMARY_CONFIG_PATH=/path/to/configs/llm/link_summary/azure.yaml
       export LINK_SUMMARY_MODEL=gpt-5.2-2025-12-11

   Backend, api_key, and params come from YAML; ``LINK_SUMMARY_MODEL``
   overrides only the model name. Useful for quick model switching without
   editing the config file.

If neither ``LINK_SUMMARY_MODEL`` nor ``LINK_SUMMARY_CONFIG_PATH`` is set,
the tool falls back to returning raw fetched content without summarization.

Prompt Configuration
====================

- ``LINK_SUMMARY_PROMPT_NAME``: Select a built-in preset (``default``, ``code``, ``paper``).
- ``LINK_SUMMARY_PROMPT_PATH``: Path to a custom prompt file (Markdown).
- Or pass ``system_prompt=...`` in the constructor (highest priority).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from awe_agent.core.config.loader import load_yaml
from awe_agent.core.llm.client import create_async_client
from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.link_reader_tool import LinkReaderTool
from awe_agent.core.tool.search.prompts import resolve_prompt

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_TOKENS = 25000
_DEFAULT_MAX_ATTEMPTS = 3


class LinkSummaryTool(Tool):
    """Fetch and summarize web content with anti-hack URL blocking.

    Two-step process: :class:`LinkReaderTool` fetches the content, then an
    LLM summarizes it based on the user's goal. See module docstring for
    LLM and prompt configuration details.

    Args:
        constraints: Optional constraints for URL blocking.
        max_content_tokens: Maximum tokens for fetched content.
        max_attempts: Retry attempts for LLM summarization.
        system_prompt: Custom system prompt (defaults to ``LINK_SUMMARY_PROMPT``).
        llm_config_path: Path to YAML config for the LLM client.
        llm_params: LLM generation parameters (e.g. ``max_completion_tokens``).
            Overrides values from YAML config. Defaults: ``max_completion_tokens=32768``.
        llm_client: Optional pre-configured async LLM client (OpenAI-compatible).
            Useful for testing or when you already have a client instance.
        llm_model: Model name to use with ``llm_client``. Required when
            ``llm_client`` is provided.
        reader: Optional :class:`LinkReaderTool` instance. Defaults to creating
            one internally with the same constraints.
    """

    def __init__(
        self,
        constraints: SearchConstraints | None = None,
        max_content_tokens: int = _DEFAULT_MAX_CONTENT_TOKENS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        system_prompt: str | None = None,
        llm_config_path: str | None = None,
        llm_params: dict[str, Any] | None = None,
        llm_client: Any = None,
        llm_model: str | None = None,
        reader: LinkReaderTool | None = None,
    ) -> None:
        self._constraints = constraints or SearchConstraints()
        self._system_prompt = resolve_prompt(system_prompt)
        self._max_attempts = max_attempts
        self._llm_config_path = llm_config_path or os.environ.get(
            "LINK_SUMMARY_CONFIG_PATH"
        )
        # LLM generation parameters — overridable via constructor or YAML config.
        # Resolved lazily in _ensure_llm_loaded(); constructor values take priority.
        self._llm_params_override = llm_params

        # Internal reader — injected or created with shared constraints
        self._reader = reader or LinkReaderTool(
            constraints=self._constraints,
            max_content_tokens=max_content_tokens,
        )

        # Injected or lazy-loaded LLM client and resolved generation params
        self._llm_client: Any = llm_client
        self._llm_model: str | None = llm_model
        # When client is injected, resolve params immediately
        if llm_client is not None:
            self._llm_params: dict[str, Any] = {"max_completion_tokens": 32768}
            if llm_params:
                self._llm_params.update(llm_params)
        else:
            self._llm_params = {}

    @property
    def name(self) -> str:
        return "link_summary"

    @property
    def description(self) -> str:
        return (
            "Fetch and summarize a web page. Provide a URL and a goal describing "
            "what information you need. The tool will fetch the page content and "
            "return an LLM-generated summary focused on your goal."
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
                "goal": {
                    "type": "string",
                    "description": (
                        "What information you need from this page. "
                        "Be specific to get a focused summary."
                    ),
                },
            },
            "required": ["url", "goal"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        url = params.get("url", "").strip()
        goal = params.get("goal", "").strip()

        if not url:
            return "Error: empty URL."
        if not goal:
            return "Error: empty goal. Please describe what information you need."

        # Check URL against constraints
        if self._constraints.is_url_blocked(url):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked by security constraints. "
                "This URL may point to the target repository and accessing it is "
                "not allowed during evaluation."
            )

        # Fetch content via LinkReaderTool
        content = await self._reader.execute({"url": url}, session=session)

        # Check if fetch returned an error
        if content.startswith("Error:") or content.startswith("ACCESS DENIED:"):
            return content
        if content.startswith("No content returned"):
            return content

        # Summarize via LLM
        summary = await self._summarize_content(content, url, goal)
        return summary

    async def _summarize_content(
        self, content: str, url: str, goal: str,
    ) -> str:
        """Call LLM with system prompt + user message.

        Lazy-loads LLM client from YAML config. Supports OpenAI, AzureOpenAI,
        and Ark backends. Retries with exponential backoff.
        """
        self._ensure_llm_loaded()

        if self._llm_client is None:
            # Fallback: return raw content with a note
            return (
                f"Content from {url} (no LLM configured for summarization):\n\n"
                f"{content}"
            )

        user_message = (
            f"## URL\n{url}\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Page Content\n{content}"
        )

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._llm_client.chat.completions.create(
                    model=self._llm_model,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    **self._llm_params,
                )
                result = response.choices[0].message.content
                return f"Summary of {url}:\n\n{result}"

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM summarization attempt %d/%d failed: %s",
                    attempt + 1, self._max_attempts, exc,
                )
                if attempt < self._max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)

        # All retries failed — return raw content with error note
        return (
            f"Failed to summarize (after {self._max_attempts} attempts: {last_error}).\n"
            f"Raw content from {url}:\n\n{content}"
        )

    def _ensure_llm_loaded(self) -> None:
        """Lazy-load LLM config from YAML file and create async client."""
        if self._llm_client is not None:
            return

        model = os.environ.get("LINK_SUMMARY_MODEL")
        if not model and not self._llm_config_path:
            logger.info(
                "No LINK_SUMMARY_MODEL or LINK_SUMMARY_CONFIG_PATH set — "
                "link_summary will return raw content without summarization."
            )
            return

        config: dict[str, Any] = {}
        if self._llm_config_path:
            config = load_yaml(self._llm_config_path)

        # Resolve model name and generation params
        self._llm_model = model or config.get("model", "gpt-4o-mini")
        # Defaults ← YAML config "params" ← constructor llm_params (highest priority)
        self._llm_params = {"max_completion_tokens": 10240}
        self._llm_params.update(config.get("params", {}))
        if self._llm_params_override:
            self._llm_params.update(self._llm_params_override)

        try:
            self._llm_client = create_async_client(
                backend=config.get("backend", "openai"),
                api_key=config.get("api_key") or os.environ.get("OPENAI_API_KEY"),
                base_url=config.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
                azure_endpoint=config.get("azure_endpoint"),
                api_version=config.get("api_version", "2024-02-01"),
            )
        except Exception as exc:
            logger.warning("Failed to create LLM client for link_summary: %s", exc)
            self._llm_client = None
