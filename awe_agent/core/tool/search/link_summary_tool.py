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

Supports any backend available through AweAgent LLMClient (OpenAI, Azure,
Anthropic, OpenAI Response API, Ark, etc.).

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
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import Message
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
        llm_params: LLM generation parameters (e.g. ``max_tokens``).
            Overrides values from YAML config. Defaults: ``max_tokens=10240``.
        llm: Optional pre-configured :class:`LLMClient` instance.
            Useful for testing or when you already have a client.
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
        llm: LLMClient | None = None,
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

        # Injected or lazy-loaded LLMClient and resolved generation params
        self._llm: LLMClient | None = llm
        # When client is injected, resolve params immediately
        if llm is not None:
            self._llm_params: dict[str, Any] = {"max_tokens": 10240}
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
        """Call LLM to summarize fetched content.

        Uses the unified LLMClient interface — works with any backend
        (OpenAI, Anthropic, Response API, Ark, etc.). Retries with
        exponential backoff on transient failures.
        """
        self._ensure_llm_loaded()

        if self._llm is None:
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
                response = await self._llm.chat(
                    messages=[
                        Message(role="system", content=self._system_prompt),
                        Message(role="user", content=user_message),
                    ],
                    **self._llm_params,
                )
                return f"Summary of {url}:\n\n{response.content}"

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
        """Lazy-load LLM from YAML config or environment variables."""
        if self._llm is not None:
            return

        model = os.environ.get("LINK_SUMMARY_MODEL")
        if not model and not self._llm_config_path:
            logger.info(
                "No LINK_SUMMARY_MODEL or LINK_SUMMARY_CONFIG_PATH set — "
                "link_summary will return raw content without summarization."
            )
            return

        config_dict: dict[str, Any] = {}
        if self._llm_config_path:
            config_dict = load_yaml(self._llm_config_path)

        # Resolve generation params: defaults ← YAML ← constructor override
        self._llm_params = {"max_tokens": 10240}
        self._llm_params.update(config_dict.get("params", {}))
        if self._llm_params_override:
            self._llm_params.update(self._llm_params_override)

        # Apply env-var overrides and model name, then build full LLMConfig
        # from the YAML dict so all fields (thinking, reasoning, extra, etc.)
        # are inherited — not just backend/api_key/base_url/model.
        if model:
            config_dict["model"] = model
        config_dict.setdefault("model", "gpt-4o-mini")
        config_dict.setdefault("backend", "openai")
        if not config_dict.get("api_key"):
            config_dict["api_key"] = os.environ.get("OPENAI_API_KEY")
        if not config_dict.get("base_url"):
            env_url = os.environ.get("OPENAI_BASE_URL")
            if env_url:
                config_dict["base_url"] = env_url

        try:
            llm_config = LLMConfig(**{
                k: v for k, v in config_dict.items()
                if k in LLMConfig.model_fields
            })
            self._llm = LLMClient(llm_config)
        except Exception as exc:
            logger.warning("Failed to create LLM client for link_summary: %s", exc)
            self._llm = None
