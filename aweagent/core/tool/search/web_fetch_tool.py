"""WebFetchTool — fetch a web page, then run a user prompt against its content.

Takes a URL and a prompt: fetches the page via :class:`WebFetchRawTool`, then
sends the content + prompt to a configured LLM and returns its response.

LLM Configuration
=================

Three ways to configure the LLM backend (can be combined):

1. **YAML config only** (recommended for production)::

       export WEB_FETCH_CONFIG_PATH=/path/to/configs/llm/web_fetch/azure.yaml

   The YAML file provides all settings: backend, api_key, model, params, etc.

2. **Environment variables only** (quick setup, no YAML needed)::

       export WEB_FETCH_MODEL=gpt-4o
       export OPENAI_API_KEY=sk-...
       export OPENAI_BASE_URL=https://api.openai.com/v1   # optional

   Uses the OpenAI backend by default. ``OPENAI_BASE_URL`` is optional
   (defaults to the official OpenAI endpoint).

3. **Both** (YAML config + model override)::

       export WEB_FETCH_CONFIG_PATH=/path/to/configs/llm/web_fetch/azure.yaml
       export WEB_FETCH_MODEL=gpt-5.2-2025-12-11

   Backend, api_key, and params come from YAML; ``WEB_FETCH_MODEL``
   overrides only the model name. Useful for quick model switching without
   editing the config file.

If neither ``WEB_FETCH_MODEL`` nor ``WEB_FETCH_CONFIG_PATH`` is set,
the tool falls back to returning raw fetched content without summarization.

Supports any backend available through AweAgent LLMClient (OpenAI, Azure,
Anthropic, OpenAI Response API, Ark, etc.).

Prompt Configuration
====================

- ``WEB_FETCH_PROMPT_NAME``: Select a built-in preset (``default``, ``code``, ``paper``).
- ``WEB_FETCH_PROMPT_PATH``: Path to a custom prompt file (Markdown).
- Or pass ``system_prompt=...`` in the constructor (highest priority).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from aweagent.core.config.loader import load_yaml
from aweagent.core.llm.client import LLMClient
from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import Message
from aweagent.core.runtime.protocol import RuntimeSession
from aweagent.core.tool.protocol import Tool
from aweagent.core.tool.search.constraints import SearchConstraints
from aweagent.core.tool.search.prompts import resolve_prompt
from aweagent.core.tool.search.web_fetch_raw_tool import WebFetchRawTool

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_TOKENS = 25000
_DEFAULT_MAX_ATTEMPTS = 3


class WebFetchTool(Tool):
    """Fetch and summarize web content with anti-hack URL blocking.

    Two-step process: :class:`WebFetchRawTool` fetches the content, then an
    LLM summarizes it based on the user's prompt. See module docstring for
    LLM and prompt configuration details.

    Args:
        constraints: Optional constraints for URL blocking.
        max_content_tokens: Maximum tokens for fetched content.
        max_attempts: Retry attempts for LLM summarization.
        system_prompt: Custom system prompt (defaults to ``WEB_FETCH_PROMPT``).
        llm_config_path: Path to YAML config for the LLM client.
        llm_config: Inline LLM configuration dict or :class:`LLMConfig`.
            Takes priority over ``llm_config_path``.
        llm_params: LLM generation parameters (e.g. ``max_tokens``).
            Overrides values from YAML config. Defaults: ``max_tokens=10240``.
        llm: Optional pre-configured :class:`LLMClient` instance.
            Useful for testing or when you already have a client.
        reader: Optional :class:`WebFetchRawTool` instance. Defaults to creating
            one internally with the same constraints.
    """

    def __init__(
        self,
        constraints: SearchConstraints | None = None,
        max_content_tokens: int = _DEFAULT_MAX_CONTENT_TOKENS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        system_prompt: str | None = None,
        llm_config_path: str | None = None,
        llm_config: dict[str, Any] | LLMConfig | None = None,
        llm_params: dict[str, Any] | None = None,
        llm: LLMClient | None = None,
        reader: WebFetchRawTool | None = None,
    ) -> None:
        self._constraints = constraints or SearchConstraints()
        self._system_prompt = resolve_prompt(system_prompt)
        self._max_attempts = max_attempts
        self._llm_config_path = llm_config_path or os.environ.get(
            "WEB_FETCH_CONFIG_PATH"
        )
        self._llm_config_inline = llm_config
        # LLM generation parameters — overridable via constructor or YAML config.
        # Resolved lazily in _ensure_llm_loaded(); constructor values take priority.
        self._llm_params_override = llm_params

        # Internal reader — injected or created with shared constraints
        self._reader = reader or WebFetchRawTool(
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
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch a web page and run a prompt against its content. Provide a URL "
            "and a prompt describing what information you need. The tool fetches "
            "the page and returns an LLM-generated response focused on your prompt."
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
                "prompt": {
                    "type": "string",
                    "description": "The prompt to run on the fetched content.",
                },
            },
            "required": ["url", "prompt"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        url = params.get("url", "").strip()
        prompt = params.get("prompt", "").strip()

        if not url:
            return "Error: empty URL."
        if not prompt:
            return "Error: empty prompt. Please describe what information you need."

        # Check URL against constraints
        if self._constraints.is_url_blocked(url):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked by security constraints. "
                "This URL may point to the target repository and accessing it is "
                "not allowed during evaluation."
            )

        # Fetch content via WebFetchRawTool
        content = await self._reader.execute({"url": url}, session=session)

        # Check if fetch returned an error
        if content.startswith("Error:") or content.startswith("ACCESS DENIED:"):
            return content
        if content.startswith("No content returned"):
            return content

        # Summarize via LLM
        return await self._summarize_content(content, url, prompt)

    async def _summarize_content(
        self, content: str, url: str, prompt: str,
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

        # Page Content first — keeps prompt-cache prefix stable across multiple
        # prompts on the same URL (the large content block stays cached; only
        # the small prompt at the end varies).
        user_message = (
            f"## Page Content\n{content}\n\n"
            f"## URL\n{url}\n\n"
            f"## Prompt\n{prompt}"
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

        model = os.environ.get("WEB_FETCH_MODEL")
        has_config = self._llm_config_inline is not None or self._llm_config_path
        if not model and not has_config:
            logger.info(
                "No web_fetch LLM config or WEB_FETCH_MODEL set — "
                "web_fetch will return raw content without summarization."
            )
            return

        config_dict: dict[str, Any] = {}
        if isinstance(self._llm_config_inline, LLMConfig):
            config_dict = self._llm_config_inline.model_dump()
        elif self._llm_config_inline is not None:
            config_dict = dict(self._llm_config_inline)
        elif self._llm_config_path:
            config_dict = load_yaml(self._llm_config_path)

        # Resolve generation params: defaults ← YAML ← constructor override
        self._llm_params = {"max_tokens": 10240}
        self._llm_params.update(config_dict.get("params", {}))
        if self._llm_params_override:
            self._llm_params.update(self._llm_params_override)

        # Inline config beats config-file loading, while env vars remain a
        # backwards-compatible fallback for older rollout scripts.
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
            logger.warning("Failed to create LLM client for web_fetch: %s", exc)
            self._llm = None
