"""System prompts for the SearchSWE agent.

Prompt content is defined in :mod:`awe_agent.tasks.beyond_swe.prompt.system`.
This module re-exports the registry and provides the ``get_system_prompt()`` accessor.
"""

from __future__ import annotations

from awe_agent.tasks.beyond_swe.prompt.system import (
    NO_TOOL_CALL_PROMPT,
    SEARCH_SYSTEM_PROMPT_BEYONDSWE,
    SEARCH_SYSTEM_PROMPT_DOMAINFIX,
    SYSTEM_PROMPT_BEYONDSWE,
    SYSTEM_PROMPTS,
)

__all__ = [
    "NO_TOOL_CALL_PROMPT",
    "SYSTEM_PROMPT_BEYONDSWE",
    "SEARCH_SYSTEM_PROMPT_BEYONDSWE",
    "SEARCH_SYSTEM_PROMPT_DOMAINFIX",
    "SYSTEM_PROMPTS",
    "get_system_prompt",
]


def get_system_prompt(key: str) -> str:
    """Get a system prompt by key. Raises KeyError for unknown keys."""
    if key not in SYSTEM_PROMPTS:
        raise KeyError(
            f"Unknown system prompt key: {key!r}. "
            f"Available: {list(SYSTEM_PROMPTS.keys())}"
        )
    return SYSTEM_PROMPTS[key]
