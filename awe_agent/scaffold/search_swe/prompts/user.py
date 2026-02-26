"""User prompts for the SearchSWE agent.

Prompt content is defined in :mod:`awe_agent.tasks.beyond_swe.prompt.user`.
This module re-exports the registry and provides the ``get_user_prompt()`` accessor.
"""

from __future__ import annotations

from awe_agent.tasks.beyond_swe.prompt.user import (
    CROSSREPO_PROMPT,
    DEPMIGRATE_PROMPT,
    DOC2REPO_PROMPT,
    DOMAINFIX_PROMPT,
    SEARCH_CROSSREPO_PROMPT,
    SEARCH_DEPMIGRATE_PROMPT,
    SEARCH_DOC2REPO_PROMPT,
    SEARCH_DOMAINFIX_PROMPT,
    USER_PROMPTS,
)

__all__ = [
    "DOC2REPO_PROMPT",
    "CROSSREPO_PROMPT",
    "DEPMIGRATE_PROMPT",
    "DOMAINFIX_PROMPT",
    "SEARCH_DOC2REPO_PROMPT",
    "SEARCH_CROSSREPO_PROMPT",
    "SEARCH_DEPMIGRATE_PROMPT",
    "SEARCH_DOMAINFIX_PROMPT",
    "USER_PROMPTS",
    "get_user_prompt",
]


def get_user_prompt(key: str) -> str:
    """Get a user prompt template by key. Raises KeyError for unknown keys."""
    if key not in USER_PROMPTS:
        raise KeyError(
            f"Unknown user prompt key: {key!r}. "
            f"Available: {list(USER_PROMPTS.keys())}"
        )
    return USER_PROMPTS[key]
