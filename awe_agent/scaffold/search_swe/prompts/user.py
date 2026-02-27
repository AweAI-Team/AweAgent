"""User prompts for the SearchSWE agent.

The scaffold layer merges prompt registries from all task modules and
performs conflict detection. Each task declares its own ``USER_PROMPTS``
dict; this module combines them into a single lookup table.
"""

from __future__ import annotations

# ── Import task-level registries ──────────────────────────────────────────────

from awe_agent.tasks.beyond_swe.prompt.user import (
    USER_PROMPTS as _BEYOND_SWE_USER_PROMPTS,
)
from awe_agent.tasks.scale_swe.prompt import (
    USER_PROMPTS as _SCALE_SWE_USER_PROMPTS,
)

# ── Merge with conflict detection ────────────────────────────────────────────

USER_PROMPTS: dict[str, str] = {}


def _merge(source: dict[str, str], label: str) -> None:
    for key, prompt in source.items():
        if key in USER_PROMPTS:
            raise ValueError(
                f"Duplicate user prompt key {key!r} from {label}. "
                f"Already registered: {list(USER_PROMPTS.keys())}"
            )
        USER_PROMPTS[key] = prompt


_merge(_BEYOND_SWE_USER_PROMPTS, "beyond_swe")
_merge(_SCALE_SWE_USER_PROMPTS,  "scale_swe")


# ── Accessor ─────────────────────────────────────────────────────────────────

__all__ = [
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
