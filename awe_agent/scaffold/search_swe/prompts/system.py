"""System prompts for the SearchSWE agent.

The scaffold layer merges prompt registries from all task modules and
performs conflict detection. Each task declares its own ``SYSTEM_PROMPTS``
dict; this module combines them into a single lookup table.
"""

from __future__ import annotations

# ── Import task-level registries ──────────────────────────────────────────────

from awe_agent.tasks.beyond_swe.prompt.system import (
    NO_TOOL_CALL_PROMPT,
    SYSTEM_PROMPTS as _BEYOND_SWE_SYSTEM_PROMPTS,
)
from awe_agent.tasks.scale_swe.prompt import (
    SYSTEM_PROMPTS as _SCALE_SWE_SYSTEM_PROMPTS,
)

# ── Merge with conflict detection ────────────────────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {}


def _merge(source: dict[str, str], label: str) -> None:
    for key, prompt in source.items():
        if key in SYSTEM_PROMPTS:
            raise ValueError(
                f"Duplicate system prompt key {key!r} from {label}. "
                f"Already registered: {list(SYSTEM_PROMPTS.keys())}"
            )
        SYSTEM_PROMPTS[key] = prompt


_merge(_BEYOND_SWE_SYSTEM_PROMPTS, "beyond_swe")
_merge(_SCALE_SWE_SYSTEM_PROMPTS,  "scale_swe")


# ── Accessor ─────────────────────────────────────────────────────────────────

__all__ = [
    "NO_TOOL_CALL_PROMPT",
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
