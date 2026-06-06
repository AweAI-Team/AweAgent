"""User prompt registry for DeepSearch tasks.

DeepSearch normally receives its user prompt from the task layer.  The registry
exists so future DeepSearch tasks can route to shared user prompt templates.
"""

from __future__ import annotations

USER_PROMPTS: dict[str, str] = {
    "raw": "{question}",
}


def get_user_prompt(key: str) -> str:
    """Get a DeepSearch user prompt template by key."""
    if key not in USER_PROMPTS:
        raise KeyError(
            f"Unknown DeepSearch user prompt key: {key!r}. "
            f"Available: {list(USER_PROMPTS.keys())}"
        )
    return USER_PROMPTS[key]


__all__ = [
    "USER_PROMPTS",
    "get_user_prompt",
]
