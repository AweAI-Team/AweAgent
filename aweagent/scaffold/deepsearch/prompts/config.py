"""Prompt routing configuration for DeepSearch."""

from __future__ import annotations

from typing import Any

PromptRoute = tuple[str, str, str]

# Key: task_type. Value: (system_key, user_key, final_answer_key).
PROMPT_ROUTES: dict[str, PromptRoute] = {
    "browsecomp": ("browsecomp", "raw", "default"),
}

_DEFAULT_ROUTE: PromptRoute = ("default", "raw", "default")


def resolve_prompt_keys(
    task_type: str | None,
) -> PromptRoute:
    """Resolve (system_key, user_key, final_answer_key) for DeepSearch."""
    if task_type in PROMPT_ROUTES:
        return PROMPT_ROUTES[task_type]
    return _DEFAULT_ROUTE


def resolve_from_task_info(task_info: dict[str, Any]) -> PromptRoute:
    """Resolve DeepSearch prompt keys from AgentContext.task_info."""
    return resolve_prompt_keys(
        task_type=task_info.get("task_type"),
    )


__all__ = [
    "PROMPT_ROUTES",
    "PromptRoute",
    "resolve_from_task_info",
    "resolve_prompt_keys",
]
