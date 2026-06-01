"""Prompt routing configuration for DeepSearch."""

from __future__ import annotations

from typing import Any

PromptRoute = tuple[str, str, str]

# Key: (dataset_id, task_type | None). Value: (system_key, user_key, final_answer_key).
PROMPT_ROUTES: dict[tuple[str, str | None], PromptRoute] = {
    ("browsecomp", None): ("browsecomp", "raw", "default"),
}

_DEFAULT_ROUTE: PromptRoute = ("default", "raw", "default")


def resolve_prompt_keys(
    dataset_id: str,
    task_type: str | None,
) -> PromptRoute:
    """Resolve (system_key, user_key, final_answer_key) for DeepSearch."""
    exact_key = (dataset_id, task_type)
    if exact_key in PROMPT_ROUTES:
        return PROMPT_ROUTES[exact_key]

    wildcard_key = (dataset_id, None)
    if wildcard_key in PROMPT_ROUTES:
        return PROMPT_ROUTES[wildcard_key]

    return _DEFAULT_ROUTE


def resolve_from_task_info(task_info: dict[str, Any]) -> PromptRoute:
    """Resolve DeepSearch prompt keys from AgentContext.task_info."""
    return resolve_prompt_keys(
        dataset_id=str(task_info.get("dataset_id", "")),
        task_type=task_info.get("task_type"),
    )


__all__ = [
    "PROMPT_ROUTES",
    "PromptRoute",
    "resolve_from_task_info",
    "resolve_prompt_keys",
]
