"""Prompt templates for BeyondSWE task types.

This module provides backward-compatible access to BeyondSWE prompts.
The canonical templates now live in the scaffold prompt system
(:mod:`awe_agent.scaffold.search_swe.prompts.user`).  This module
re-exports them and provides the legacy ``get_beyond_swe_prompt()``
helper.
"""

from __future__ import annotations

from awe_agent.scaffold.search_swe.prompts.user import (
    CROSS_REPO_PROMPT,
    DOC2REPO_PROMPT,
    DOMAIN_PROMPT,
    REFACTOR_PROMPT,
)

# ── Registry ─────────────────────────────────────────────────────────────────

TASK_TYPE_PROMPTS: dict[str, str] = {
    "doc2repo": DOC2REPO_PROMPT,
    "cross-repo": CROSS_REPO_PROMPT,
    "cross_repo": CROSS_REPO_PROMPT,
    "refactor": REFACTOR_PROMPT,
    "domain": DOMAIN_PROMPT,
}


def get_beyond_swe_prompt(
    task_type: str,
    workspace_dir: str = "/workspace",
    problem_statement: str = "",
    base_commit: str = "",
    repo_document: str = "",
) -> str:
    """Build the user prompt for a BeyondSWE instance.

    .. deprecated::
        Prefer using the scaffold prompt system directly via
        :func:`~awe_agent.scaffold.search_swe.prompts.config.resolve_prompt_keys`.
    """
    template = TASK_TYPE_PROMPTS.get(task_type)
    if template is None:
        raise ValueError(
            f"Unknown BeyondSWE task type: {task_type!r}. "
            f"Expected one of: {list(TASK_TYPE_PROMPTS.keys())}"
        )
    return template.format(
        workspace_dir=workspace_dir,
        problem_statement=problem_statement,
        base_commit=base_commit,
        workspace_tree="",
        installed_packages="",
        REPO_DOCUMENT=repo_document,
    )
