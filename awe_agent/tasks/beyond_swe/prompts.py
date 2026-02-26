"""Prompt templates for BeyondSWE task types.

Provides ``get_beyond_swe_prompt()`` convenience helper that formats
BeyondSWE user prompt templates with instance-specific values.
"""

from __future__ import annotations

from awe_agent.tasks.beyond_swe.prompt.user import (
    CROSSREPO_PROMPT,
    DEPMIGRATE_PROMPT,
    DOC2REPO_PROMPT,
    DOMAINFIX_PROMPT,
)

# ── Registry ─────────────────────────────────────────────────────────────────

TASK_TYPE_PROMPTS: dict[str, str] = {
    "doc2repo": DOC2REPO_PROMPT,
    "crossrepo": CROSSREPO_PROMPT,
    "depmigrate": DEPMIGRATE_PROMPT,
    "domainfix": DOMAINFIX_PROMPT,
}


def get_beyond_swe_prompt(
    task_type: str,
    workspace_dir: str = "/workspace",
    problem_statement: str = "",
    base_commit: str = "",
    repo_document: str = "",
) -> str:
    """Build the user prompt for a BeyondSWE instance."""
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
