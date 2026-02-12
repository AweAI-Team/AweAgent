"""Prompt templates for SWE-Bench tasks.

This module provides backward-compatible access to SWE-Bench prompts.
The canonical template now lives in the scaffold prompt system
(:mod:`awe_agent.scaffold.search_swe.prompts.user`).
"""

from __future__ import annotations

from awe_agent.scaffold.search_swe.prompts.user import SWE_BENCH_PROMPT


def get_swe_bench_prompt(
    problem_statement: str,
    workspace_dir: str = "/testbed",
    language: str = "python",
    task_guidance: str | None = None,
) -> str:
    """Build the user prompt for a SWE-Bench instance.

    .. deprecated::
        Prefer using the scaffold prompt system directly via
        :func:`~awe_agent.scaffold.search_swe.prompts.config.resolve_prompt_keys`.
    """
    return SWE_BENCH_PROMPT.format(
        workspace_dir=workspace_dir,
        problem_statement=problem_statement,
        base_commit="",
        workspace_tree="",
        installed_packages="",
        REPO_DOCUMENT="",
    )
