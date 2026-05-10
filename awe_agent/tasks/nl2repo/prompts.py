"""Prompt template helpers for the NL2Repo task.

Provides ``get_nl2repo_prompt()`` convenience helper that returns the
exact upstream NL2RepoBench user prompt string.  The original launcher
uses a single hard-coded prompt with no placeholders, so this function
just returns the constant — kept as a function for symmetry with
``awe_agent/tasks/beyond_swe/prompts.py::get_beyond_swe_prompt``.
"""

from __future__ import annotations

from awe_agent.tasks.nl2repo.prompt.user import NL2REPO_PROMPT

# ── Registry ─────────────────────────────────────────────────────────────────

TASK_TYPE_PROMPTS: dict[str, str] = {
    "nl2repo": NL2REPO_PROMPT,
}


def get_nl2repo_prompt() -> str:
    """Return the NL2Repo user prompt verbatim from NL2RepoBench."""
    return NL2REPO_PROMPT
