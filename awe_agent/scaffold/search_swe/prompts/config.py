"""Prompt routing configuration.

Maps (dataset_id, task_type, search_mode) to (system_prompt_key, user_prompt_key).
This is the single source of truth for all prompt selection logic.

The mapping is derived from empirical prompt tuning — each combination has been
validated on its respective benchmark. When adding new datasets or task types,
add a new entry here rather than hard-coding prompt selection elsewhere.
"""

from __future__ import annotations

from typing import Any

# ── Route table ──────────────────────────────────────────────────────────────
#
# Key:   (dataset_id, task_type | None, search_enabled)
# Value: (system_prompt_key, user_prompt_key)
#
# task_type=None means the route applies regardless of task_type.
# More specific routes take priority over wildcard routes.

PROMPT_ROUTES: dict[tuple[str, str | None, bool], tuple[str, str]] = {
    # ── BeyondSWE ────────────────────────────────────────────────────
    ("beyond_swe", "doc2repo", False):     ("base",          "doc2repo"),
    ("beyond_swe", "doc2repo", True):      ("search",        "search_doc2repo"),
    ("beyond_swe", "cross-repo", False):   ("base",          "cross_repo"),
    ("beyond_swe", "cross-repo", True):    ("search",        "search_cross_repo"),
    ("beyond_swe", "refactor", False):     ("base",          "refactor"),
    ("beyond_swe", "refactor", True):      ("search",        "search_refactor"),
    ("beyond_swe", "domain", False):       ("base",          "domain"),
    ("beyond_swe", "domain", True):        ("search_domain", "search_domain"),
}

# Default fallback when no exact route matches
_DEFAULT_ROUTE: tuple[str, str] = ("base", "domain")


def resolve_prompt_keys(
    dataset_id: str,
    task_type: str | None,
    search: bool,
) -> tuple[str, str]:
    """Resolve (system_key, user_key) for the given context.

    Lookup order:
    1. Exact match: (dataset_id, task_type, search)
    2. Wildcard:     (dataset_id, None, search)
    3. Default fallback
    """
    # Exact match
    key = (dataset_id, task_type, search)
    if key in PROMPT_ROUTES:
        return PROMPT_ROUTES[key]

    # Wildcard on task_type
    wildcard_key = (dataset_id, None, search)
    if wildcard_key in PROMPT_ROUTES:
        return PROMPT_ROUTES[wildcard_key]

    return _DEFAULT_ROUTE


def resolve_from_task_info(
    task_info: dict[str, Any],
    search: bool,
) -> tuple[str, str]:
    """Convenience wrapper that extracts fields from task_info dict."""
    dataset_id = task_info.get("dataset_id", "")
    task_type = task_info.get("task_type")
    return resolve_prompt_keys(dataset_id, task_type, search)
