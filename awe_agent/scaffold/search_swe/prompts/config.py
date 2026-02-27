"""Prompt routing configuration.

Maps (dataset_id, task_type, search_mode) to (system_prompt_key, user_prompt_key).
This is the single source of truth for all prompt selection logic.

    | Task       | Search | System Key       | User Key           |
    |------------|--------|------------------|--------------------|
    | Doc2Repo   | 0      | beyondswe        | doc2repo           |
    | Doc2Repo   | 1      | search_beyondswe | search_doc2repo    |
    | CrossRepo  | 0      | beyondswe        | crossrepo          |
    | CrossRepo  | 1      | search_beyondswe | search_crossrepo   |
    | DepMigrate | 0      | beyondswe        | depmigrate         |
    | DepMigrate | 1      | search_beyondswe | search_depmigrate  |
    | DomainFix  | 0      | beyondswe        | domainfix          |
    | DomainFix  | 1      | search_domainfix | search_domainfix   |
    | ScaleSWE   | 0      | openhands        | scaleswe           |
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
    ("beyond_swe", "doc2repo", False):     ("beyondswe",        "doc2repo"),
    ("beyond_swe", "doc2repo", True):      ("search_beyondswe", "search_doc2repo"),
    ("beyond_swe", "crossrepo", False):    ("beyondswe",        "crossrepo"),
    ("beyond_swe", "crossrepo", True):     ("search_beyondswe", "search_crossrepo"),
    ("beyond_swe", "depmigrate", False):   ("beyondswe",        "depmigrate"),
    ("beyond_swe", "depmigrate", True):    ("search_beyondswe", "search_depmigrate"),
    ("beyond_swe", "domainfix", False):    ("beyondswe",        "domainfix"),
    ("beyond_swe", "domainfix", True):     ("search_domainfix", "search_domainfix"),

    # ── ScaleSWE ─────────────────────────────────────────────────────
    ("scale_swe", None, False):            ("openhands", "scaleswe"),
}

# Default fallback when no exact route matches
_DEFAULT_ROUTE: tuple[str, str] = ("beyondswe", "domainfix")


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
