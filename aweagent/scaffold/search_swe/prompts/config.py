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
    | NL2Repo    | 0      | beyondswe        | nl2repo            |
    | SWE-bench-Pro | 0   | openhands        | swebenchpro        |
    | DeNovoSWE  | 0      | beyondswe        | denovoswe_doc2repo |

An unmatched (dataset_id, task_type, search) raises rather than falling back
to a default prompt: a wrong-but-plausible prompt silently tanks a run, so we
fail fast at resolution instead.
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

    # ── NL2Repo ──────────────────────────────────────────────────────
    # NL2RepoBench has a single from-scratch repo-generation task type.
    # Reuses the BeyondSWE non-search system prompt because both are
    # OpenHands-style coding agents working on a clean Linux workspace;
    # the user prompt is the verbatim NL2RepoBench instruction string.
    ("nl2repo", "nl2repo", False):         ("beyondswe", "nl2repo"),
    ("nl2repo", None, False):              ("beyondswe", "nl2repo"),

    # ── SWE-bench-Pro ────────────────────────────────────────────────
    ("swe_bench_pro", None, False):        ("openhands", "swebenchpro"),

    # ── DeNovoSWE ────────────────────────────────────────────────────
    # doc2repo from a natural-language spec with source-cleaned images.
    # Reuses the BeyondSWE non-search system prompt (same agent persona);
    # only the user prompt is denovo-specific (the ``_v2`` variant is
    # selected in get_prompt). DeNovoSWE runs search-off only.
    ("denovo_swe", "doc2repo", False):     ("beyondswe", "denovoswe_doc2repo"),
}


def resolve_prompt_keys(
    dataset_id: str,
    task_type: str | None,
    search: bool,
) -> tuple[str, str]:
    """Resolve (system_key, user_key) for the given context.

    Lookup order:
    1. Exact match: (dataset_id, task_type, search)
    2. Wildcard:     (dataset_id, None, search)

    Raises KeyError when neither matches. There is deliberately no default
    fallback — a silently-wrong prompt is worse than a loud failure, and every
    supported (dataset, task_type) is listed in ``PROMPT_ROUTES`` above.
    """
    # Exact match
    key = (dataset_id, task_type, search)
    if key in PROMPT_ROUTES:
        return PROMPT_ROUTES[key]

    # Wildcard on task_type
    wildcard_key = (dataset_id, None, search)
    if wildcard_key in PROMPT_ROUTES:
        return PROMPT_ROUTES[wildcard_key]

    raise KeyError(
        f"No prompt route for (dataset_id={dataset_id!r}, "
        f"task_type={task_type!r}, search={search}). "
        f"Add it to PROMPT_ROUTES or set agent.system_prompt_file to override."
    )


def resolve_from_task_info(
    task_info: dict[str, Any],
    search: bool,
) -> tuple[str, str]:
    """Convenience wrapper that extracts fields from task_info dict."""
    dataset_id = task_info.get("dataset_id", "")
    task_type = task_info.get("task_type")
    return resolve_prompt_keys(dataset_id, task_type, search)
