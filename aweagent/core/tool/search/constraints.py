"""SearchConstraints — filtering and URL blocking for anti-hack evaluation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchConstraints:
    """Constraints to prevent information leakage during evaluation.

    Encapsulates both search-result filtering and URL blocking.
    Patterns are matched with ``re.match`` (anchored at start) and
    ``re.IGNORECASE``.

    Example::

        constraints = SearchConstraints.from_repo("django/django")
        filtered, count = constraints.filter_search_results(results)
    """

    # Regex patterns to filter search results, keyed by field name.
    # e.g. {"url": [r'.*github\\.com/[^/]+/django(/|$|\\?).*'], "title": [...]}
    blocked_patterns: dict[str, list[str]] = field(default_factory=dict)

    # ── Repo metadata (set by from_repo) ───────────────────────────────
    _repo_owner: str | None = field(default=None, repr=False)
    _repo_name: str | None = field(default=None, repr=False)

    # ── Constructors ───────────────────────────────────────────────────

    @classmethod
    def from_repo(cls, repo: str) -> SearchConstraints:
        """Build standard anti-hack patterns from ``'owner/repo'`` or ``'repo'`` string.

        Generates URL patterns blocking GitHub, GitLab, and raw content URLs
        for the target repo while avoiding false positives (e.g. ``repo-extension``).
        """
        parts = repo.strip().split("/")
        if len(parts) >= 2:
            owner, repo_name = parts[0], parts[1]
        else:
            owner, repo_name = None, parts[0]

        escaped_name = re.escape(repo_name)

        # Build URL block patterns — match the repo slug with boundary
        # to avoid false positives like "django-extensions"
        url_patterns: list[str] = []

        if owner:
            escaped_owner = re.escape(owner)
            # GitHub: github.com/owner/repo
            url_patterns.append(
                rf".*github\.com/{escaped_owner}/{escaped_name}(/|$|\?).*"
            )
            # GitLab: gitlab.com/owner/repo
            url_patterns.append(
                rf".*gitlab\.com/{escaped_owner}/{escaped_name}(/|$|\?).*"
            )
        else:
            # Without owner, match any owner
            url_patterns.append(
                rf".*github\.com/[^/]+/{escaped_name}(/|$|\?).*"
            )
            url_patterns.append(
                rf".*gitlab\.com/[^/]+/{escaped_name}(/|$|\?).*"
            )

        # Raw content URLs
        url_patterns.append(
            rf".*raw\.githubusercontent\.com/.*/{escaped_name}/.*"
        )

        return cls(
            blocked_patterns={"url": url_patterns},
            _repo_owner=owner,
            _repo_name=repo_name,
        )

    # ── URL blocking ───────────────────────────────────────────────────

    def is_url_blocked(self, url: str) -> bool:
        """Check if a single URL matches any blocked ``'url'`` pattern."""
        for pattern in self.blocked_patterns.get("url", []):
            try:
                if re.match(pattern, url, re.IGNORECASE):
                    return True
            except re.error:
                logger.warning("Invalid regex in blocked_patterns['url']: %s", pattern)
        return False

    # ── Result filtering ───────────────────────────────────────────────

    def filter_search_results(
        self, results: list[dict],
    ) -> tuple[list[dict], int]:
        """Filter blocked items from a list of search result dicts.

        Each result dict is checked against all blocked_patterns for matching
        field names. A result is blocked if *any* field matches *any* pattern
        for that field.

        Returns:
            A tuple of ``(filtered_results, num_filtered)``.
        """
        if not self.blocked_patterns:
            return results, 0

        filtered: list[dict] = []
        num_filtered = 0

        for item in results:
            blocked = False
            for field_name, patterns in self.blocked_patterns.items():
                value = item.get(field_name, "")
                if not isinstance(value, str):
                    continue
                for pattern in patterns:
                    try:
                        if re.match(pattern, value, re.IGNORECASE):
                            blocked = True
                            break
                    except re.error:
                        logger.warning(
                            "Invalid regex in blocked_patterns[%r]: %s",
                            field_name,
                            pattern,
                        )
                if blocked:
                    break

            if blocked:
                num_filtered += 1
            else:
                filtered.append(item)

        return filtered, num_filtered

    # ── Bash blocklist patterns ────────────────────────────────────────

    def get_bash_blocklist_patterns(self) -> list[str]:
        """Generate additional bash command blocklist patterns for the repo.

        Returns patterns for ``git clone/fetch``, ``github.io``,
        ``api.github.com``, and ``raw.githubusercontent`` URLs targeting
        the repo. Only meaningful when constructed via :meth:`from_repo`.
        """
        if not self._repo_name:
            return []

        escaped = re.escape(self._repo_name)
        patterns = [
            rf".*git\s+clone\s+.*{escaped}.*",
            rf".*git\s+fetch\s+.*{escaped}.*",
        ]

        if self._repo_owner:
            escaped_owner = re.escape(self._repo_owner)
            patterns.extend([
                rf".*github\.io/{escaped_owner}/{escaped}.*",
                rf".*api\.github\.com/repos/{escaped_owner}/{escaped}.*",
                rf".*raw\.githubusercontent\.com/{escaped_owner}/{escaped}.*",
            ])
        else:
            patterns.extend([
                rf".*github\.io/[^/]+/{escaped}.*",
                rf".*api\.github\.com/repos/[^/]+/{escaped}.*",
                rf".*raw\.githubusercontent\.com/[^/]+/{escaped}.*",
            ])

        return patterns

    # ── Merging ────────────────────────────────────────────────────────

    def merge(self, other: SearchConstraints) -> SearchConstraints:
        """Merge two constraint sets (union of all patterns).

        Returns a **new** :class:`SearchConstraints` without mutating either
        operand.
        """
        merged: dict[str, list[str]] = {}
        all_keys = set(self.blocked_patterns) | set(other.blocked_patterns)
        for key in all_keys:
            merged[key] = list(
                dict.fromkeys(  # deduplicate while preserving order
                    self.blocked_patterns.get(key, [])
                    + other.blocked_patterns.get(key, [])
                )
            )
        return SearchConstraints(
            blocked_patterns=merged,
            _repo_owner=self._repo_owner or other._repo_owner,
            _repo_name=self._repo_name or other._repo_name,
        )
