""":class:`PackageSearchConstraints` — package-name-aware anti-hack blocklist.

A subclass of :class:`SearchConstraints` that tightens the bash
blocklist with *package-name-targeted* patterns on top of the
repo-URL patterns the base class already generates.

Motivation

Benchmarks that ask the agent to re-implement an existing Python
package (NL2Repo, DeNovoSWE doc2repo, etc.) are vulnerable to the
agent simply fetching the real upstream package and copying its
source — the default scoring (run golden tests against this
"implementation") then reports a trivially perfect result.  The base
:class:`SearchConstraints` only catches GitHub/GitLab URLs and
``git clone`` / ``git fetch`` aimed at the target repo; it does
nothing about the PyPI ecosystem (``pip install``, ``pip download``,
``uv add``, ``poetry add``, ``conda install`` ...), downloader commands
that fetch the target package/source, or about name variants
(``My-Pkg`` vs ``my_pkg`` vs ``MY-PKG``).

This class plugs into exactly the same integration point the base
class uses — :meth:`get_bash_blocklist_patterns` is concatenated into
``ExecuteBashTool._blocklist`` by :class:`SearchSWEAgent.__init__` —
so tasks opt in by simply returning an instance of this class from
:meth:`Task.get_search_constraints`.

Regex dialect notes

``ExecuteBashTool`` compiles patterns with :func:`re.compile` (no
``re.IGNORECASE`` flag) and matches with :meth:`re.Pattern.match`
(anchored at start).  Every pattern therefore:

* begins with the inline ``(?i)`` flag — Python 3.12+ requires
  global flags at the very start of the expression;
* follows with ``.*`` so it matches anywhere in the command string;
* uses :func:`re.escape` on literal identifiers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from aweagent.core.tool.search.constraints import SearchConstraints

logger = logging.getLogger(__name__)


# ── Name normalisation ──────────────────────────────────────────────────────


def _extract_identifiers(raw: str) -> list[str]:
    """Extract every plausible identifier from a raw package-name string.

    Handles the three shapes seen across NL2Repo / DeNovoSWE data:

    * plain name, e.g. ``"cerberus"`` or ``"PyAutoGUI"``;
    * PyPI name with separators, e.g. ``"typing-extensions"``;
    * VCS URL, e.g. ``"git+https://github.com/wting/autojump"`` or
      ``"https://github.com/hamuchiwa/AutoRCCar.git"``.

    Returns a list that contains the short repo name and — for VCS
    URLs — the ``owner/repo`` slug.  Empty and None inputs yield an
    empty list.
    """
    out: list[str] = []
    raw = (raw or "").strip()
    if not raw:
        return out

    stripped = raw[4:] if raw.lower().startswith("git+") else raw

    m = re.search(
        r"(?:github|gitlab|bitbucket|codeberg)\.(?:com|org)/"
        r"([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+?)(?:\.git)?/?(?:$|[?#])",
        stripped,
    )
    if m:
        owner, repo = m.group(1), m.group(2)
        out.append(repo)
        out.append(f"{owner}/{repo}")
        return out

    # Bare ``owner/repo`` slug without a hosting domain — this is the
    # shape the ``repo`` field on ``Instance`` usually takes.  Without
    # this branch the plain-name regex below would stop at the slash
    # and we'd lose the repo name entirely.
    m = re.match(r"^([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)$", stripped)
    if m:
        owner, name = m.group(1), m.group(2)
        out.append(name)
        out.append(f"{owner}/{name}")
        return out

    # Plain name (possibly with PyPI extras like ``pkg[extra]``).
    m = re.match(r"([A-Za-z0-9_.\-]+)", stripped)
    if m:
        out.append(m.group(1))
    else:
        out.append(stripped)
    return out


def _name_variants(name: str) -> set[str]:
    """Fan ``name`` out across ``-``/``_``/``.`` rewrites.

    Case-insensitivity is handled via the ``(?i)`` inline flag at the
    regex level so we do **not** also expand upper/lower copies — that
    would multiply the pattern count without adding coverage.
    """
    s = (name or "").strip()
    if not s:
        return set()
    out = {
        s,
        s.replace("-", "_"),
        s.replace("_", "-"),
        s.replace(".", "_"),
        s.replace(".", "-"),
    }
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", s)
    if camel_split != s:
        out.add(camel_split)
        out.add(camel_split.replace("-", "_"))
    # Double-substitution picks up multi-separator forms like
    # ``foo.bar-baz`` → ``foo_bar_baz`` in one round.
    out |= {v.replace("-", "_") for v in list(out)}
    out |= {v.replace("_", "-") for v in list(out)}
    return {v for v in out if v and not any(c.isspace() for c in v)}


def _build_package_variants(
    *seed_strings: str | None,
    min_length: int = 3,
) -> list[str]:
    """Deduped variant set seeded from every non-empty input.

    Short tokens (``<3`` chars by default) are dropped to avoid the
    catastrophic false-positive case of a package literally named
    ``re`` or ``os`` — no real NL2Repo/DeNovoSWE entry hits this, but
    it is cheap defence in depth.
    """
    seen: set[str] = set()
    for raw in seed_strings:
        if not raw:
            continue
        for ident in _extract_identifiers(raw):
            seen |= _name_variants(ident)
    return sorted(v for v in seen if len(v) >= min_length)


# ── Pattern building blocks ────────────────────────────────────────────────


def _q(word: str) -> str:
    """Regex-escape ``word`` wrapped in ``\\b`` boundaries."""
    return rf"\b{re.escape(word)}\b"


# Every known way to ask a package manager to pull/query a named dist.
#
# Version-suffix matchers use ``[\d.]*`` (not ``\d*``) so that versioned
# executables like ``pip3.10`` / ``python3.11`` are caught.  The python
# ``-m pip`` form matches ``pip\d*`` internally so ``python -m pip3``
# also hits.
_PACKAGE_MANAGERS_WITH_NAME = [
    # pip variants (pip, pip3, pip3.10, /usr/bin/pip, sudo pip, env-prefixed pip, etc.)
    # ``show`` / ``list`` / ``inspect`` / ``debug`` are introspection commands
    # that leak the installed file path (and from there the source) — block
    # them too when paired with the target package name.  The user-prompt
    # ANTI_CHEAT block already calls them out as forbidden; the runtime
    # regex enforces it.
    r"pip[\d.]*\s+(?:install|download|search|index\s+versions|wheel|fetch|show|list|inspect|debug)",
    r"python[\d.]*\s+-m\s+pip\d*\s+(?:install|download|search|index\s+versions|wheel|fetch|show|list|inspect|debug)",
    # pipx
    r"pipx\s+(?:install|run|fetch|inject|list|inspect)",
    # pipenv
    r"pipenv\s+(?:install|update|run\s+pip)",
    # uv (+ uv pip + uvx shorthand)
    r"uv\s+(?:pip\s+)?(?:install|download|add|tool\s+install|show|inspect)",
    r"uvx\b",
    # poetry
    r"poetry\s+(?:add|fetch|update|show|inspect)",
    # conda family
    r"(?:conda|mamba|micromamba)\s+(?:install|create|update|search|inspect|list)",
    # setuptools / pdm / hatch / legacy
    r"easy_install[\d.]*",
    r"pdm\s+(?:add|update|show|inspect)",
    r"hatch\s+run\s+pip",
]

_DOWNLOADERS = r"(?:curl|wget|aria2c?|http|httpie|lwp-download|fetch)"

_SHELL_COMMAND_PREFIX = (
    r"(?:^|[;&|]\s*)"
    r"(?:(?:sudo|command|time)\s+|timeout\s+\S+\s+|"
    r"env\s+(?:\S+=\S+\s+)*)*"
)


def _package_arg(word: str) -> str:
    """Match a package requirement argument, not a path segment."""
    escaped = re.escape(word)
    return (
        rf"(?<![/A-Za-z0-9_.-]){escaped}"
        r"(?:\[[^\s;&|]*\])?"
        r"(?:(?:==|~=|!=|<=|>=|<|>)[^\s;&|]+)?"
        r"(?=$|[\s;&|])"
    )


# ── Public API ──────────────────────────────────────────────────────────────


@dataclass
class PackageSearchConstraints(SearchConstraints):
    """:class:`SearchConstraints` with package-name-targeted bash blocks.

    Construct via :meth:`from_package_name` (single-name convenience)
    or :meth:`from_identifiers` (multi-name, preferred when the caller
    has extras such as ``import_names`` and a repo slug that differs
    from the PyPI name).

    Example::

        constraints = PackageSearchConstraints.from_identifiers(
            package_name="typing-extensions",
            repo="python/typing_extensions",
            extra_identifiers=["typing_extensions"],
        )
    """

    _package_variants: list[str] = field(default_factory=list, repr=False)

    # ── Constructors ──────────────────────────────────────────────────

    @classmethod
    def from_package_name(
        cls,
        package_name: str,
        instance_id: str | None = None,
        *,
        extra_identifiers: list[str] | None = None,
        repo: str | None = None,
    ) -> PackageSearchConstraints:
        """Build constraints from a primary package name plus optional extras.

        ``package_name`` is the authoritative PyPI/VCS form (what
        upstream pipeline stages normally store in ``pypi_name``).
        ``instance_id`` is used as a secondary identifier when the
        CSV/JSONL row doesn't carry a separate package name.
        ``extra_identifiers`` lets the caller feed in ``import_names``
        or additional candidate names discovered by the extractor.
        ``repo`` is the ``owner/name`` slug used to seed the base
        class's URL/git patterns — pass it explicitly when the repo
        name differs from the distribution name.
        """
        return cls.from_identifiers(
            [package_name, instance_id, *(extra_identifiers or [])],
            repo=repo,
        )

    @classmethod
    def from_identifiers(
        cls,
        identifiers: list[str | None],
        *,
        repo: str | None = None,
    ) -> PackageSearchConstraints:
        """Build constraints from an arbitrary list of seed identifiers.

        ``identifiers`` is the preferred entry point when the caller
        has a mix of sources (PyPI name, repo slug, module name, raw
        VCS URL) — each non-empty string is fanned out into its
        variant set and the union is stored on the instance.  ``repo``
        is *also* folded into the variant set so that rows lacking a
        separate ``pypi_name`` still block ``pip install <repo-name>``.
        """
        # Fold repo into the variant seed as well — otherwise a row
        # carrying only a repo slug (``someone/cool-pkg``) would fall
        # back to the base class's URL patterns alone and miss
        # ``pip install cool-pkg``.
        variants = _build_package_variants(*identifiers, repo)

        # Seed base-class URL/git patterns from the best available slug.
        base_seed = repo or ""
        if not base_seed:
            for raw in identifiers:
                parsed = _extract_identifiers(raw or "")
                if parsed:
                    # ``owner/repo`` if the input was a VCS URL,
                    # otherwise the plain name.
                    base_seed = parsed[-1]
                    break
        base = SearchConstraints.from_repo(base_seed) if base_seed else SearchConstraints()

        return cls(
            blocked_patterns=dict(base.blocked_patterns),
            _repo_owner=base._repo_owner,
            _repo_name=base._repo_name,
            _package_variants=variants,
        )

    # ── Overrides ──────────────────────────────────────────────────────

    def get_bash_blocklist_patterns(self) -> list[str]:
        """Return base repo blocks + package-name anti-hack patterns.

        Ordering (high → low trust): base class patterns, then per-variant:

        1. Package-manager install/download/add for every known PM, but
           only when the target package name appears as a package
           argument in that same command segment.
        2. ``git clone`` / ``git submodule add`` mentioning the name.
        3. Network downloaders mentioning the name.

        Generic repo-introspection and non-search external-fetch blocks
        (``git log --all``, bare ``git fetch``, non-search ``git clone``,
        etc.) are still supplied by :class:`SearchSWEAgent`; this method
        only avoids over-blocking legitimate local package work.
        """
        patterns: list[str] = list(super().get_bash_blocklist_patterns())

        for v in self._package_variants:
            qv = _q(v)
            pkg_arg = _package_arg(v)
            for pm in _PACKAGE_MANAGERS_WITH_NAME:
                patterns.append(
                    rf"(?i).*{_SHELL_COMMAND_PREFIX}{pm}\b[^\n;&|]*{pkg_arg}.*"
                )
            patterns.append(
                rf"(?i).*{_SHELL_COMMAND_PREFIX}git\s+clone\b[^\n;&|]*{qv}.*"
            )
            patterns.append(
                rf"(?i).*{_SHELL_COMMAND_PREFIX}git\s+submodule\s+add\b[^\n;&|]*{qv}.*"
            )
            patterns.append(
                rf"(?i).*{_SHELL_COMMAND_PREFIX}{_DOWNLOADERS}\b[^\n;&|]*{qv}.*"
            )

        return patterns
