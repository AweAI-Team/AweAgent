"""NL2Repo-specific anti-hack :class:`SearchConstraints`.

The NL2Repo agent's job is to *re-implement* the target package from a
natural-language spec (``start.md``).  A common failure mode we've
observed is the agent downloading the real upstream package and
copying its source verbatim — that produces a trivially "perfect"
score against the golden tests.

This module builds a strict, per-instance regex blocklist that is
plugged into the existing :class:`~awe_agent.core.tool.search.constraints.SearchConstraints`
machinery (same path that realswe / beyondswe use).  The blocklist is
passed through :meth:`SearchSWEAgent.__init__` → ``ExecuteBashTool``,
which regex-matches every agent bash command **before** it runs.

Scope (what we try to stop)

- Package managers fetching the target package: ``pip`` / ``pip3`` /
  ``pipx`` / ``python -m pip`` / ``uv`` / ``poetry`` / ``conda`` /
  ``mamba`` — ``install`` / ``download`` / ``add`` / ``wheel`` / ``show``.
- ``git clone`` / ``git submodule add`` pointing at the target repo
  or any URL containing the package identifier.
- ``curl`` / ``wget`` / ``aria2c`` / ``http`` / ``httpie`` hitting
  PyPI-family distribution hosts (``pypi.org``,
  ``files.pythonhosted.org``, ``pypi.python.org``, ``test.pypi.org``,
  ``pypistats.org``) or any URL that mentions the package identifier.
- Python introspection shortcuts like
  ``python -c 'import pkg; ...inspect.getsource...'`` or
  ``python -c 'import pkg; ...__file__...'`` used to read the
  installed upstream source.

All patterns begin with the inline ``(?i)`` flag (at the very start, as
required by Python 3.12+ regex grammar) because
``ExecuteBashTool._blocklist`` compiles with :func:`re.compile` **without**
``re.IGNORECASE``.  Immediately after the flag we put ``.*`` because the
tool uses :meth:`re.Pattern.match` (anchored at start), so the pattern
has to explicitly allow leading content.

Name variants

Package names in NL2RepoBench's CSV come in many forms:
``Cherry`` / ``PyAutoGUI`` / ``Box`` / ``jusText`` (mixed case),
``typing-extensions`` / ``google_images_download`` (hyphen ↔ underscore
PyPI vs module-name skew), and even
``git+https://github.com/wting/autojump`` (direct VCS URL).  We
enumerate every sensible variant so the agent can't slip by simply by
swapping one separator or casing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from awe_agent.core.tool.search.constraints import SearchConstraints

logger = logging.getLogger(__name__)


# ── Name normalisation ──────────────────────────────────────────────────────


def _extract_identifiers(raw: str) -> list[str]:
    """Pull every plausible identifier out of a ``package_name`` field.

    Handles the three shapes seen in ``nl2repo_data_info.csv``:

    * plain name, e.g. ``"cerberus"`` or ``"PyAutoGUI"``
    * PyPI name with separators, e.g. ``"typing-extensions"``
    * VCS URL, e.g. ``"git+https://github.com/wting/autojump"`` or
      ``"git+https://github.com/hamuchiwa/AutoRCCar.git"``.

    Returns a list that includes the short repo name plus the
    ``owner/repo`` slug where applicable, so downstream variant
    generation can fan them out.
    """
    out: list[str] = []
    raw = (raw or "").strip()
    if not raw:
        return out

    # Strip ``git+`` prefix
    stripped = raw[4:] if raw.lower().startswith("git+") else raw

    # VCS URL (github / gitlab / bitbucket / codeberg)
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

    # Fallback: plain name (possibly with PyPI extras like ``pkg[extra]``).
    # Trim the ``[extra]`` suffix so we generate variants for the real name.
    m = re.match(r"([A-Za-z0-9_.\-]+)", stripped)
    if m:
        out.append(m.group(1))
    else:
        out.append(stripped)
    return out


def _name_variants(name: str) -> set[str]:
    """Fan out ``name`` into its ``-``/``_``/``.`` rewrites.

    Case is handled separately via the ``(?i)`` inline flag at the
    regex level, so we don't explode the set with upper/lower copies.
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
    # Double-substitution so ``foo.bar-baz`` reaches ``foo_bar_baz`` etc.
    out |= {v.replace("-", "_") for v in list(out)}
    out |= {v.replace("_", "-") for v in list(out)}
    # Drop empty and anything with whitespace / characters we can't put in a
    # word-boundary regex without escaping (escape later).
    return {v for v in out if v and not any(c.isspace() for c in v)}


def _build_package_variants(package_name: str, instance_id: str | None) -> list[str]:
    """Return the sorted, deduplicated set of identifiers to block.

    We seed from both ``package_name`` (authoritative PyPI/VCS form in
    the CSV) and ``instance_id`` (the directory name, used as a
    fallback when ``package_name`` is missing) because they sometimes
    differ — e.g. ``instance_id="google-images-download"`` vs
    ``package_name="google_images_download"``.
    """
    seen: set[str] = set()
    for raw in (package_name, instance_id):
        for ident in _extract_identifiers(raw or ""):
            seen |= _name_variants(ident)
    # Drop very short tokens that would generate catastrophic false
    # positives (e.g. a package literally named ``re`` or ``os``).  The
    # benchmark has none at the time of writing, but this is defence in
    # depth — users will notice if we over-block.
    return sorted(v for v in seen if len(v) >= 3)


# ── Pattern helpers ─────────────────────────────────────────────────────────
#
# ``ExecuteBashTool._blocklist`` is ``[re.compile(p) for p in ...]`` with no
# ``re.IGNORECASE`` flag, and matched with ``.match()``.  So every pattern
# must:
#
#   * begin with the inline ``(?i)`` flag (Python 3.12+ requires global
#     flags at the very start of the expression — anywhere else raises
#     ``re.PatternError``)
#   * follow the flag with ``.*`` so it matches anywhere in the command
#     (``.match()`` is anchored at the start of the subject)
#   * use ``re.escape`` on literals that contain regex metacharacters


def _q(word: str) -> str:
    """Regex-escape a literal and wrap it with ``\\b`` word boundaries.

    ``re.escape`` is the right primitive for literals; ``\\b`` with
    Python's default ``re`` treats ``-`` as a non-word character, so
    ``\\bmy-pkg\\b`` behaves sensibly (boundaries at the outer edges,
    not around the internal dash).
    """
    return rf"\b{re.escape(word)}\b"


_PACKAGE_MANAGERS_WITH_NAME = [
    # pip / pip3 / python -m pip
    r"pip\d*\s+(?:install|download|show|index\s+versions|wheel|fetch)",
    r"python\d?\s+-m\s+pip\s+(?:install|download|show|index\s+versions|wheel|fetch)",
    # pipx
    r"pipx\s+(?:install|run|fetch)",
    # uv (+ uv pip)
    r"uv\s+(?:pip\s+)?(?:install|download|add|sync|tool\s+install|export)",
    # poetry
    r"poetry\s+(?:add|install|fetch)",
    # conda / mamba / micromamba
    r"(?:conda|mamba|micromamba)\s+(?:install|create|update|search)",
    # easy_install / setuptools
    r"easy_install\d*",
    # pdm
    r"pdm\s+(?:add|install)",
    # hatch
    r"hatch\s+(?:project|build|env\s+install)",
]


# Distribution hosts / mirrors — we treat any agent-initiated fetch of these
# as suspect.  Not per-package; even a bare ``curl pypi.org/simple/...`` is
# a smell since the agent has no legitimate reason to browse PyPI.
_PYPI_HOSTS = [
    r"pypi\.org",
    r"pypi\.python\.org",
    r"test\.pypi\.org",
    r"files\.pythonhosted\.org",
    r"pythonhosted\.org",
    r"pypistats\.org",
    r"conda\.anaconda\.org",
    r"anaconda\.org",
    r"repo\.anaconda\.com",
]

_DOWNLOADERS = r"(?:curl|wget|aria2c?|http|httpie|lwp-download|fetch)"


# ── Public API ──────────────────────────────────────────────────────────────


@dataclass
class NL2RepoSearchConstraints(SearchConstraints):
    """:class:`SearchConstraints` with package-name-targeted bash blocks.

    Keeps the base class's URL/search filtering behaviour intact (so the
    agent's search and link-reader tools still respect repo-URL blocks)
    and **extends** :meth:`get_bash_blocklist_patterns` with the
    anti-hack patterns built from the instance's package name and
    variants.
    """

    _package_variants: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_package_name(
        cls,
        package_name: str,
        instance_id: str | None = None,
    ) -> NL2RepoSearchConstraints:
        """Build anti-hack constraints for one NL2Repo instance.

        Seeds the base-class URL blocks from the package's repo slug
        (falling back to ``instance_id`` if ``package_name`` is a plain
        identifier), then stashes every name variant for use by
        :meth:`get_bash_blocklist_patterns`.
        """
        identifiers = _extract_identifiers(package_name or "")
        if not identifiers and instance_id:
            identifiers = _extract_identifiers(instance_id)

        # Use the first identifier (either ``owner/repo`` or plain name)
        # to seed the base class's repo-URL patterns.
        base_seed = identifiers[-1] if identifiers else (instance_id or "")
        base = SearchConstraints.from_repo(base_seed) if base_seed else SearchConstraints()

        variants = _build_package_variants(package_name, instance_id)
        return cls(
            blocked_patterns=dict(base.blocked_patterns),
            _repo_owner=base._repo_owner,
            _repo_name=base._repo_name,
            _package_variants=variants,
        )

    # ── Overrides ──────────────────────────────────────────────────────

    def get_bash_blocklist_patterns(self) -> list[str]:
        """Return base repo blocks + NL2Repo's anti-hack patterns.

        The list is composed of, in order:

        1. Whatever the base class produces (git clone/fetch for the
           target repo name, github.io / raw.githubusercontent URLs).
        2. Per-variant package-manager install / download patterns
           across ``pip``, ``pip3``, ``python -m pip``, ``pipx``,
           ``uv``, ``poetry``, ``conda``, ``mamba``, ``micromamba``,
           ``easy_install``, ``pdm``, ``hatch``.
        3. Per-variant ``git clone`` / ``git submodule add`` patterns.
        4. Per-variant downloader (``curl``/``wget``/...) when the URL
           contains the package identifier.
        5. Per-variant ``python -c`` introspection patterns
           (``inspect.getsource`` / ``__file__`` / ``importlib``).
        6. Global distribution-host blocks (PyPI / Anaconda / raw CDNs)
           for any downloader command.
        """
        patterns: list[str] = list(super().get_bash_blocklist_patterns())

        for v in self._package_variants:
            qv = _q(v)
            # 1 — Package managers: any "install/download/add" of this name.
            for pm in _PACKAGE_MANAGERS_WITH_NAME:
                patterns.append(rf"(?i).*{pm}.*{qv}.*")
            # 2 — git clone / submodule add referencing the name.
            patterns.append(rf"(?i).*git\s+clone\b.*{qv}.*")
            patterns.append(rf"(?i).*git\s+submodule\s+add\b.*{qv}.*")
            # 3 — URL downloaders referencing the name.
            patterns.append(rf"(?i).*{_DOWNLOADERS}\b.*{qv}.*")
            # 4 — Python introspection shortcuts:
            #     python -c '... import <v> ... inspect.getsource ...'
            #     python -c '... import <v> ... __file__ ...'
            #     python -c '... importlib ... <v> ...'
            #
            # Note: ``\b-c`` would not match because ``-`` is a non-word
            # char and the preceding space is also non-word — no word
            # boundary exists.  We anchor on whitespace instead.
            patterns.append(
                rf"(?i).*python\d?\b[^\n]*\s-c\b[^\n]*\bimport\s+{re.escape(v)}\b[^\n]*"
                r"(?:inspect\.getsource|__file__|importlib|get_data|pkgutil).*"
            )
            patterns.append(
                rf"(?i).*python\d?\b[^\n]*\s-m\s+{re.escape(v)}\b.*"
            )

        # 5 — Distribution hosts: any downloader hitting PyPI / Anaconda.
        host_group = "|".join(_PYPI_HOSTS)
        patterns.append(
            rf"(?i).*{_DOWNLOADERS}\b[^\n]*(?:{host_group}).*"
        )
        # And any shell-builtin redirect of those hosts via ``python -m http`` etc.
        patterns.append(rf"(?i).*https?://[^\s]*(?:{host_group}).*")

        # 6 — Catch-all for wheel / sdist artefacts anywhere on disk.
        #     If the agent already has a wheel cached somewhere, they can
        #     extract it — make that very obvious in trajectories so it
        #     shows up in logs even though we can't fully prevent it at
        #     the shell level.
        for v in self._package_variants:
            qv = _q(v)
            patterns.append(rf"(?i).*\bunzip\b[^\n]*{qv}[^\n]*\.whl.*")
            patterns.append(rf"(?i).*\b(?:tar|gunzip)\b[^\n]*{qv}[^\n]*\.(?:tar\.gz|tgz|zip).*")

        return patterns
