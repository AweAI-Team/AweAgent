"""Predict whether ``clean.sh`` would delete a given file.

``clean.sh`` runs at:

* ``task.prepare_session`` — before the agent starts (de-novo setup).
* ``evaluator._evaluate_once`` — before applying agent + test patches.

It deletes files by a mix of (a) directory-name wipes (recursive
``rm -rf`` for any dir whose basename matches a fixed list) and
(b) file-name/extension rules with a small *preserve* allow-list for
root-level config files.

We need to predict its behaviour purely from a path string so that:

1. ``extract_patch.py`` can avoid including files that ``clean.sh``
   would preserve anyway — they'd survive into the eval workspace, so
   shipping them in the patch is wasted bytes (and trips the
   ``_files_added_by_test_patch`` predelete + reapply path in the
   evaluator for no reason).
2. The Bug B whitelist scanner only restores files that ``clean.sh``
   would actually delete (mostly ``.py`` / ``.json`` / ``.yml`` inside
   sibling helper dirs), not the binary fixtures that survive.

This module is the single source of truth for the prediction logic.
``clean.sh`` is the authority; if it changes, update this file and the
unit tests.
"""

from __future__ import annotations

import re


# ── Directories whose **basename** triggers ``rm -rf`` in clean.sh ───
# Mirrors clean.sh line 87-94 exactly: precise (lower-cased) match,
# no substring fuzz.
_WIPE_DIR_NAMES = frozenset({
    "test", "tests",
    "testsuite", "testsuites",
    "testing", "test_suite",
})


# ── Python-source preserve filter (clean.sh line 107) ────────────────
# The actual filter is ``grep -vE PATTERN`` applied to the **full path**
# emitted by ``find`` (e.g. ``./helpers/ez_setup.py``).  The pattern is
# anchored at end with ``$`` only, NOT at start, so any path whose last
# characters match counts as preserved:
#
#   ``ez_setup.py``                  → ends in ``setup.py``  → preserved
#   ``pkg/conda_setup.py``           → ends in ``setup.py``  → preserved
#   ``mesonbuild/msetup.py``         → ends in ``setup.py``  → preserved
#   ``foo/LICENSE.MD``               → ends in ``LICENSE.MD`` → preserved
#                                                              (LICENSE.* greedy)
#
# We mirror that exact regex so our predictor agrees with clean.sh
# even on the looser corners.  Verified against 70k+ files in
# post_clean_snapshots.jsonl.
_PRESERVE_PATH_RE = re.compile(
    r"(?:"
    r"pyproject\.toml"
    r"|setup\.py"
    r"|setup\.cfg"
    r"|requirements.*\.txt"
    r"|MANIFEST\.in"
    r"|Dockerfile"
    r"|\.python-version"
    r"|Makefile"
    r"|poetry\.lock|Pipfile\.lock|pdm\.lock|uv\.lock"
    r"|\.gitignore|\.gitattributes"
    r"|LICENSE.*|LICENCE.*"
    r")$"
)


# ── File extensions that ``clean.sh`` deletes outright (line 105-148) ─
# clean.sh uses ``find -name "*.ext"`` which is **case-SENSITIVE** on
# Linux (unlike ``-iname`` which is what the test-file rule uses).  So
# ``foo.MD`` survives even though ``foo.md`` gets deleted.  We compare
# basename suffixes verbatim, not lower-cased.
_DELETE_EXACT_BASENAMES = frozenset({
    ".coverage", "coverage.xml", "coverage.json",
    ".DS_Store", ".env", ".envrc",
    ".coveragerc",
})


# Plain-extension delete list — case-SENSITIVE.
_DELETE_EXTENSIONS = (
    # Python source (preserve list above takes precedence)
    ".py", ".pyx", ".pxd", ".pyi",
    # Compiled / packaged native code
    ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".egg", ".whl",
    # Doc / config that may leak implementation
    ".rst", ".md",
    ".yml", ".yaml",
    ".json",
    ".log", ".trace",
    ".ipynb",
)


# Multi-part extensions (must be matched before the single-ext list).
_DELETE_MULTI_SUFFIXES = (
    ".min.js", ".min.css",
)


# Test-file name patterns (clean.sh line 96-100, ``-iname`` → CASE-IN).
_TEST_FILE_NAME_RE = re.compile(
    r"^(?:test_.*\.py|.*_test\.py|.*_tests\.py|conftest\.py)$",
    re.IGNORECASE,
)


def will_clean_delete(path: str) -> bool:
    """Return True iff ``clean.sh`` would delete ``path``.

    ``path`` must be a forward-slash relative path inside the workdir,
    e.g. ``helpers/singleton.py``.  Absolute paths and ``./`` prefixes
    are tolerated by stripping them.

    The check sequence intentionally matches ``clean.sh``'s execution
    order:

    1.  Any ancestor directory whose basename is in ``_WIPE_DIR_NAMES``
        causes an unconditional ``rm -rf``, so the file inside is
        guaranteed deleted regardless of extension.
    2.  Test file-name patterns (``test_*.py`` etc.) — clean.sh line
        96-100 deletes these anywhere, not just under test dirs.
    3.  Root-level preserve list — names like ``setup.py`` /
        ``LICENSE.md`` survive even though their extension is in the
        delete list.  clean.sh applies preserve based on basename, so
        ``helpers/setup.py`` ALSO matches (this matches the observed
        behaviour for kobotoolbox where ``helpers/setup.py`` survived
        clean).
    4.  Extension-based delete (Python source, JSON, YAML, MD, RST,
        IPYNB, native binaries, etc.).
    5.  Default: surviving.

    Edge cases handled explicitly:

    * Empty path / single ``.`` → False (no file to delete).
    * Case insensitivity for directory-name wipes (clean.sh
      lowercases via ``tr``).
    * Multi-suffix extensions (``foo.min.js``) checked before single
      suffix (``.js`` isn't in our delete list anyway, but the order
      keeps the rule explicit).
    """
    if not path or path == ".":
        return False

    # Normalise: drop leading ``./`` and any leading ``/``.  We don't
    # support globs or unicode tricks beyond what Python's str gives us.
    norm = path.lstrip("/").removeprefix("./")
    if not norm:
        return False

    parts = norm.split("/")
    basename = parts[-1]
    # NOTE: we keep ``basename`` case-as-is below — clean.sh's
    # extension-based deletes use ``-name``, which is case-SENSITIVE.

    # 1. Wipe-dir ancestor check.  clean.sh's ``find -type d`` walks
    # every directory; if *any* ancestor matches the wipe list, the
    # whole subtree is removed.  ``tr [:upper:] [:lower:]`` in
    # clean.sh makes this comparison case-INsensitive.
    for seg in parts[:-1]:
        if seg.lower() in _WIPE_DIR_NAMES:
            return True

    # 2. Test file-name patterns (``find -iname`` → case-INsensitive).
    if _TEST_FILE_NAME_RE.match(basename):
        return True

    # 3. Preserve filter — applied to the **full path** with end-of-line
    # anchor only, matching the actual ``grep -vE PATTERN$`` behaviour.
    # This means e.g. ``helpers/ez_setup.py`` survives because the path
    # ends in ``setup.py``.
    if _PRESERVE_PATH_RE.search(norm):
        return False

    # 4. Exact-basename delete list (case-sensitive in clean.sh).
    if basename in _DELETE_EXACT_BASENAMES:
        return True

    # 5. Multi-suffix extensions (must be tried before single suffix).
    for ms in _DELETE_MULTI_SUFFIXES:
        if basename.endswith(ms):
            return True

    # 6. Single-extension delete list (case-sensitive).
    for ext in _DELETE_EXTENSIONS:
        if basename.endswith(ext):
            return True

    # 7. Anything not matched survives.
    return False
