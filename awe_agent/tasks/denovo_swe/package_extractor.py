"""Embedded rule-based package-name extractor executed in each runtime container.

Runs inside a per-instance image (``git checkout <parent_commit>`` has
already been applied) and tries to recover the Python distribution
name + import names with a **broad** set of strategies.  Python
repositories in the wild are wildly inconsistent — they put setup
files in subdirectories, define the name via Python variables,
inherit from dynamic metadata, ship pre-built dist-info, declare
everything in a conda ``meta.yaml``, etc.  Every strategy below has
been seen in real projects and they all run, contributing candidates.
The runner picks the best one (priority-ordered) with every candidate
preserved for audit.

Strategies (ordered high → low priority):

1. **Root ``pyproject.toml``** — PEP 621 ``[project] name``, Poetry
   ``[tool.poetry] name``, Flit ``[tool.flit.metadata] module``,
   setuptools ``[tool.setuptools.dynamic] name``, Maturin
   ``[tool.maturin] name/module-name``.  TOML is parsed with
   ``tomllib`` (3.11+) or ``tomli``; regex fallback always runs too
   because Poetry/setuptools sometimes write TOML that stricter
   parsers reject.
2. **Root ``setup.py``** — AST walk every ``Call`` named ``setup``
   (direct, ``setuptools.setup``, ``distutils.core.setup``).  Also
   resolves simple top-level string assignments so that
   ``NAME = "foo"; setup(name=NAME)`` works.  If AST parsing fails we
   fall back to a broad ``name=["']...["']`` regex.
3. **Root ``setup.cfg``** — ``[metadata] name``, ``[options] packages``.
4. **Nested setup files** — walk at most ``_MAX_SCAN_DEPTH`` levels,
   skip known build/test/venv directories, repeat strategies 1-3 on
   each match.  Real monorepos sometimes keep the packaging files at
   ``python/setup.py`` or ``packages/<name>/pyproject.toml``.
5. **dist-info / egg-info** — ``*.dist-info/METADATA`` or
   ``*.egg-info/PKG-INFO`` anywhere under the workdir give us the
   canonical distribution name even for pre-installed editable repos.
   ``top_level.txt`` under those directories lists the import names.
6. **Version / ``__about__`` files** — top-level
   ``__about__.py`` / ``_version.py`` / ``version.py`` / ``_about.py``
   often carry ``__title__`` / ``__package_name__`` / ``NAME`` string
   literals that mirror the distribution name.
7. **Conda ``meta.yaml``** — ``package.name`` inside a recipe dir.
   Useful for the handful of NL2Repo projects that ship a conda recipe.
8. **``tox.ini``** — ``[tox] envlist`` occasionally names the package;
   we parse it but only as a very-low-priority candidate.
9. **Directory scan** — every top-level directory containing
   ``__init__.py`` (and the ``src/`` layout) contributes an import
   name.  Common test/doc/build dirs are filtered out.
10. **Repo basename** — final fallback; the runner may replace this
    with the ``repo`` field from the caller JSONL.

Emits JSON with:

* ``pypi_name``: the best-priority candidate name (``""`` if none).
* ``pypi_name_candidates``: every candidate with its source path.
* ``import_names``: sorted unique top-level import identifiers.
* ``setup_files``: relative paths of every packaging file discovered
  (useful for debugging unexpected layouts).
* ``workdir``: echo of the input for traceability.
"""

from __future__ import annotations


def get_extractor_script() -> str:
    """Return the doc2repo package-name extractor script body."""
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import configparser
import json
import re
from pathlib import Path
from typing import Any


# ── Tunables ───────────────────────────────────────────────────────────────

# Walk depth for nested setup files — 3 levels catches the common
# ``python/`` / ``packages/<foo>/`` / ``src/<foo>/`` monorepo layouts
# without drowning in vendored third-party copies.
_MAX_SCAN_DEPTH = 3

# Directory names we never descend into — saves time and, crucially,
# prevents vendored copies of another package from hijacking the name.
_SCAN_DIR_BLOCKLIST = {
    ".git", ".hg", ".svn", ".tox", ".nox",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", ".env",
    "node_modules",
    "build", "dist", "out",
    "tests", "test", "testing", "test_suite",
    "docs", "doc", "documentation",
    "examples", "example", "samples",
    "third_party", "thirdparty", "vendor", "vendored", "deps",
    "site-packages",
}

# Top-level import directories that aren't "the package we care about".
# Used to filter directory-scan import candidates.
_IMPORT_BLOCKLIST = {
    "tests", "test", "testing", "test_suite",
    "docs", "doc", "documentation",
    "examples", "example", "samples", "demos", "demo",
    "build", "dist", "out",
    "scripts", "bin", "tools",
    "benchmarks", "bench",
    "src",  # src-layout — we descend into it separately
}

# Priority ranking for candidate sources — lower number = higher trust.
_PRIORITY = [
    ("pyproject.toml:project.name",                       10),
    ("pyproject.toml:tool.poetry.name",                   11),
    ("pyproject.toml:tool.flit.metadata.module",          12),
    ("pyproject.toml:tool.maturin.name",                  13),
    ("pyproject.toml:tool.setuptools.dynamic",            14),
    ("pyproject.toml:regex",                              15),
    ("setup.py",                                          20),
    ("setup.py:regex",                                    25),
    ("setup.cfg",                                         30),
    ("dist-info",                                         40),
    ("egg-info",                                          41),
    ("__about__.py",                                      50),
    ("_version.py",                                       51),
    ("version.py",                                        52),
    ("_about.py",                                         53),
    ("meta.yaml",                                         60),
    ("tox.ini",                                           70),
]


def _source_priority(source: str) -> int:
    """Map a candidate ``source`` string to a comparable priority integer.

    Nested files add a small penalty per directory so that a root-level
    ``setup.py`` outranks ``subpkg/setup.py``.  Unknown sources sort
    after everything else but before the hard-coded basename fallback.
    """
    depth_penalty = source.count("/") * 5
    for prefix, base in _PRIORITY:
        if source == prefix or source.endswith("/" + prefix) or source.startswith(prefix + ":"):
            return base + depth_penalty
    if source == "workdir_basename":
        return 10_000
    return 9_000 + depth_penalty


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_toml(path: Path) -> dict | None:
    """Parse TOML with ``tomllib`` (3.11+) or ``tomli``; else ``None``."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return None
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return None


def _clean(s: str) -> str:
    return (s or "").strip().strip('"').strip("'").strip()


def _add_candidate(candidates: list[dict[str, str]], source: str, name: str) -> None:
    name = _clean(name)
    if not name:
        return
    # Skip common false positives
    if name.lower() in {"unknown", "package", "example", "my_package", "mypackage"}:
        return
    for existing in candidates:
        if existing["name"].lower() == name.lower() and existing["source"] == source:
            return
    candidates.append({"source": source, "name": name})


def _add_import(names: set[str], name: str) -> None:
    name = _clean(name)
    if not name:
        return
    if name.startswith("src."):
        name = name[4:]
    top = name.split(".", 1)[0]
    if not top or top.startswith("_") or top in _IMPORT_BLOCKLIST:
        return
    # Filter out obvious non-identifiers (e.g. ``find:`` placeholder).
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", top):
        return
    names.add(top)


def _iter_candidate_files(workdir: Path, filenames: set[str]) -> list[Path]:
    """BFS the workdir up to ``_MAX_SCAN_DEPTH`` looking for filenames.

    Respects ``_SCAN_DIR_BLOCKLIST`` to avoid descending into vendored
    copies / build trees.  Returns matches in depth-then-alpha order so
    root-level matches are processed first.
    """
    hits: list[tuple[int, Path]] = []
    queue: list[tuple[Path, int]] = [(workdir, 0)]
    while queue:
        current, depth = queue.pop(0)
        try:
            entries = sorted(current.iterdir())
        except Exception:
            continue
        for entry in entries:
            if entry.is_dir():
                if depth >= _MAX_SCAN_DEPTH:
                    continue
                if entry.name in _SCAN_DIR_BLOCKLIST or entry.name.startswith("."):
                    continue
                queue.append((entry, depth + 1))
            elif entry.is_file() and entry.name in filenames:
                hits.append((depth, entry))
    hits.sort(key=lambda t: (t[0], t[1].as_posix()))
    return [p for _, p in hits]


def _iter_glob_files(workdir: Path, globs: list[str]) -> list[Path]:
    """Collect files matching any glob; bounded depth via rglob with prune."""
    hits: list[Path] = []
    for pattern in globs:
        try:
            for p in workdir.rglob(pattern):
                # Skip anything under a blocklisted directory
                rel_parts = p.relative_to(workdir).parts
                if any(part in _SCAN_DIR_BLOCKLIST or part.startswith(".") for part in rel_parts[:-1]):
                    continue
                if len(rel_parts) > _MAX_SCAN_DEPTH + 2:
                    continue
                hits.append(p)
        except Exception:
            continue
    return hits


# ── Strategy 1: pyproject.toml ─────────────────────────────────────────────


def _from_pyproject(path: Path, rel: str, candidates, imports, setup_files):
    setup_files.append(rel)
    data = _load_toml(path)
    if data is not None:
        # PEP 621
        project = data.get("project") or {}
        _add_candidate(candidates, f"{rel}:project.name", str(project.get("name") or ""))
        for pkg in project.get("packages") or []:
            if isinstance(pkg, str):
                _add_import(imports, pkg)

        # Poetry
        poetry = (data.get("tool") or {}).get("poetry") or {}
        _add_candidate(candidates, f"{rel}:tool.poetry.name", str(poetry.get("name") or ""))
        for pkg in poetry.get("packages") or []:
            if isinstance(pkg, dict):
                _add_import(imports, str(pkg.get("include") or ""))
            elif isinstance(pkg, str):
                _add_import(imports, pkg)

        # Flit
        flit_meta = ((data.get("tool") or {}).get("flit") or {}).get("metadata") or {}
        _add_candidate(candidates, f"{rel}:tool.flit.metadata.module", str(flit_meta.get("module") or ""))
        # Newer Flit layout
        flit_module = ((data.get("tool") or {}).get("flit") or {}).get("module") or {}
        _add_candidate(candidates, f"{rel}:tool.flit.module.name", str(flit_module.get("name") or ""))

        # Maturin (Rust+Python)
        maturin = (data.get("tool") or {}).get("maturin") or {}
        _add_candidate(candidates, f"{rel}:tool.maturin.name", str(maturin.get("name") or ""))
        _add_candidate(candidates, f"{rel}:tool.maturin.module-name", str(maturin.get("module-name") or ""))

        # setuptools dynamic / packages
        setuptools_cfg = (data.get("tool") or {}).get("setuptools") or {}
        dyn = setuptools_cfg.get("dynamic") or {}
        if isinstance(dyn, dict):
            dyn_name = dyn.get("name")
            if isinstance(dyn_name, dict):
                _add_candidate(
                    candidates, f"{rel}:tool.setuptools.dynamic",
                    str(dyn_name.get("file") or dyn_name.get("attr") or ""),
                )
        pkgs_cfg = setuptools_cfg.get("packages")
        if isinstance(pkgs_cfg, list):
            for p in pkgs_cfg:
                if isinstance(p, str):
                    _add_import(imports, p)
        elif isinstance(pkgs_cfg, dict):
            for p in (pkgs_cfg.get("find") or {}).get("include") or []:
                if isinstance(p, str):
                    _add_import(imports, p)

        # Entry-points imply an import name on the left of ``:``.
        for entry in (project.get("scripts") or {}).values():
            if isinstance(entry, str) and ":" in entry:
                _add_import(imports, entry.split(":", 1)[0])
        for group in (project.get("entry-points") or {}).values():
            if isinstance(group, dict):
                for entry in group.values():
                    if isinstance(entry, str) and ":" in entry:
                        _add_import(imports, entry.split(":", 1)[0])

    # Regex fallback — run always, cheap.
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    # Match ``name = "foo"`` or ``name = 'foo'``, first occurrence (root).
    for m in re.finditer(
        r'^\s*name\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE,
    ):
        _add_candidate(candidates, f"{rel}:regex", m.group(1))


# ── Strategy 2: setup.py ────────────────────────────────────────────────────


def _collect_setup_py_globals(tree: ast.AST) -> dict[str, str]:
    """Harvest top-level ``NAME = "str"`` assignments for later lookup.

    Handles the surprisingly common pattern ::

        NAME = "my-pkg"
        setup(name=NAME, ...)
    """
    out: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            value = None
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                value = None
            if not isinstance(value, str):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = value
    return out


def _from_setup_py(path: Path, rel: str, candidates, imports, setup_files):
    setup_files.append(rel)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    # Regex pass first — always useful even if AST fails.
    for m in re.finditer(r'\bname\s*=\s*["\']([^"\']+)["\']', source):
        _add_candidate(candidates, f"{rel}:regex", m.group(1))

    try:
        tree = ast.parse(source)
    except Exception:
        return

    globals_ = _collect_setup_py_globals(tree)

    def _resolve(node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except Exception:
            pass
        if isinstance(node, ast.Name) and node.id in globals_:
            return globals_[node.id]
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr
        if func_name != "setup":
            continue

        for kw in node.keywords:
            if kw.arg == "name":
                v = _resolve(kw.value)
                if isinstance(v, str):
                    _add_candidate(candidates, rel, v)
            elif kw.arg == "packages":
                v = _resolve(kw.value)
                if isinstance(v, list):
                    for p in v:
                        if isinstance(p, str):
                            _add_import(imports, p)
            elif kw.arg == "py_modules":
                v = _resolve(kw.value)
                if isinstance(v, list):
                    for m_ in v:
                        if isinstance(m_, str):
                            _add_import(imports, m_)


# ── Strategy 3: setup.cfg ───────────────────────────────────────────────────


def _from_setup_cfg(path: Path, rel: str, candidates, imports, setup_files):
    setup_files.append(rel)
    try:
        c = configparser.ConfigParser()
        c.read(path, encoding="utf-8")
    except Exception:
        return

    if c.has_option("metadata", "name"):
        _add_candidate(candidates, rel, c.get("metadata", "name", fallback=""))

    if c.has_option("options", "packages"):
        raw = c.get("options", "packages", fallback="")
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if line and line not in {"find:", "find_namespace:"}:
                _add_import(imports, line)
    if c.has_option("options", "py_modules"):
        raw = c.get("options", "py_modules", fallback="")
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if line:
                _add_import(imports, line)


# ── Strategy 4: dist-info / egg-info ───────────────────────────────────────


def _from_dist_info(workdir: Path, candidates, imports):
    for dir_pattern, meta_basename, tag in (
        ("*.dist-info", "METADATA", "dist-info"),
        ("*.egg-info", "PKG-INFO", "egg-info"),
    ):
        # Scan anywhere under the workdir (some repos ship pre-built
        # dist-info in a nested build/ dir — we already skip build/,
        # so only "real" ones make it here).
        for p in _iter_glob_files(workdir, [dir_pattern + "/" + meta_basename]):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            m = re.search(r"^Name:\s*(.+)$", text, re.MULTILINE)
            if m:
                rel = p.relative_to(workdir).as_posix()
                _add_candidate(candidates, rel, m.group(1).strip())
        # top_level.txt next to the metadata
        for p in _iter_glob_files(workdir, [dir_pattern + "/top_level.txt"]):
            try:
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    _add_import(imports, line.strip())
            except Exception:
                pass


# ── Strategy 5: __about__.py / _version.py / version.py / _about.py ────────


_VERSION_FILENAMES = ["__about__.py", "_version.py", "version.py", "_about.py"]
_VERSION_NAME_KEYS = ("__title__", "__package_name__", "__pkgname__", "__distname__", "NAME", "PACKAGE", "PROJECT")


def _from_version_files(workdir: Path, candidates):
    for p in _iter_candidate_files(workdir, set(_VERSION_FILENAMES)):
        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = p.relative_to(workdir).as_posix()

        # Fast regex for ``__title__ = "foo"`` and siblings.
        for key in _VERSION_NAME_KEYS:
            pat = rf'^\s*{re.escape(key)}\s*=\s*["\']([^"\']+)["\']'
            m = re.search(pat, source, re.MULTILINE)
            if m:
                _add_candidate(candidates, p.name, m.group(1))
                break
        else:
            # AST fallback for cases where the regex misses quoting
            try:
                tree = ast.parse(source)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    try:
                        v = ast.literal_eval(node.value)
                    except Exception:
                        continue
                    if not isinstance(v, str):
                        continue
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id in _VERSION_NAME_KEYS:
                            _add_candidate(candidates, p.name, v)


# ── Strategy 7: conda meta.yaml ────────────────────────────────────────────


def _from_conda_meta(workdir: Path, candidates):
    for p in _iter_glob_files(workdir, ["meta.yaml", "recipe/meta.yaml", "conda.recipe/meta.yaml"]):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        # Grep ``name: foo`` under a ``package:`` key, tolerant of
        # Jinja {{ }} templating that PyYAML would choke on.
        m = re.search(r'^\s*package\s*:\s*\n(?:\s+[^\n]*\n)*?\s+name\s*:\s*["\']?([A-Za-z0-9_.\-{}\s]+?)["\']?\s*$', text, re.MULTILINE)
        if m:
            name = re.sub(r"\{\{.*?\}\}", "", m.group(1)).strip()
            if name:
                _add_candidate(candidates, "meta.yaml", name)


# ── Strategy 8: tox.ini ────────────────────────────────────────────────────


def _from_tox_ini(workdir: Path, candidates):
    p = workdir / "tox.ini"
    if not p.exists():
        return
    try:
        c = configparser.ConfigParser()
        c.read(p, encoding="utf-8")
    except Exception:
        return
    # ``[tox] envlist`` itself isn't a name, but ``package_name`` in
    # various sections sometimes is.
    for section in c.sections():
        for key in ("package_name", "pkg", "project_name"):
            if c.has_option(section, key):
                _add_candidate(candidates, "tox.ini", c.get(section, key))


# ── Strategy 9: top-level directory scan ───────────────────────────────────


def _from_directory_scan(workdir: Path, imports):
    if not workdir.is_dir():
        return
    for entry in workdir.iterdir():
        if entry.name in _SCAN_DIR_BLOCKLIST or entry.name.startswith("."):
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            _add_import(imports, entry.name)
    src = workdir / "src"
    if src.is_dir():
        for entry in src.iterdir():
            if entry.name in _SCAN_DIR_BLOCKLIST or entry.name.startswith("."):
                continue
            if entry.is_dir() and (entry / "__init__.py").exists():
                _add_import(imports, entry.name)


# ── Runner ──────────────────────────────────────────────────────────────────


def _run(args: argparse.Namespace) -> int:
    meta: dict[str, Any] = {"error": None}
    candidates: list[dict[str, str]] = []
    imports_set: set[str] = set()
    setup_files: list[str] = []

    try:
        workdir = Path(args.workdir).resolve()
        if not workdir.is_dir():
            raise FileNotFoundError(f"workdir not found: {args.workdir}")

        # 1-3. Walk for nested setup files and apply strategies 1-3.
        #      Respects ``_MAX_SCAN_DEPTH`` and the dir blocklist.
        targets = _iter_candidate_files(
            workdir,
            filenames={"pyproject.toml", "setup.py", "setup.cfg"},
        )
        for path in targets:
            rel = path.relative_to(workdir).as_posix()
            if path.name == "pyproject.toml":
                _from_pyproject(path, rel, candidates, imports_set, setup_files)
            elif path.name == "setup.py":
                _from_setup_py(path, rel, candidates, imports_set, setup_files)
            elif path.name == "setup.cfg":
                _from_setup_cfg(path, rel, candidates, imports_set, setup_files)

        # 4-8. Auxiliary sources.
        _from_dist_info(workdir, candidates, imports_set)
        _from_version_files(workdir, candidates)
        _from_conda_meta(workdir, candidates)
        _from_tox_ini(workdir, candidates)

        # 9. Directory scan (import names only).
        _from_directory_scan(workdir, imports_set)

        # Priority-order candidates; dedupe same-name entries, keeping
        # the highest-priority source for each name.
        ranked = sorted(
            candidates,
            key=lambda c: (_source_priority(c["source"]), c["source"]),
        )
        seen: set[str] = set()
        deduped: list[dict[str, str]] = []
        for c in ranked:
            key = c["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)

        pypi_name = deduped[0]["name"] if deduped else ""

        out = {
            "pypi_name": pypi_name,
            "pypi_name_candidates": deduped,
            "import_names": sorted(imports_set),
            "setup_files": setup_files,
            "workdir": str(workdir),
        }
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        out = {
            "pypi_name": "",
            "pypi_name_candidates": candidates,
            "import_names": sorted(imports_set),
            "setup_files": setup_files,
            "workdir": args.workdir,
        }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    with open(args.meta_output, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="doc2repo package-name extractor")
    parser.add_argument("--workdir", required=True, help="Repository root inside the runtime")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--meta-output", required=True, help="Output meta JSON path")
    return parser.parse_args()


def main() -> int:
    return _run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
'''
