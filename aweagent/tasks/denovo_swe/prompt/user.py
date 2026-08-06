"""DeNovoSWE user prompt templates.

Based on BeyondSWE doc2repo prompts — the task is to build a repository
from a specification document, but with source-cleaned images.
"""

from __future__ import annotations

# ── DeNovoSWE: Non-search prompt ─────────────────────────────────────────────


DENOVOSWE_DOC2REPO_PROMPT = """I need you to implement a software repository from scratch based on a strict architectural specification.

### 1. Context & Environment
* **Workspace Directory**: `{workspace_dir}`
* **Current File Structure**: The workspace has been source-cleaned — most repo files are gone. Discover the remaining layout yourself with **one** command, e.g. `find {workspace_dir} -maxdepth 4 -type f -not -path '*/.git/*' | sort | head -60`. Do NOT spend multiple turns re-listing the same paths. Re-list only after you have created new files yourself.

<DO_NOT_MODIFY>
The following files are off-limits and will be reverted/ignored by the evaluator — do not read, edit, create, or delete them:
* **`README.md`** — this file holds the spec, and the FULL spec text is already inlined below in this prompt. Do not `cat` it (you already have it) and do not modify it.
* **`tests/`, `test_*.py`, `*_test.py`, `*_tests.py`, `conftest.py`** — the evaluator nukes all test files in the workspace before scoring and re-applies the original test_patch. Any test edits you make are discarded; if your "fix" required editing a test, your implementation is wrong.
* **`.gitignore`** — runner-managed; do not touch.
* **Hidden runner-helper files** (`.subdoc_partial_clean.*`, `.clean.sh`, etc.) — runtime artefacts; do not touch.

Edits MUST land inside the real package directory (`<pkg>/...`) — not `tests/`, not `README.md`.
</DO_NOT_MODIFY>
* **Installation Status**: The original package has been **uninstalled** by the source-cleaning step. You will re-install it with `pip install -e .` after writing the code (Phase 3, Step 1). Changes inside the package directory (name determined in Phase 1, Step 1) take effect immediately once the editable install succeeds.

### 2. Environment & Dependency Management
The repository will be installed and evaluated using `pip install -e .`. You must manage dependencies strictly via `setup.py`.

<CURRENT_ENVIRONMENT>
The following packages are **ALREADY INSTALLED** in the environment. You can use them directly without reinstalling:
```text
{installed_packages}
```
*(Note: Do not assume any other packages exist unless listed above.)*
</CURRENT_ENVIRONMENT>

<DEPENDENCY_RULES>
1. **Setup.py is King**: You MUST list all necessary runtime dependencies in `setup.py`.
2. **Avoid Redundancy**: If a package is already in **<CURRENT_ENVIRONMENT>**, do **NOT** add it to `setup.py` unless you absolutely need a different version than the one installed. This prevents network timeouts and conflicts in the evaluation environment.
3. **Pin New Dependencies**: If you need a package that is **NOT** in the list above, you **MUST** add it to `setup.py` and **explicitly pin the version**.
4. **Ignore requirements.txt**: Do not create or update `requirements.txt`. Only modify `setup.py`.
</DEPENDENCY_RULES>

### 3. The Specification (`README.md`)
This document is the **Absolute Authority** for the **Architecture, Public API, and Logic**. You must implement the repository exactly as described here.
```markdown
{REPO_DOCUMENT}
```

---

### 4. Task Instructions

**File-Editing Discipline**
For all file writes/edits in the workspace, use the `str_replace_editor` tool — not `cat <<'EOF'` heredocs or `sed -i` (quoting/escaping bugs silently corrupt code). Plan each file in one shot before writing it; for a small in-place edit use `str_replace_editor` (the `old_str` must occur exactly once in the file), and for a full rewrite `rm <path>` then call `str_replace_editor` to create the file fresh (it fails if the path already exists).

**Phase 1: Analysis & Research (Critical Discipline)**
* **Analyze the Spec**: The spec is **already inlined above in this prompt** — work from it directly. Do **NOT** open, `cat`, or `Read` `README.md`. Use the inlined spec to deduce the required directory structure and class hierarchy based strictly on the **Import Paths** defined in the document.
* **Check Dependencies**: Compare spec requirements against <CURRENT_ENVIRONMENT>. Decide which ones need to be added to `setup.py` and which ones are already present.

**Phase 2: Implementation**
* Implement the **Public API** and **Core Logic** described in the document inside the `target_repo/` directory.
* **Strict Constraints**:
    * Function signatures (arguments, types, return values) MUST match the document exactly.
    * Ensure all imports use relative imports (e.g., `from . import utils`) or absolute imports starting with `target_repo`.
    * If you create a `__main__.py` for the CLI, the entrypoint MUST be guarded by `if __name__ == "__main__":`. A bare `sys.exit(main())` at module top level runs the CLI on `from package import __main__` and breaks every test that imports it.
* **Internal Implementation Flexibility**:
    * The document lists the **Core Public API**. You are encouraged to create necessary **private helper functions**, **internal constants**, or **utility classes** to support the logic.
    * **Guidelines**:
        * Keep internal helpers private (prefix with `_`) where appropriate.
        * You may create new utility files inside the `target_repo/` directory if the logic requires it, but **do not abuse this**---keep the structure clean and logical.
        * Do NOT change the signature of the documented Public APIs.

**Phase 3: Verification (Self-Correction)**
* Since this is a clean-room implementation, **NO existing tests are provided**.
* You are responsible for verifying your own code:
    * Create a standalone script (e.g., `verify_implementation.py`) to import your new classes/functions and assert they behave as documented.
    * OR write simple `pytest` cases to check critical logic.
* **Goal**: Ensure your implementation runs without errors and matches the spec before you finish.

**Phase 4: Submission**
* **No git operations needed — you don't need to perform any git operations.** The evaluator captures your file changes automatically by diffing the workspace against the baseline commit when you call `finish`. Do **NOT** run any git command (`git add`, `git commit`, `git config user.*`, `git status`, `git diff`, `git checkout`, etc.) — manual git operations have **zero** effect on your score and only waste turns.
* Once you are confident:
    1. Delete any temporary verification scripts (`verify_implementation.py`) to keep the repo clean.
    2. Ensure `setup.py` is configured correctly according to <DEPENDENCY_RULES>.
    3. Submit your work using the "finish" tool.

<ANTI_CHEAT_CONSTRAINT>
**ABSOLUTE RULE: DO NOT DOWNLOAD THE TARGET PACKAGE'S SOURCE CODE FROM THE INTERNET.** You MUST implement the target project from scratch based solely on the specification document above. It is a HARD VIOLATION to look up, download, install, clone, or copy the target package's source code from any external source — this includes PyPI, GitHub, GitLab, Bitbucket, Codeberg, Anaconda, conda-forge, personal mirrors, company-internal mirrors, Google/Bing cache, the Wayback Machine, any local wheel/sdist cache on disk, or any other channel.

Specifically, the following are **FORBIDDEN** and will be auto-blocked:

* Package managers pulling the target: `pip install <target>`, `pip3.X install <target>`, `pip download <target>`, `pip show <target>`, `pip wheel <target>`, `python -m pip install <target>`, `pipx install <target>`, `pipenv install <target>`, `uv pip install <target>`, `uv add <target>`, `uvx <target>`, `poetry add <target>`, `conda/mamba/micromamba install <target>`, `pdm add <target>`, `hatch env install <target>`, `easy_install <target>` — in every path-prefixed / sudo-prefixed / env-prefixed / versioned form.
* VCS: `git clone` or `git submodule add` pointing at the target repo (anywhere: github.com, gitlab.com, bitbucket.org, codeberg.org, any mirror).
* Raw downloads: `curl`/`wget`/`aria2c`/`http`/`httpie`/`lwp-download`/`fetch` hitting PyPI-family hosts (`pypi.org`, `files.pythonhosted.org`, `pypi.python.org`, `test.pypi.org`, `pythonhosted.org`, `pypistats.org`, `conda.anaconda.org`, `anaconda.org`, `repo.anaconda.com`) or any URL containing the target name.
* Python introspection: `python -c 'import <target>; inspect.getsource(<target>)'`, reading `<target>.__file__`, or any `importlib`/`pkgutil`/`get_data` trick to dump the installed upstream source.
* Extracting on-disk artefacts: `unzip <target>*.whl`, `tar xf <target>*.tar.gz`, etc.

The following are **ALLOWED**:

* Installing *third-party* dependencies your own implementation needs (e.g. `pip install requests`, `pip install numpy`).
* Running `pip install -e .` on the code you yourself wrote inside the workspace.
* Reading any file already present in the workspace at the start of your session — **except `README.md`**, whose full spec is already inlined in this prompt and which is in the `<DO_NOT_MODIFY>` set above.

Violations are detected automatically by the `execute_bash` tool's security policy and will be blocked — the blocked command returns an error instead of running. Repeated violations will not succeed; do not try to obfuscate (variable-name rewrites, base64 decode pipes, curl into a random-named file, etc.) — the blocklist covers name variants (`Pkg` / `pkg` / `pkg-name` / `pkg_name` / `PKG_NAME`) and all common package-manager spellings.
</ANTI_CHEAT_CONSTRAINT>
"""


# ── DeNovoSWE v2: Non-search prompt (with public API verification gate) ─────

DENOVOSWE_DOC2REPO_PROMPT_V2 = """I need you to implement a software repository from scratch based on a strict architectural specification.

### 1. Context & Environment
* **Workspace Directory**: `{workspace_dir}`
* **Current File Structure**: The workspace has been source-cleaned — most repo files are gone. Discover the remaining layout yourself with **one** command, e.g. `find {workspace_dir} -maxdepth 4 -type f -not -path '*/.git/*' | sort | head -60`. Do NOT spend multiple turns re-listing the same paths. Re-list only after you have created new files yourself.

<DO_NOT_MODIFY>
The following files are off-limits and will be reverted/ignored by the evaluator — do not read, edit, create, or delete them:
* **`README.md`** — this file holds the spec, and the FULL spec text is already inlined below in this prompt. Do not `cat` it (you already have it) and do not modify it.
* **`tests/`, `test_*.py`, `*_test.py`, `*_tests.py`, `conftest.py`** — the evaluator nukes all test files in the workspace before scoring and re-applies the original test_patch. Any test edits you make are discarded; if your "fix" required editing a test, your implementation is wrong.
* **`.gitignore`** — runner-managed; do not touch.
* **Hidden runner-helper files** (`.subdoc_partial_clean.*`, `.clean.sh`, etc.) — runtime artefacts; do not touch.

Edits MUST land inside the real package directory (`<pkg>/...`) — not `tests/`, not `README.md`.
</DO_NOT_MODIFY>
* **Installation Status**: The original package has been **uninstalled** by the source-cleaning step. You will re-install it with `pip install -e .` after writing the code (Phase 3, Step 1). Changes inside the package directory (name determined in Phase 1, Step 1) take effect immediately once the editable install succeeds.

### 2. Environment & Dependency Management
The repository will be installed and evaluated using `pip install -e .`. You must manage dependencies strictly via `setup.py`.

<CURRENT_ENVIRONMENT>
The following packages are **ALREADY INSTALLED** in the environment. You can use them directly without reinstalling:
```text
{installed_packages}
```
*(Note: Do not assume any other packages exist unless listed above.)*
</CURRENT_ENVIRONMENT>

<DEPENDENCY_RULES>
1. **Setup.py is King**: You MUST list all necessary runtime dependencies in `setup.py`.
2. **Avoid Redundancy**: If a package is already in **<CURRENT_ENVIRONMENT>**, do **NOT** add it to `setup.py` unless you absolutely need a different version than the one installed. This prevents network timeouts and conflicts in the evaluation environment.
3. **Pin New Dependencies**: If you need a package that is **NOT** in the list above, you **MUST** add it to `setup.py` and **explicitly pin the version**.
4. **Ignore requirements.txt**: Do not create or update `requirements.txt`. Only modify `setup.py`.
</DEPENDENCY_RULES>

### 3. The Specification (`README.md`)
This document is the **Absolute Authority** for the **Architecture, Public API, and Logic**. You must implement the repository exactly as described here.
```markdown
{REPO_DOCUMENT}
```

---

### 4. Task Instructions

**File-Editing Discipline**
For all file writes/edits in the workspace, use the `str_replace_editor` tool — not `cat <<'EOF'` heredocs or `sed -i` (quoting/escaping bugs silently corrupt code). Plan each file in one shot before writing it; for a small in-place edit use `str_replace_editor` (the `old_str` must occur exactly once in the file), and for a full rewrite `rm <path>` then call `str_replace_editor` to create the file fresh (it fails if the path already exists).

**Phase 1: Analysis & Research (Critical Discipline)**
* **Step 1 (must-do FIRST) — Determine the import (package) name.** The *repo directory name* is **NOT necessarily** the *import name* (what tests will `import` your code as). Before reading the inlined spec or writing any code, check the workspace for `setup.py`, `setup.cfg`, and `pyproject.toml` — **read just ONE** of these (whichever first declares the package name via `packages=` / `name=` / `[project] name`) and stop. No need to open all three. Use this import name everywhere: top-level package directory, internal imports, verification scripts. Do **NOT** create a `target_repo/` directory.
* **Analyze the Spec**: The spec is **already inlined above in this prompt** — work from it directly. Do **NOT** open, `cat`, or `Read` `README.md`. Use the inlined spec to deduce the required directory structure and class hierarchy based strictly on the **Import Paths** defined in the document.
* **Check Dependencies**: Compare spec requirements against <CURRENT_ENVIRONMENT>. Decide which ones need to be added to `setup.py` and which ones are already present.

**Phase 2: Implementation**
* **Use the import name you determined in Phase 1, Step 1** — all your source code MUST go into the directory that matches that import name. Do **NOT** create a directory called `target_repo/`.
* Implement the **Public API** and **Core Logic** described in the document inside the package directory.
* **Strict Constraints**:
    * Function signatures (arguments, types, return values) MUST match the document exactly.
    * Ensure all imports use relative imports (e.g., `from . import utils`) or absolute imports starting with the **real package name** (e.g., `from mypackage.module import func`).
    * If you create a `__main__.py` for the CLI, the entrypoint MUST be guarded by `if __name__ == "__main__":`. A bare `sys.exit(main())` at module top level runs the CLI on `from package import __main__` and breaks every test that imports it.
    * Always define `__version__` in the top-level `__init__.py` (e.g. `__version__ = "0.1.0"`) even if the spec doesn't mention versioning — test suites commonly do `from <pkg> import __version__`, and a missing one crashes test collection for the whole file.
    * In `<pkg>/<subpkg>/__init__.py`, NEVER write `from <pkg>.<subpkg> import name1, name2`. It re-enters the module being initialised and causes circular `ImportError` non-deterministically depending on caller import order. Use relative imports instead: `from .module_a import name1`, `from .module_b import name2`.
    * **Declare `__all__` explicitly at the top of every `__init__.py`** — list every re-exported public symbol. This is the spec-compliance contract AND the checklist your Phase 3 inline import probe iterates over.
    * **Implement leaf modules before writing their `__init__.py`** — build `__init__.py` re-exports last, against modules that already exist. Assembling `__init__.py` against not-yet-written modules is the #1 cause of avoidable circular-import / `NameError` cascades.
    * **Inherit the directory layout declared by `setup.py` / `setup.cfg` / `pyproject.toml`** — typically a flat `<pkg>/` under the workspace root. Do **NOT** introduce a `src/<pkg>/` subdir if the setup file doesn't already use one.
    * When the spec illustrates a function called with keyword arguments (e.g. `client._api_request(endpoint='/x', mode='GET')`), every internal call site MUST use the same keyword form. Mock-based tests assert `mock.assert_called_with(endpoint=..., mode=...)` and a positional call fails even when functionally equivalent.
    * When parsing structured input (XML/JSON/CSV/binary), never silently fall back to `date.today()`, `datetime.now()`, `0`, or `""` for fields the document declares. If a documented field is missing in the input, raise an error — silent defaults produce tests that pass today and fail tomorrow.
* **Internal Implementation Flexibility**:
    * The document lists the **Core Public API**. You are encouraged to create necessary **private helper functions**, **internal constants**, or **utility classes** to support the logic.
    * **Guidelines**:
        * Keep internal helpers private (prefix with `_`) where appropriate.
        * You may create new utility files inside the package directory if the logic requires it, but **do not abuse this**---keep the structure clean and logical.
        * Do NOT change the signature of the documented Public APIs.
        * Note: official tests sometimes import internal helpers (`_helper`, `_build_xxx`) directly for white-box testing. Hints about which internals matter: coverage configs in `setup.cfg` / `pyproject.toml`, and module names referenced in the spec narrative even if not in the Public API table.

**Phase 3: Verification (MANDATORY — Finish Gate)**

Before you may call the "finish" tool, you MUST complete ALL of the following verification steps. This is the most critical phase — your implementation will be evaluated by running `pip install -e .` followed by official unit tests that import and exercise the public API. If imports fail or core classes/functions are missing, **you will score zero**.

<FINISH_GATE>
**Step 0: Re-confirm the import name from Phase 1, Step 1** — use it (not any placeholder like `target_repo`) in all verification imports below.

**Step 1: Re-install in editable mode**
```bash
pip install -e .
```
Use plain `pip install -e .` only — do **NOT** pass `--force-reinstall` / `--upgrade` / `-U`, which can silently upgrade `pytest` and break the evaluator.

**Step 1.5: Confirm the install actually succeeded (do this from `/tmp`, NOT the source dir)**
```bash
cd /tmp && python -c "import <pkg>; print(<pkg>.__file__)"
```
This MUST be run from `/tmp` (or any directory outside the source tree). Running `python -c "import <pkg>"` from inside the source directory can succeed even when the install failed, because Python's `sys.path` picks up the source directory directly. If this command fails with `ModuleNotFoundError`, your install is broken — even if `pip install -e .` exited 0. Common causes: `setup.py` hooks that error (e.g. `setuptools_scm`, custom `get_git_version()`), `src/<pkg>/` layouts the install couldn't link, or compiled extensions (`.so`/Rust/Cython) that failed to build. Fix `setup.py` / `pyproject.toml` and reinstall before continuing.

> A passing import only confirms packaging is correct — it does NOT prove behavior is correct. Before calling `finish`, exercise the documented public API end-to-end with realistic inputs and assert the results match what the spec says. A short Python script (`/tmp/smoke.py`), an inline `python -c "..."`, or your own pytest cases under `tests/` are all acceptable — just don't rely on import-only checks.

**Step 2: Verify every public import path**
**Preferred form — a single inline `python -c` that enumerates your `__all__` list in one tuple-import.** If any symbol is missing the traceback names it on line 1 — cleaner than pytest's collection stacktrace and strictly faster. Use it as a pre-filter **before** running pytest (pytest buries `ImportError` deep in collection errors). Use the import name from Step 0:
```bash
cd /tmp && python -c "from <pkg> import (A, B, C, D, E)"
```
For larger surfaces, fall back to a verification script (e.g., `/tmp/verify_public_api.py`) that imports **every** public module, class, and function path explicitly mentioned in the specification. **Use the real package name from Step 0**, for example:
```python
# /tmp/verify_public_api.py  — replace <pkg> with the REAL package name!
from <pkg> import SomePublicClass
from <pkg>.module import documented_function
from <pkg>.subpackage import AnotherClass

# Verify callability
assert callable(documented_function), "documented_function is not callable"

# Verify instantiation with minimal valid arguments
obj = SomePublicClass()

print("All public API imports and basic checks passed!")
```

If the spec shows any function being CALLED with keyword args, also add a mock-based call-style check — mock-based tests in the evaluator will fail otherwise:
```python
from unittest.mock import patch
with patch.object(SomePublicClass, '_internal_call') as m:
    SomePublicClass().documented_method(arg='value')
    # match the EXACT call form shown in the spec:
    m.assert_called_with(endpoint='/x', mode='GET')
```

**Step 3: Run the verification (from `/tmp`, not from the source dir)**
```bash
cd /tmp && python verify_public_api.py
```

**Step 4: Verify package-level exports**
Ensure that the package's `__init__.py` exports all top-level names documented in the specification. If the spec says `from <pkg> import X`, then `X` must be importable from `<pkg>` directly.

**Step 5: Verify CLI entry points (if applicable)**
If the specification defines CLI entry points (e.g., console_scripts in setup.py), run `<command> --help` to verify they are registered and functional.

**Step 6: Functional smoke test**
Go beyond import checks — write a few basic functional assertions to verify that core methods produce correct results (not just that they exist). For example, if the spec says a `parse()` function returns a list of tokens, call it with a simple input and assert the output structure is correct. This catches issues where imports pass but the implementation logic is broken.

When to use which tool:
- **`python -c "..."`** for most checks: import, attribute lookup, single function call, return-value comparison. Fast iteration; sufficient for most projects.
- **A small `pytest` file** is recommended for things that need pytest to actually load: pytest plugin entry points (especially if your project IS a pytest plugin like `pytest-xxx` — the plugin only registers when pytest runs), `conftest.py` you wrote, or fixtures you wrote.

Focus on verifying your implementation — don't let test-writing substitute for implementation work.

**RULES:**
* If **ANY** import, instantiation, or smoke test fails in Steps 2-6, you MUST fix the issue before finishing. Common causes:
    - Missing `__init__.py` files
    - Missing re-exports in `__init__.py` (e.g., `from .module import PublicClass`)
    - Missing internal `_helper` symbols that official tests may import directly (white-box testing — see Phase 2 guidance)
    - Typos in module/class/function names that don't match the spec
    - Circular imports
    - Missing dependencies in `setup.py`
    - Using a placeholder directory name (e.g. `target_repo/`) instead of the import name from Phase 1, Step 1
</FINISH_GATE>

**Phase 4: Submission**
* **No git operations needed — you don't need to perform any git operations.** The evaluator captures your file changes automatically by diffing the workspace against the baseline commit when you call `finish`. Do **NOT** run any git command (`git add`, `git commit`, `git config user.*`, `git status`, `git diff`, `git checkout`, etc.) — manual git operations have **zero** effect on your score and only waste turns.
* Once the verification passes and you are confident in your implementation:
    1. Delete any temporary verification scripts (`/tmp/verify_public_api.py`, `verify_implementation.py`) to keep the repo clean.
    2. Ensure `setup.py` is configured correctly according to <DEPENDENCY_RULES>.
    3. Submit your work using the "finish" tool.

<ANTI_CHEAT_CONSTRAINT>
**ABSOLUTE RULE: DO NOT DOWNLOAD THE TARGET PACKAGE'S SOURCE CODE FROM THE INTERNET.** You MUST implement the target project from scratch based solely on the specification document above. It is a HARD VIOLATION to look up, download, install, clone, or copy the target package's source code from any external source — this includes PyPI, GitHub, GitLab, Bitbucket, Codeberg, Anaconda, conda-forge, personal mirrors, company-internal mirrors, Google/Bing cache, the Wayback Machine, any local wheel/sdist cache on disk, or any other channel.

Specifically, the following are **FORBIDDEN** and will be auto-blocked:

* Package managers pulling the target: `pip install <target>`, `pip3.X install <target>`, `pip download <target>`, `pip show <target>`, `pip wheel <target>`, `python -m pip install <target>`, `pipx install <target>`, `pipenv install <target>`, `uv pip install <target>`, `uv add <target>`, `uvx <target>`, `poetry add <target>`, `conda/mamba/micromamba install <target>`, `pdm add <target>`, `hatch env install <target>`, `easy_install <target>` — in every path-prefixed / sudo-prefixed / env-prefixed / versioned form.
* VCS: `git clone` or `git submodule add` pointing at the target repo (anywhere: github.com, gitlab.com, bitbucket.org, codeberg.org, any mirror).
* Raw downloads: `curl`/`wget`/`aria2c`/`http`/`httpie`/`lwp-download`/`fetch` hitting PyPI-family hosts (`pypi.org`, `files.pythonhosted.org`, `pypi.python.org`, `test.pypi.org`, `pythonhosted.org`, `pypistats.org`, `conda.anaconda.org`, `anaconda.org`, `repo.anaconda.com`) or any URL containing the target name.
* Python introspection: `python -c 'import <target>; inspect.getsource(<target>)'`, reading `<target>.__file__`, or any `importlib`/`pkgutil`/`get_data` trick to dump the installed upstream source.
* Extracting on-disk artefacts: `unzip <target>*.whl`, `tar xf <target>*.tar.gz`, etc.

The following are **ALLOWED**:

* Installing *third-party* dependencies your own implementation needs (e.g. `pip install requests`, `pip install numpy`).
* Running `pip install -e .` on the code you yourself wrote inside the workspace.
* Reading any file already present in the workspace at the start of your session — **except `README.md`**, whose full spec is already inlined in this prompt and which is in the `<DO_NOT_MODIFY>` set above.

Violations are detected automatically by the `execute_bash` tool's security policy and will be blocked — the blocked command returns an error instead of running. Repeated violations will not succeed; do not try to obfuscate (variable-name rewrites, base64 decode pipes, curl into a random-named file, etc.) — the blocklist covers name variants (`Pkg` / `pkg` / `pkg-name` / `pkg_name` / `PKG_NAME`) and all common package-manager spellings.
</ANTI_CHEAT_CONSTRAINT>
"""


# ── Task-level registry ──────────────────────────────────────────────────────

USER_PROMPTS: dict[str, str] = {
    "denovoswe_doc2repo":            DENOVOSWE_DOC2REPO_PROMPT,
    "denovoswe_doc2repo_v2":         DENOVOSWE_DOC2REPO_PROMPT_V2,
}
