"""NL2Repo user prompt templates.

Ported **verbatim** from the original NL2RepoBench OpenHands launcher
(``openhands/openhands_app.py`` lines 200-201).  This is a single-line user
prompt with no placeholders — keeping it byte-identical to the upstream
benchmark is the only way to guarantee that the LLM sees the exact same
instruction string and that downstream evaluation numbers stay comparable.
"""

from __future__ import annotations

# ── NL2Repo: Non-search prompt ──────────────────────────────────────────────
#
# Source: ``openhands/openhands_app.py``
#
#     command = ['python', '-m', 'openhands.core.main',
#                '--config-file=/custom/path/config.toml', '-t',
#                'According to the start.md in the workspace, implement '
#                'the entire project as per the requirements specified in '
#                'the document, ensuring that the final product can be '
#                'directly run in the current directory. The running '
#                'requirements should comply with the <API Usage Guide> '
#                'section of the document. Please complete this task step '
#                'by step.']
#
# Do NOT add markdown sections, phases, or extra hints — that would diverge
# from upstream and silently change benchmark scores.

NL2REPO_PROMPT = (
    "According to the start.md in the workspace, implement the entire "
    "project as per the requirements specified in the document, ensuring "
    "that the final product can be directly run in the current directory. "
    "The running requirements should comply with the <API Usage Guide> "
    "section of the document. Please complete this task step by step.\n"
    "\n"
    "IMPORTANT ANTI-CHEAT CONSTRAINT: You MUST implement the target "
    "project from scratch based solely on start.md. You are strictly "
    "forbidden from looking up, downloading, installing, cloning, or "
    "copying the target package's source code from any external "
    "source (PyPI, GitHub/GitLab/Bitbucket, Anaconda, personal mirrors, "
    "local caches, or any other channel). Specifically, do NOT run any "
    "form of `pip install <target>`, `pip download <target>`, "
    "`pip show <target>`, `python -m pip install <target>`, `pipx`, "
    "`uv add/install <target>`, `poetry add <target>`, "
    "`conda/mamba install <target>`, `git clone` / `git submodule` of "
    "the target repo, or `curl`/`wget`/`aria2c` against PyPI or the "
    "target repo; and do NOT use `python -c 'import <target>; "
    "inspect.getsource(...)'` or `__file__` tricks to read the "
    "installed upstream source. You may install *third-party* "
    "dependencies your own implementation needs (e.g. `pip install "
    "requests`) and you may run `pip install -e .` on the code you "
    "yourself wrote, but the target package itself must never be "
    "fetched. Violations will be detected automatically by the "
    "execute_bash tool's security policy and blocked."
)


# ── Task-level registry ─────────────────────────────────────────────────────
# Declares which prompt keys this task provides. The scaffold layer merges
# these dicts from all tasks and performs conflict detection — see
# ``awe_agent/scaffold/search_swe/prompts/user.py``.

USER_PROMPTS: dict[str, str] = {
    "nl2repo": NL2REPO_PROMPT,
}
