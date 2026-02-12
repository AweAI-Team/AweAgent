"""User prompt templates for all task types and search modes.

Each template is a format string with placeholders for instance-specific data.
Templates are registered in :data:`USER_PROMPTS` by key, referenced from
the route table in ``config.py``.

Template variables:
    workspace_dir      — Path to the repository root (e.g. "/workspace")
    problem_statement  — Issue description text
    base_commit        — Base commit hash
    workspace_tree     — Output of ``tree`` command (doc2repo only)
    installed_packages — Output of ``pip freeze`` (doc2repo only)
    REPO_DOCUMENT      — Specification document content (doc2repo only)
"""

from __future__ import annotations

# ── SWE-Bench ────────────────────────────────────────────────────────────────
# Re-exported from the task module for centralized access.
# The actual templates live in tasks/swe_bench/prompts.py — we reference a
# simplified version here so the prompt system is self-contained.

SWE_BENCH_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a code repository in `{workspace_dir}`. Consider the following issue:

<issue>
{problem_statement}
</issue>

Please implement the necessary changes to resolve this issue.

**Important notes:**
- I've already taken care of all changes to any test files. You do NOT need to \
modify any tests.
- The development environment is already set up.
- Make minimal changes to non-test files in `{workspace_dir}`.

Follow this workflow to solve the issue:

1. **READING**: Carefully read the issue. Identify the core problem, key error \
messages, and affected components. Reword it in your own terms.

2. **RUNNING**: Explore the repo structure. Check README or setup docs. Verify \
the environment works by running existing tests or simple commands.

3. **EXPLORATION**: Use grep/find to locate relevant source files. Identify the \
root cause and the minimal set of files that need to change.

4. **REPRODUCTION**: Write a minimal script that reproduces the bug. Confirm \
the issue exists before attempting a fix.

5. **ANALYSIS**: Before coding, explicitly state:
   - What the problem is
   - Where in the code it occurs
   - What the correct behavior should be
   - Your proposed fix

6. **IMPLEMENTATION**: Make focused, minimal changes. Do not refactor unrelated \
code or change coding style.

7. **VERIFICATION**: Run the reproduction script again. Run related test suites. \
Use `git diff` to review your changes are correct and complete.

Important: Your task is complete when you have made the code changes. You do not \
need to create a pull request or commit."""


# ── BeyondSWE: Non-search prompts ───────────────────────────────────────────

DOC2REPO_PROMPT = """\
I need you to implement a software repository from scratch based on a strict \
architectural specification.

### 1. Context & Environment
* **Workspace Directory**: `{workspace_dir}`
* **Current File Structure**:
```text
{workspace_tree}
```
* **Installation Status**: The `target_repo` package is already installed in \
editable mode (`pip install -e .`). Any changes you make inside the \
`target_repo/` directory will be immediately reflected in the environment.

### 2. Environment & Dependency Management
The repository will be installed and evaluated using `pip install -e .`. You \
must manage dependencies strictly via `setup.py`.

<CURRENT_ENVIRONMENT>
The following packages are **ALREADY INSTALLED** in the environment. You can use \
them directly without reinstalling:
```text
{installed_packages}
```
*(Note: Do not assume any other packages exist unless listed above.)*
</CURRENT_ENVIRONMENT>

<DEPENDENCY_RULES>
1. **Setup.py is King**: You MUST list all necessary runtime dependencies in `setup.py`.
2. **Avoid Redundancy**: If a package is already in **<CURRENT_ENVIRONMENT>**, \
do **NOT** add it to `setup.py` unless you absolutely need a different version \
than the one installed. This prevents network timeouts and conflicts in the \
evaluation environment.
3. **Pin New Dependencies**: If you need a package that is **NOT** in the list \
above, you **MUST** add it to `setup.py` and **explicitly pin the version**.
4. **Ignore requirements.txt**: Do not create or update `requirements.txt`. \
Only modify `setup.py`.
</DEPENDENCY_RULES>

### 3. The Specification (`repo_document.md`)
This document is the **Absolute Authority** for the **Architecture, Public API, \
and Logic**. You must implement the repository exactly as described here.
```markdown
{REPO_DOCUMENT}
```

---

### 4. Task Instructions

**Phase 1: Analysis & Research (Critical Discipline)**
* **Analyze the Spec**: Read `repo_document.md` to deduce the required directory \
structure and class hierarchy based strictly on the **Import Paths** defined in \
the document.
* **Check Dependencies**: Compare spec requirements against <CURRENT_ENVIRONMENT>. \
Decide which ones need to be added to `setup.py` and which ones are already present.

**Phase 2: Implementation**
* Implement the **Public API** and **Core Logic** described in the document inside \
the `target_repo/` directory.
* **Strict Constraints**:
    * Function signatures (arguments, types, return values) MUST match the \
document exactly.
    * Ensure all imports use relative imports (e.g., `from . import utils`) or \
absolute imports starting with `target_repo`.
* **Internal Implementation Flexibility**:
    * The document lists the **Core Public API**. You are encouraged to create \
necessary **private helper functions**, **internal constants**, or **utility \
classes** to support the logic.
    * Keep internal helpers private (prefix with `_`) where appropriate.
    * You may create new utility files inside the `target_repo/` directory if \
the logic requires it, but keep the structure clean and logical.
    * Do NOT change the signature of the documented Public APIs.

**Phase 3: Verification (Self-Correction)**
* Since this is a clean-room implementation, **NO existing tests are provided**.
* You are responsible for verifying your own code:
    * Create a standalone script (e.g., `verify_implementation.py`) to import \
your new classes/functions and assert they behave as documented.
    * OR write simple `pytest` cases to check critical logic.
* **Goal**: Ensure your implementation runs without errors and matches the spec \
before you finish.

**Phase 4: Submission**
* Once you are confident:
    1. Delete any temporary verification scripts (`verify_implementation.py`) \
to keep the repo clean.
    2. Ensure `setup.py` is configured correctly according to <DEPENDENCY_RULES>.
    3. Submit your work using the "finish" tool."""


CROSS_REPO_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir}. \
Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the \
<issue_description>. This means you DON'T have to modify the testing logic or \
any of the tests in any way!
Your task is to make the minimal changes to non-test files in the {workspace_dir} \
directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices \
or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, \
and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing \
the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.3 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible \
solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error \
messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix \
the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce \
and verify the issue.
   4.1 Look at existing test files in the repository to understand the test \
format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen \
solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't \
broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your \
changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and analysis. It's fine if your \
thinking process is lengthy - quality and completeness are more important \
than brevity."""


REFACTOR_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir}. \
Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <issue_description> are met?
Your task is to make the minimal changes to non-test files in the {workspace_dir} \
directory to ensure the <issue_description> is satisfied.

IMPORTANT CONSTRAINT: You are strictly forbidden from attempting to resolve \
issues by downgrading dependencies. Your goal is to refactor the source code \
to make it compatible with the currently installed library versions.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices \
or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, \
and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing \
the issue.

Phase 2. RUNNING: verify the pre-installed environment and run tests
   2.1 The environment is ALREADY INSTALLED with newer versions of dependencies.
   2.2 If the tests fail to import the project modules, you may install the \
current package in editable mode (e.g. `pip install --no-deps -e .`).
   2.3 Iterate and figure out how to run the tests.

Phase 3. EXPLORATION: find the files that are related to the problem and possible \
solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error \
messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, identify all components or packages in \
the repository that are affected by the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce \
and verify the issue.
   4.1 Look at existing test files in the repository to understand the test \
format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen \
solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests for the modified code AND any downstream components \
that depend on it. (e.g., if you modify a core library, check if dependent \
packages pass their tests).

8. FINAL REVIEW: Carefully re-read the problem description and compare your \
changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your \
thinking process is lengthy - quality and completeness are more important \
than brevity."""


DOMAIN_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir}. \
Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the \
<issue_description>. This means you DON'T have to modify the testing logic or \
any of the tests in any way!
Your task is to make the minimal changes to non-test files in the {workspace_dir} \
directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices \
or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, \
and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing \
the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme to understand the setup, BUT assume the pre-installed \
environment is the source of truth.
   2.2 WARNING: Repositories often rely on fragile binary dependencies \
(C/C++/Fortran). DO NOT run `pip install` or upgrade packages (numpy, pandas, \
gdal, etc.) unless strictly necessary. You might break the environment.
   2.3 Iterate and figure out how to run the tests. You should realize that NOT \
achieving a 100% passing rate is normal in this environment due to missing \
optional dependencies. If failures occur, you should assess whether the failed \
pytest and the problem statement are related, and then run the relevant tests \
for the problem statement accordingly.

Phase 3. EXPLORATION: find the files that are related to the problem and possible \
solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error \
messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix \
the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce \
and verify the issue.
   4.1 Look at existing test files in the repository to understand the test \
format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 If the issue requires external services (e.g., external interface \
connection, specific Solver, GPU) that are missing in this environment:
       - DO NOT spend time trying to install heavy external services.
       - Mock the external service in the reproduction script.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen \
solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't \
broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your \
changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass. \
Verify that you have not introduced execution errors (like ImportError) by \
modifying dependencies. The code must run in the original environment state \
as much as possible.

Be thorough in your exploration, testing, and analysis. It's fine if your \
thinking process is lengthy - quality and completeness are more important \
than brevity."""


# ── BeyondSWE: Search prompts ───────────────────────────────────────────────

SEARCH_DOC2REPO_PROMPT = """\
I need you to implement a software repository from scratch based on a strict \
architectural specification.

### 1. Context & Environment
* **Workspace Directory**: `{workspace_dir}`
* **Current File Structure**:
```text
{workspace_tree}
```
* **Installation Status**: The `target_repo` package is already installed in \
editable mode (`pip install -e .`). Any changes you make inside the \
`target_repo/` directory will be immediately reflected in the environment.

### 2. Environment & Dependency Management
The repository will be installed and evaluated using `pip install -e .`. You \
must manage dependencies strictly via `setup.py`.

<CURRENT_ENVIRONMENT>
The following packages are **ALREADY INSTALLED** in the environment.
```text
{installed_packages}
```
*(Note: Do not assume any other packages exist unless listed above.)*
</CURRENT_ENVIRONMENT>

<DEPENDENCY_RULES>
1. **Setup.py is King**: You MUST list all necessary runtime dependencies in \
`setup.py`.
2. **Avoid Redundancy**: If a package is already in **<CURRENT_ENVIRONMENT>**, \
do **NOT** add it to `setup.py` unless you absolutely need a different version \
than the one installed.
3. **Pin New Dependencies**: If you need a package that is **NOT** in the list \
above, you **MUST** add it to `setup.py` and **explicitly pin the version**.
4. **Ignore requirements.txt**: Do not create or update `requirements.txt`. \
Only modify `setup.py`.
</DEPENDENCY_RULES>

### 3. The Specification (`repo_document.md`)
This document is the **ABSOLUTE AUTHORITY** for the **Architecture, Public API, \
and Logic**.
**WARNING**: You must implement the repository **exactly** as described here.

```markdown
{REPO_DOCUMENT}
```

---

### 4. Task Instructions

**Phase 1: Analysis & Constraints (Spec Compliance)**
* **Analyze the Spec**: Read `repo_document.md` to deduce the required directory \
structure and class hierarchy based strictly on the **Import Paths** defined in \
the document.
* **Check Dependencies**: Compare spec requirements against <CURRENT_ENVIRONMENT>. \
Decide which ones need to be added to `setup.py` and which ones are already present.
* **Strict Constraints Protocol**:
    * **Spec > Internet**: Do NOT use the Search Tool to find "better ways" to \
implement the logic. The logic described in the document is the ground truth.
    * **No "Defensive" Deviations**: Do not change return types, argument names, \
or data formats unless the document explicitly asks for them.
    * **Search Usage Limit**: Use the Search Tool **ONLY** if you encounter:
        1. Syntactical errors related to specific library versions (e.g., \
`pydantic` v1 vs v2 differences).
        2. Missing imports or dependencies that are implied but not listed.
    **MANDATORY Deep Reading**: If a search result looks promising, you **MUST** \
use the **LinkSummary** tool to fetch and read the actual page content. Treat \
search snippets merely as relevance indicators. **Never write code based solely \
on search snippets.**
    * **Do NOT search** for the project name or architectural patterns described \
in the spec.

**Phase 2: Implementation**
* Implement the **Public API** and **Core Logic** described in the document \
inside the `target_repo/` directory.
* **Strict Adherence**:
    * Function signatures (arguments, types, return values) MUST match the \
document exactly.
    * Ensure all imports use relative imports (e.g., `from . import utils`) or \
absolute imports starting with `target_repo`.
* **Coding Strategy**:
    * **Minimalism**: Implement exactly what is requested. Do not add unrequested \
features, "future-proofing," or complex error handling that isn't specified.
    * **Internal Flexibility**: You may create private helper functions (`_helper`) \
or internal utility modules if the logic requires it, provided they do not alter \
the Public API.

**Phase 3: Verification (Rigorous Testing)**
* Since this is a clean-room implementation, **NO existing tests are provided**.
* **Mandatory Pytest**: You MUST create a standard test file (e.g., \
`tests/test_verify.py`) and run it using `pytest`.
    * **Do NOT use a simple print-script**.
    * Write explicit `assert` statements to verify:
        1. Class signatures and method existence.
        2. Return types and data formats.
        3. Critical logic paths (e.g., hash chaining consistency, graph \
connectivity).
* **Refinement**: If `pytest` fails, analyze the error. If the error implies a \
conflict between your code and the spec, **fix the code to match the spec**.

**Phase 4: Submission**
* Once you are confident:
    1. Delete the temporary `tests/` directory to keep the repo clean.
    2. Ensure `setup.py` is configured correctly according to <DEPENDENCY_RULES>.
    3. Submit your work using the "finish" tool."""


SEARCH_CROSS_REPO_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir}. \
Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the \
<issue_description>. This means you DON'T have to modify the testing logic or \
any of the tests in any way!
Your task is to make the minimal changes to non-test files in the {workspace_dir} \
directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices \
or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, \
and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing \
the issue.
   1.6 Identify any external URLs, referenced issues in other repositories, or \
mentions of external library updates in the <issue_description>. List these \
clearly as targets for the search phase.

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.3 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible \
solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error \
messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix \
the issue.
   3.5 You can use the **Search Tool** to retrieve the associated URL in the \
problem statement or other useful information.
   3.6 **Attention**:
       - **MANDATORY Deep Reading**: If a search result looks promising, you \
**MUST** use the **LinkSummary** tool to fetch and read the actual page content. \
Treat search snippets merely as relevance indicators. Snippets are often \
incomplete or outdated. **Never write code based solely on search snippets.**

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce \
and verify the issue.
   4.1 Look at existing test files in the repository to understand the test \
format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is, specifically citing any external \
specifications or dependency changes found during the search phase if relevant.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen \
solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't \
broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your \
changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your \
thinking process is lengthy - quality and completeness are more important \
than brevity."""


SEARCH_REFACTOR_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir}. \
Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <issue_description> are met?
Your task is to make the minimal changes to non-test files in the {workspace_dir} \
directory to ensure the <issue_description> is satisfied.

IMPORTANT CONSTRAINT: You are strictly forbidden from attempting to resolve \
issues by downgrading dependencies. Your goal is to refactor the source code to \
make it compatible with the currently installed library versions.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices \
or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, \
and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing \
the issue.
   1.6 Identify specific external libraries causing warnings or errors. Analyze \
if the error implies a missing dependency or a breaking API change. List specific \
search queries targeting "migration guides", "changelogs", or "release notes" \
for these libraries.

Phase 2. RUNNING: verify the pre-installed environment and run tests
   2.1 The environment is ALREADY INSTALLED with newer versions of dependencies.
   2.2 If the tests fail to import the project modules, you may install the \
current package in editable mode (e.g. `pip install --no-deps -e .`).
   2.3 Iterate and figure out how to run the tests.

Phase 3. EXPLORATION: find the files that are related to the problem and possible \
solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error \
messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, identify all components or packages in \
the repository that are affected by the issue.
   3.5 You can use the **Search Tool** to find the Migration Guide or Changelog \
for the upgraded library.
       - Goal: Determine if a feature was removed, renamed, or requires new \
dependencies.
       - Verification: Ensure the solution applies to the currently installed \
version in the environment.
   3.6 **Attention**:
       - **MANDATORY Deep Reading**: If a search result looks promising, you \
**MUST** use the **LinkSummary** tool to fetch and read the actual page content. \
Treat search snippets merely as relevance indicators. Snippets are often \
incomplete or outdated. **Never write code based solely on search snippets.**

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce \
and verify the issue.
   4.1 Look at existing test files in the repository to understand the test \
format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is. Cite the official documentation or \
migration guide found during the search phase to justify your decision.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen \
solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests for the modified code AND any downstream components \
that depend on it. (e.g., if you modify a core library, check if dependent \
packages pass their tests).

8. FINAL REVIEW: Carefully re-read the problem description and compare your \
changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your \
thinking process is lengthy - quality and completeness are more important \
than brevity."""


SEARCH_DOMAIN_PROMPT = """\
<uploaded_files>
{workspace_dir}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir}. \
Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the \
requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the \
<issue_description>. This means you DON'T have to modify the testing logic or \
any of the tests in any way!
Your task is to make the minimal changes to non-test files in the {workspace_dir} \
directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices \
or conventions in them.
   1.2 Highlight message errors, method names, variables, file names, stack traces, \
and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing \
the issue.
   1.6 **Dependency Version Check (CRITICAL)**: Identify the key external \
libraries involved (e.g., pandas, scipy, internal solvers). Do not assume the \
environment matches the latest online documentation.
   1.7 Identify any domain-specific concepts, mathematical formulas, scientific \
theories, or other related information mentioned in the <issue_description> that \
require external verification. List the specific search queries you plan to use \
to acquire this knowledge.

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme to understand the setup, BUT assume the pre-installed \
environment is the source of truth.
   2.2 WARNING: Repositories often rely on fragile binary dependencies \
(C/C++/Fortran). DO NOT run `pip install` or upgrade packages (numpy, pandas, \
gdal, etc.) unless strictly necessary. You might break the environment.
   2.3 Iterate and figure out how to run the tests. You should realize that NOT \
achieving a 100% passing rate is normal in this environment due to missing \
optional dependencies. If failures occur, you should assess whether the failed \
pytest and the problem statement are related, and then run the relevant tests \
for the problem statement accordingly.

Phase 3. EXPLORATION: find the files that are related to the problem and possible \
solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error \
messages.
   3.2 Identify all files related to the problem statement.
   3.3 From the possible file locations, select the most likely location to fix \
the issue.
   3.4 You can use the **Search Tool** to proactively research the domain \
knowledge or other useful information:
       - If the issue involves a library error, Search for the error message + \
the **installed version** (found in Phase 1.6).
       - If the issue involves an API change (e.g., "Xpress 9.4"), explicitly \
Search for **"Release Notes"** or **"Migration Guide"**.
       - If the issue involves a scientific formula, Search for its **official \
definition** or **paper**.
   3.5 **Attention**:
       - **MANDATORY Deep Reading**: If a search result looks promising, you \
**MUST** use the **LinkSummary** tool to fetch and read the actual page content. \
Treat search snippets merely as relevance indicators. Snippets are often \
incomplete or outdated. **Never write code based solely on search snippets.**

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce \
and verify the issue.
   4.1 Look at existing test files in the repository to understand the test \
format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 If the issue requires external services (e.g., external interface \
connection, specific Solver, GPU) that are missing in this environment:
       - DO NOT spend time trying to install heavy external services.
       - Mock the external service in the reproduction script.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is, specifically citing official \
documentation, API references, or scientific principles found during the search \
phase to justify your fix.
   5.2 State clearly where the problem is located. Explain **WHY** your proposed \
fix is correct.
       - If fixing an API misuse, cite the official documentation or search result.
       - If fixing a formula, cite the physical definition found.
       - **Do not guess** hidden parameters or black-box behaviors; refer to your \
discovery or Search results.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen \
solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't \
broken anything.
       - Ensure you haven't broken other features.
       - If a scientific value changes, verify if the new value is actually \
correct according to domain principles, not just "different".

8. FINAL REVIEW: Carefully re-read the problem description and compare your \
changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass. \
Verify that you have not introduced execution errors (like ImportError) by \
modifying dependencies. The code must run in the original environment state \
as much as possible.

Be thorough in your exploration, testing, and reasoning. It's fine if your \
thinking process is lengthy - quality and completeness are more important \
than brevity."""


# ── Registry ─────────────────────────────────────────────────────────────────

USER_PROMPTS: dict[str, str] = {
    # SWE-Bench
    "swe_bench":           SWE_BENCH_PROMPT,

    # BeyondSWE — non-search
    "doc2repo":            DOC2REPO_PROMPT,
    "cross_repo":          CROSS_REPO_PROMPT,
    "refactor":            REFACTOR_PROMPT,
    "domain":              DOMAIN_PROMPT,

    # BeyondSWE — search-enabled
    "search_doc2repo":     SEARCH_DOC2REPO_PROMPT,
    "search_cross_repo":   SEARCH_CROSS_REPO_PROMPT,
    "search_refactor":     SEARCH_REFACTOR_PROMPT,
    "search_domain":       SEARCH_DOMAIN_PROMPT,
}


def get_user_prompt(key: str) -> str:
    """Get a user prompt template by key. Raises KeyError for unknown keys."""
    if key not in USER_PROMPTS:
        raise KeyError(
            f"Unknown user prompt key: {key!r}. "
            f"Available: {list(USER_PROMPTS.keys())}"
        )
    return USER_PROMPTS[key]
