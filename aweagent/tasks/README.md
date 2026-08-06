# Adding a New Task

This guide explains how to create a new task type in AweAgent, including prompt definitions and evaluation.

## Directory Structure

Each task lives in its own package under `aweagent/tasks/`:

```
aweagent/tasks/
  beyond_swe/           # BeyondSWE benchmark (Doc2Repo, CrossRepo, DepMigrate, DomainFix)
    prompt/
      __init__.py       # Re-exports
      system.py         # System prompt constants + SYSTEM_PROMPTS registry dict
      user.py           # User prompt constants + USER_PROMPTS registry dict
    task.py
    evaluator.py
  scale_swe/            # ScaleSWE benchmark (SWE-bench style)
    prompt.py           # Both system and user prompts + registry dicts
    task.py
    evaluator.py
  your_task/            # <- Your new task
    prompt.py
    task.py
    evaluator.py
    __init__.py
```

## Step-by-Step Guide

### 1. Define Prompts (`your_task/prompt.py`)

Create prompt constants and declare **task-level registry dicts** (`SYSTEM_PROMPTS` and `USER_PROMPTS`). The scaffold layer will import and merge these dicts.

```python
"""YourTask prompt templates."""

from __future__ import annotations

# ── System prompt ────────────────────────────────────────────────────────────

YOUR_TASK_SYSTEM_PROMPT = """You are a Senior Software Engineer...
"""

# ── User prompt ──────────────────────────────────────────────────────────────

YOUR_TASK_USER_PROMPT = """We are addressing the following issue:

--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

The repository is located at `{workspace_dir}`.
"""

# ── Task-level registry ─────────────────────────────────────────────────────
# Declares which keys this task provides. Scaffold layers merge these dicts
# from all tasks and perform conflict detection.

SYSTEM_PROMPTS: dict[str, str] = {
    "your_task": YOUR_TASK_SYSTEM_PROMPT,
}

USER_PROMPTS: dict[str, str] = {
    "your_task": YOUR_TASK_USER_PROMPT,
}
```

**Key rules:**

- Registry keys must be globally unique across all tasks. If two tasks register the same key, the scaffold will raise `ValueError` at import time.
- You can register multiple keys if your task has variants (e.g., `"search_your_task"` for search-enabled mode).
- User prompt templates receive keyword arguments from `Task.get_prompt()` — common variables include `{workspace_dir}`, `{problem_statement}`, `{base_commit}`.

### 2. Implement the Task class (`your_task/task.py`)

Subclass `aweagent.core.task.protocol.Task` and implement required methods:

```python
from aweagent.core.task.protocol import Task
from aweagent.core.task.types import Instance

class YourTask(Task):
    def get_instances(self, instance_ids=None) -> list[Instance]:
        """Load and return task instances."""
        ...

    def get_prompt(self, instance: Instance) -> str:
        """Format the user prompt for a given instance."""
        # Use lazy imports to avoid circular dependency
        from aweagent.scaffold.search_swe.prompts.config import resolve_prompt_keys
        from aweagent.scaffold.search_swe.prompts.user import get_user_prompt

        _, user_key = resolve_prompt_keys(
            dataset_id=self.dataset_id, task_type=None, search=False,
        )
        template = get_user_prompt(user_key)
        return template.format(
            workspace_dir=instance.workdir,
            problem_statement=instance.problem_statement,
        )

    def get_image(self, instance: Instance) -> str:
        """Return the container image for the instance."""
        return instance.image

    def get_setup_commands(self, instance: Instance) -> list[str]:
        """Return shell commands to run before the agent starts."""
        return list(instance.setup_commands)

    def get_task_info(self, instance: Instance) -> dict:
        """Return metadata dict for prompt routing and logging."""
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            ...
        }

    def default_evaluator(self, timeout=None):
        """Return the default evaluator for this task."""
        ...
```

**Important:** Use **lazy imports** inside `get_prompt()` (not at module level) to avoid circular dependencies. The chain `scaffold/prompts/user.py` -> `your_task/prompt.py` -> `your_task/__init__.py` -> `your_task/task.py` -> `scaffold/prompts/user.py` would otherwise create a circular import.

### 3. Implement the Evaluator (`your_task/evaluator.py`)

If your task uses the same F2P/P2P test evaluation as BeyondSWE, inherit from `BeyondSWEEvaluator`:

```python
from aweagent.tasks.beyond_swe.evaluator import BeyondSWEEvaluator

class YourTaskEvaluator(BeyondSWEEvaluator):
    async def run_tests(self, instance, session):
        return await self._eval_beyondswe(instance, session)
```

Otherwise, implement the `Evaluator` protocol directly.

### 4. Register with the Scaffold

After defining prompts and the task class, register them with the scaffold layer. See [`aweagent/scaffold/search_swe/prompts/README.md`](../scaffold/search_swe/prompts/README.md) for details.

In summary, you need to:

1. Add a route in `scaffold/search_swe/prompts/config.py`
2. Import and merge your `SYSTEM_PROMPTS` in `scaffold/search_swe/prompts/system.py`
3. Import and merge your `USER_PROMPTS` in `scaffold/search_swe/prompts/user.py`

### 5. Register the task

Tasks are resolved through the `aweagent.task` registry (mirroring the agent /
evaluator / runtime registries), so there is no hand-written dispatch to edit.

1. Give your task class a `from_config(cls, config)` classmethod that pulls its
   own parameters from the global config (the `Task` base provides a default
   that reads `dataset_id` / `data_file`; override it only if you need more):

   ```python
   @classmethod
   def from_config(cls, config):
       return cls(
           dataset_id=config.task.dataset_id,
           data_file=config.task.data_file,
           # ...your task-specific params, read from config.task.* / config.agent.* ...
       )
   ```

2. Register it in `aweagent/core/task/registry.py` (built-in) and add an entry
   under `[project.entry-points."aweagent.task"]` in `pyproject.toml`
   (so editable installs of downstream packages are discovered too):

   ```toml
   your_task = "aweagent.tasks.your_task.task:YourTask"
   ```

   The shared pipeline (`aweagent/core/task/pipeline.py`) then builds it via
   `task_registry.get(config.task.type).from_config(config)` — the CLI and all
   recipes go through this one path.

### 6. Create Config and Recipe

- Add a config file in `configs/tasks/your_task.yaml`
- Add a recipe in `recipes/your_task/` (see `recipes/scale_swe/` for an example)

## Prompt Template Variables

Common template variables available for user prompts:

| Variable | Source | Description |
|----------|--------|-------------|
| `{workspace_dir}` | `instance.workdir` | Working directory inside the container |
| `{problem_statement}` | `instance.problem_statement` | Issue/task description |
| `{base_commit}` | `instance.base_commit` | Git commit to base the work on |
| `{workspace_tree}` | generated | Directory tree (Doc2Repo only) |
| `{installed_packages}` | generated | pip freeze output (Doc2Repo only) |
| `{REPO_DOCUMENT}` | from data | Repository specification (Doc2Repo only) |

## Existing Tasks

| Task | Dataset ID | Task Types | Prompts |
|------|-----------|------------|---------|
| BeyondSWE | `beyond_swe` | `doc2repo`, `crossrepo`, `depmigrate`, `domainfix` | 8 user + 3 system |
| ScaleSWE | `scale_swe` | (none, always issue-resolving) | 1 user + 1 system |
