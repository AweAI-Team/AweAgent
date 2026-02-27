# Scaffold Prompt System

This directory contains the prompt routing and registration system for the `search_swe` scaffold. It merges prompt registries from all task modules into a unified lookup table.

## Architecture Overview

```
Task Layer (owns prompts)              Scaffold Layer (owns registry)
========================               ============================

beyond_swe/prompt/system.py            scaffold/search_swe/prompts/
  SYSTEM_PROMPTS = {                     system.py
    "beyondswe": ...,          ──────>     _merge(BEYOND_SWE_SYSTEM_PROMPTS)
    "search_beyondswe": ...,               _merge(SCALE_SWE_SYSTEM_PROMPTS)
    "search_domainfix": ...,               => merged SYSTEM_PROMPTS
  }
                                         user.py
beyond_swe/prompt/user.py                 _merge(BEYOND_SWE_USER_PROMPTS)
  USER_PROMPTS = {                         _merge(SCALE_SWE_USER_PROMPTS)
    "doc2repo": ...,           ──────>     => merged USER_PROMPTS
    "crossrepo": ...,
    ...                                  config.py
  }                                       PROMPT_ROUTES = {
                                            (dataset_id, task_type, search)
scale_swe/prompt.py                           => (system_key, user_key)
  SYSTEM_PROMPTS = {                      }
    "openhands": ...,          ──────>
  }
  USER_PROMPTS = {
    "scaleswe": ...,           ──────>
  }
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Route table: maps `(dataset_id, task_type, search)` to `(system_key, user_key)` |
| `system.py` | Merges `SYSTEM_PROMPTS` dicts from all tasks, provides `get_system_prompt()` |
| `user.py` | Merges `USER_PROMPTS` dicts from all tasks, provides `get_user_prompt()` |

## How Prompt Routing Works

The route table in `config.py` is the single source of truth for prompt selection:

```python
PROMPT_ROUTES = {
    # (dataset_id, task_type, search_enabled): (system_key, user_key)
    ("beyond_swe", "doc2repo", False):  ("beyondswe", "doc2repo"),
    ("beyond_swe", "doc2repo", True):   ("search_beyondswe", "search_doc2repo"),
    ("scale_swe",  None,       False):  ("openhands", "scaleswe"),
    ...
}
```

`resolve_prompt_keys(dataset_id, task_type, search)` resolves in order:
1. **Exact match**: `(dataset_id, task_type, search)`
2. **Wildcard**: `(dataset_id, None, search)` — matches any task_type
3. **Default fallback**: `("beyondswe", "domainfix")`

### Two Consumers

The resolved keys are consumed by two different components:

- **Task** uses `user_key` — calls `get_user_prompt(user_key)` to format the instance-specific prompt
- **Agent (SearchSWEAgent)** uses `system_key` — calls `get_system_prompt(system_key)` to set the agent's system prompt

```
                  resolve_prompt_keys()
                 /                      \
           system_key               user_key
              |                         |
    SearchSWEAgent                  Task.get_prompt()
    .get_system_prompt()            .get_user_prompt()
```

## Adding a New Task's Prompts

When you create a new task and want it to work with the `search_swe` scaffold, follow these 3 steps:

### Step 1: Add route entry (`config.py`)

```python
PROMPT_ROUTES = {
    ...
    # ── YourTask ───────────────────────────────────────────────────
    ("your_dataset", None, False): ("your_system_key", "your_user_key"),
}
```

Use `task_type=None` (wildcard) if your task has no sub-types. Add multiple entries if you support both search and non-search modes.

### Step 2: Import and merge system prompts (`system.py`)

```python
from awe_agent.tasks.your_task.prompt import (
    SYSTEM_PROMPTS as _YOUR_TASK_SYSTEM_PROMPTS,
)

# Add at the end of the merge section:
_merge(_YOUR_TASK_SYSTEM_PROMPTS, "your_task")
```

### Step 3: Import and merge user prompts (`user.py`)

```python
from awe_agent.tasks.your_task.prompt import (
    USER_PROMPTS as _YOUR_TASK_USER_PROMPTS,
)

# Add at the end of the merge section:
_merge(_YOUR_TASK_USER_PROMPTS, "your_task")
```

## Conflict Detection

The `_merge()` function raises `ValueError` if a duplicate key is detected:

```python
def _merge(source: dict[str, str], label: str) -> None:
    for key, prompt in source.items():
        if key in SYSTEM_PROMPTS:
            raise ValueError(
                f"Duplicate system prompt key {key!r} from {label}. "
                f"Already registered: {list(SYSTEM_PROMPTS.keys())}"
            )
        SYSTEM_PROMPTS[key] = prompt
```

This ensures prompt keys are globally unique and catches accidental collisions at import time rather than at runtime.

## Current Registry

### System Prompts

| Key | Source Task | Description |
|-----|-----------|-------------|
| `beyondswe` | beyond_swe | Standard engineering prompt |
| `search_beyondswe` | beyond_swe | Engineering + search capabilities |
| `search_domainfix` | beyond_swe | Domain-specific search (scientific, versioned APIs) |
| `openhands` | scale_swe | OpenHands-style system prompt |

### User Prompts

| Key | Source Task | Description |
|-----|-----------|-------------|
| `doc2repo` | beyond_swe | Build repo from spec |
| `crossrepo` | beyond_swe | Cross-repo issue resolution |
| `depmigrate` | beyond_swe | Dependency migration |
| `domainfix` | beyond_swe | Domain-specific bug fix |
| `search_doc2repo` | beyond_swe | Doc2Repo with search tools |
| `search_crossrepo` | beyond_swe | CrossRepo with search tools |
| `search_depmigrate` | beyond_swe | DepMigrate with search tools |
| `search_domainfix` | beyond_swe | DomainFix with search tools |
| `scaleswe` | scale_swe | SWE-bench standard format |
