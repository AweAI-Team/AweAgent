"""SearchSWE prompt system.

Public API:
    resolve_prompt_keys  — Route (dataset_id, task_type, search) to prompt keys.
    get_system_prompt    — Retrieve a system prompt by key.
    get_user_prompt      — Retrieve a user prompt template by key.
    NO_TOOL_CALL_PROMPT  — Reminder sent when LLM returns no tool calls.
"""

from awe_agent.scaffold.search_swe.prompts.config import (
    resolve_from_task_info,
    resolve_prompt_keys,
)
from awe_agent.scaffold.search_swe.prompts.system import (
    NO_TOOL_CALL_PROMPT,
    get_system_prompt,
)
from awe_agent.scaffold.search_swe.prompts.user import get_user_prompt

__all__ = [
    "NO_TOOL_CALL_PROMPT",
    "get_system_prompt",
    "get_user_prompt",
    "resolve_from_task_info",
    "resolve_prompt_keys",
]
