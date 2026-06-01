"""DeepSearch prompt registry."""

from awe_agent.scaffold.deepsearch.prompts.config import (
    resolve_from_task_info,
    resolve_prompt_keys,
)
from awe_agent.scaffold.deepsearch.prompts.final_answer import (
    NO_FINAL_ANSWER_FALLBACK,
    OMITTED_TOOL_RESULT,
    FinalAnswerPrompts,
    get_final_answer_prompts,
)
from awe_agent.scaffold.deepsearch.prompts.system import (
    NO_TOOL_CALL_PROMPT,
    get_system_prompt,
)
from awe_agent.scaffold.deepsearch.prompts.user import get_user_prompt

__all__ = [
    "FinalAnswerPrompts",
    "NO_FINAL_ANSWER_FALLBACK",
    "NO_TOOL_CALL_PROMPT",
    "OMITTED_TOOL_RESULT",
    "get_final_answer_prompts",
    "get_system_prompt",
    "get_user_prompt",
    "resolve_from_task_info",
    "resolve_prompt_keys",
]
