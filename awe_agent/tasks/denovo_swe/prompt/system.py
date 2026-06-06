"""DeNovoSWE system prompt templates.

Reuses BeyondSWE system prompts — identical agent persona and guidelines.
"""

from __future__ import annotations

from awe_agent.tasks.beyond_swe.prompt.system import (
    SEARCH_SYSTEM_PROMPT_BEYONDSWE,
    SYSTEM_PROMPT_BEYONDSWE,
)

# DeNovoSWE reuses the same system prompts as BeyondSWE doc2repo.
SYSTEM_PROMPTS: dict[str, str] = {
    "denovoswe":        SYSTEM_PROMPT_BEYONDSWE,
    "search_denovoswe": SEARCH_SYSTEM_PROMPT_BEYONDSWE,
}
