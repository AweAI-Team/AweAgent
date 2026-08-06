"""DeNovoSWE prompt templates.

DeNovoSWE reuses the BeyondSWE **system** prompt (same agent persona) — see the
route table in ``scaffold/search_swe/prompts/config.py`` — so this package only
ships the doc2repo **user** prompt (v1 + v2). Search mode is not used.
"""

from aweagent.tasks.denovo_swe.prompt.user import USER_PROMPTS

__all__ = [
    "USER_PROMPTS",
]
