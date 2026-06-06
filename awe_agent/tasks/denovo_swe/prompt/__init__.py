"""DeNovoSWE prompt templates.

System and user prompts for DeNovoSWE doc2repo task in both standard
and search-enabled modes.
"""

from awe_agent.tasks.denovo_swe.prompt.system import SYSTEM_PROMPTS
from awe_agent.tasks.denovo_swe.prompt.user import USER_PROMPTS

__all__ = [
    "SYSTEM_PROMPTS",
    "USER_PROMPTS",
]
