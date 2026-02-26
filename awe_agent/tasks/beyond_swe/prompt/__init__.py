"""BeyondSWE prompt templates.

System and user prompts for all BeyondSWE task types (Doc2Repo, CrossRepo,
DepMigrate, DomainFix) in both standard and search-enabled modes.
"""

from awe_agent.tasks.beyond_swe.prompt.system import (
    SEARCH_SYSTEM_PROMPT_BEYONDSWE,
    SEARCH_SYSTEM_PROMPT_DOMAINFIX,
    SYSTEM_PROMPT_BEYONDSWE,
    SYSTEM_PROMPTS,
)
from awe_agent.tasks.beyond_swe.prompt.user import (
    CROSSREPO_PROMPT,
    DEPMIGRATE_PROMPT,
    DOC2REPO_PROMPT,
    DOMAINFIX_PROMPT,
    SEARCH_CROSSREPO_PROMPT,
    SEARCH_DEPMIGRATE_PROMPT,
    SEARCH_DOC2REPO_PROMPT,
    SEARCH_DOMAINFIX_PROMPT,
    USER_PROMPTS,
)

__all__ = [
    "DOC2REPO_PROMPT",
    "CROSSREPO_PROMPT",
    "DEPMIGRATE_PROMPT",
    "DOMAINFIX_PROMPT",
    "SEARCH_DOC2REPO_PROMPT",
    "SEARCH_CROSSREPO_PROMPT",
    "SEARCH_DEPMIGRATE_PROMPT",
    "SEARCH_DOMAINFIX_PROMPT",
    "SYSTEM_PROMPT_BEYONDSWE",
    "SEARCH_SYSTEM_PROMPT_BEYONDSWE",
    "SEARCH_SYSTEM_PROMPT_DOMAINFIX",
    "SYSTEM_PROMPTS",
    "USER_PROMPTS",
]
