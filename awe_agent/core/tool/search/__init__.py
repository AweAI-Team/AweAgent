"""Web tools — search, fetch + summarize, and raw fetch with pluggable backends."""

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.prompts import PROMPT_REGISTRY, resolve_prompt
from awe_agent.core.tool.search.web_fetch_raw_tool import WebFetchRawTool
from awe_agent.core.tool.search.web_fetch_tool import WebFetchTool
from awe_agent.core.tool.search.web_search_tool import WebSearchTool

__all__ = [
    "SearchConstraints",
    "WebSearchTool",
    "WebFetchTool",
    "WebFetchRawTool",
    "PROMPT_REGISTRY",
    "resolve_prompt",
]
