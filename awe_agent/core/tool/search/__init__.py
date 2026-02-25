"""Search tools — web search, link reading, and summarization with pluggable backends."""

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.link_reader_tool import LinkReaderTool
from awe_agent.core.tool.search.link_summary_tool import LinkSummaryTool
from awe_agent.core.tool.search.prompts import PROMPT_REGISTRY, resolve_prompt
from awe_agent.core.tool.search.search_tool import SearchTool

__all__ = [
    "SearchConstraints",
    "SearchTool",
    "LinkReaderTool",
    "LinkSummaryTool",
    "PROMPT_REGISTRY",
    "resolve_prompt",
]
