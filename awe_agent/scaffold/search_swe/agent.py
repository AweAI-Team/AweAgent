"""SearchSWE Agent — deep research for coding.

The single unified agent for all AweAgent tasks. Dynamically selects system and
user prompts based on ``(dataset_id, task_type, search_mode)`` via the route
table in :mod:`~awe_agent.scaffold.search_swe.prompts.config`.

Supports two modes of operation:

- **Standard** (``enable_search=False``): Bash + Editor + Think.
- **Search**  (``enable_search=True``):  Adds SearchTool and LinkSummaryTool
  for evidence-based problem solving with web research.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.tool.builtin.bash import BashTool
from awe_agent.core.tool.builtin.editor import FileEditorTool
from awe_agent.core.tool.builtin.search import LinkSummaryTool, SearchTool
from awe_agent.core.tool.builtin.think import ThinkTool
from awe_agent.core.tool.protocol import Tool
from awe_agent.scaffold.search_swe.prompts.config import resolve_from_task_info
from awe_agent.scaffold.search_swe.prompts.system import get_system_prompt

if TYPE_CHECKING:
    from awe_agent.core.config.schema import AweAgentConfig

logger = logging.getLogger(__name__)

# Default bash constraints to prevent information leakage during evaluation
_DEFAULT_BLACKLIST = [
    r".*git clone.*",
    r".*git fetch.*",
    r".*git pull.*",
    r".*git log.*--all.*",
    r".*git verify-pack.*",
    r".*git fsck.*",
    r".*git cat-file.*",
    r".*api\.github\.com.*",
    r".*github\.io.*",
    r".*githubusercontent.*",
]


class SearchSWEAgent(Agent):
    """Unified coding agent with optional web search capabilities.

    Dynamically resolves the system prompt from the task context using
    the prompt routing table.  When ``enable_search=True``, the agent
    gains access to :class:`SearchTool` and :class:`LinkSummaryTool`
    for evidence-based coding.

    Args:
        enable_search: Enable web search tools. Defaults to ``False``.
        bash_timeout: Maximum seconds per bash command. Defaults to ``180``.
        max_output_length: Truncate bash output beyond this many characters.
        bash_blacklist: Regex patterns for blocked bash commands.
        enable_think: Include the Think tool. Defaults to ``True``.
        search_max_results: Max results from web search.
        blocked_search_domains: Domains blocked from search results.

    Example::

        agent = SearchSWEAgent(enable_search=True)
        # System prompt is resolved automatically from task_info
    """

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> SearchSWEAgent:
        """Create a SearchSWEAgent from the global config."""
        return cls(
            enable_search=config.agent.enable_search,
            bash_timeout=config.agent.bash_timeout,
            max_output_length=config.agent.max_output_length,
            bash_blacklist=config.security.bash_blacklist or None,
        )

    def __init__(
        self,
        enable_search: bool = False,
        bash_timeout: int = 180,
        max_output_length: int = 16000,
        bash_blacklist: list[str] | None = None,
        enable_think: bool = True,
        search_max_results: int = 5,
        blocked_search_domains: list[str] | None = None,
    ) -> None:
        self._enable_search = enable_search

        # Core tools
        self._tools: list[Tool] = [
            BashTool(
                timeout=bash_timeout,
                max_output_length=max_output_length,
                blacklist=bash_blacklist or list(_DEFAULT_BLACKLIST),
            ),
            FileEditorTool(),
        ]

        if enable_think:
            self._tools.append(ThinkTool())

        if enable_search:
            self._tools.append(SearchTool(
                max_results=search_max_results,
                blocked_domains=blocked_search_domains,
            ))
            self._tools.append(LinkSummaryTool())

    # ── Agent protocol ────────────────────────────────────────────────

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        """Resolve the system prompt dynamically from the route table."""
        system_key, _ = resolve_from_task_info(task_info, search=self._enable_search)
        return get_system_prompt(system_key)

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    async def step(self, context: AgentContext) -> Action:
        """Call LLM with conversation and tools, return action."""
        response = await context.llm.chat(
            messages=context.messages,
            tools=context.get_tool_schemas(),
        )

        if response.tool_calls:
            return Action(
                type="tool_call",
                content=response.content,
                thinking=response.thinking,
                tool_calls=[tc.to_dict() for tc in response.tool_calls],
                token_ids=response.completion_token_ids,
                logprobs=response.logprobs,
            )

        return Action(
            type="finish",
            content=response.content,
            thinking=response.thinking,
            token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
        )
