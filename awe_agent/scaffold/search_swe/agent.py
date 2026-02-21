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
from awe_agent.core.llm.format import get_format
from awe_agent.core.tool.code import ExecuteBashTool, FinishTool, StrReplaceEditorTool, ThinkTool
from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.search import LinkSummaryTool, SearchConstraints, SearchTool
from awe_agent.scaffold.search_swe.prompts.config import resolve_from_task_info
from awe_agent.scaffold.search_swe.prompts.system import NO_TOOL_CALL_PROMPT, get_system_prompt

# Name of the finish tool — used to detect explicit task completion.
_FINISH_TOOL_NAME = "finish"

if TYPE_CHECKING:
    from awe_agent.core.config.schema import AweAgentConfig

logger = logging.getLogger(__name__)

# ── Bash blocklists ──────────────────────────────────────────────
# Always blocked regardless of mode (prevent repo introspection)
_ALWAYS_BLOCKED = [
    r".*git log.*--all.*",
    r".*git verify-pack.*",
    r".*git fsck.*",
    r".*git cat-file.*",
]

# Additional blocks for non-search mode (prevent external data fetching)
_NON_SEARCH_BLOCKED = [
    r".*git clone.*",
    r".*git fetch.*",
    r".*git pull.*",
    r".*api\.github\.com.*",
    r".*github\.io.*",
    r".*githubusercontent.*",
]

# Combined default (used when no explicit blocklist and search is off)
_DEFAULT_BLOCKLIST = _ALWAYS_BLOCKED + _NON_SEARCH_BLOCKED


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
        bash_blocklist: Regex patterns for blocked bash commands.
        enable_think: Include the Think tool. Defaults to ``True``.
        blocked_search_domains: Domains blocked from search results.

    Example::

        agent = SearchSWEAgent(enable_search=True)
        # System prompt is resolved automatically from task_info
    """

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> SearchSWEAgent:
        """Create a SearchSWEAgent from the global config."""
        # Build search constraints from security config
        search_constraints: SearchConstraints | None = None
        if config.security.blocked_search_patterns:
            search_constraints = SearchConstraints(
                blocked_patterns=config.security.blocked_search_patterns,
            )

        return cls(
            enable_search=config.agent.enable_search,
            bash_timeout=config.agent.bash_timeout,
            bash_max_timeout=config.agent.bash_max_timeout,
            max_output_length=config.agent.max_output_length,
            bash_blocklist=config.security.bash_blocklist or None,
            search_constraints=search_constraints,
            tool_call_format=config.agent.tool_call_format,
        )

    @classmethod
    def from_config_with_constraints(
        cls,
        config: AweAgentConfig,
        instance_constraints: SearchConstraints,
    ) -> SearchSWEAgent:
        """Create a SearchSWEAgent merging config-level and instance-level constraints."""
        # Start with config-level constraints
        search_constraints: SearchConstraints | None = None
        if config.security.blocked_search_patterns:
            search_constraints = SearchConstraints(
                blocked_patterns=config.security.blocked_search_patterns,
            )

        # Merge with instance-level constraints
        if search_constraints is not None:
            search_constraints = search_constraints.merge(instance_constraints)
        else:
            search_constraints = instance_constraints

        return cls(
            enable_search=config.agent.enable_search,
            bash_timeout=config.agent.bash_timeout,
            bash_max_timeout=config.agent.bash_max_timeout,
            max_output_length=config.agent.max_output_length,
            bash_blocklist=config.security.bash_blocklist or None,
            search_constraints=search_constraints,
            tool_call_format=config.agent.tool_call_format,
        )

    def __init__(
        self,
        enable_search: bool = False,
        bash_timeout: int = 180,
        bash_max_timeout: int = 600,
        max_output_length: int = 32000,
        bash_blocklist: list[str] | None = None,
        enable_think: bool = False,
        search_constraints: SearchConstraints | None = None,
        max_empty_retries: int = 3,
        tool_call_format: str = "openai_function",
    ) -> None:
        self._max_empty_retries = max_empty_retries
        self._format = get_format(tool_call_format)
        self._enable_search = enable_search

        # Select base blocklist based on mode
        if bash_blocklist is not None:
            effective_blocklist = list(bash_blocklist)        # User-explicit: respect verbatim
        elif enable_search:
            effective_blocklist = list(_ALWAYS_BLOCKED)       # Search mode: relax git/github
        else:
            effective_blocklist = list(_DEFAULT_BLOCKLIST)    # Non-search: full lockdown
        if search_constraints is not None:
            effective_blocklist.extend(search_constraints.get_bash_blocklist_patterns())

        # Core tools
        self._tools: list[Tool] = [
            ExecuteBashTool(
                timeout=bash_timeout,
                max_output_length=max_output_length,
                blocklist=effective_blocklist,
                max_timeout=bash_max_timeout,
            ),
            StrReplaceEditorTool(),
        ]

        if enable_think:
            self._tools.append(ThinkTool())

        if enable_search:
            self._tools.append(SearchTool(
                constraints=search_constraints,
            ))
            self._tools.append(LinkSummaryTool(
                constraints=search_constraints,
            ))

        # Finish tool — the agent MUST call this to explicitly end the task.
        # Without it the loop cannot distinguish "agent forgot to call a tool"
        # from "agent believes the task is done".
        self._tools.append(FinishTool())

    # ── Agent protocol ────────────────────────────────────────────────

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        """Resolve the system prompt dynamically from the route table."""
        system_key, _ = resolve_from_task_info(task_info, search=self._enable_search)
        prompt = get_system_prompt(system_key)
        # Append tool descriptions for text-based formats (e.g. CodeActXML)
        tool_schemas = [tool.schema for tool in self._tools]
        suffix = self._format.get_system_prompt_suffix(tool_schemas)
        if suffix:
            prompt = prompt + "\n" + suffix
        return prompt

    def get_tools(self) -> list[Tool]:
        return list(self._tools)

    def get_no_tool_call_prompt(self) -> str | None:
        """Remind the LLM to use tools or call ``finish``."""
        return NO_TOOL_CALL_PROMPT

    @staticmethod
    def _is_valid_response(response: Any) -> bool:
        """Check if an LLM response is valid (non-empty and not truncated)."""
        has_content = bool(response.content) or bool(response.tool_calls)
        truncated = getattr(response, "finish_reason", None) == "length"
        return has_content and not truncated

    async def step(self, context: AgentContext) -> Action:
        """Call LLM with conversation and tools, return action.

        Returns one of three action types:

        * ``"finish"``    — LLM invoked the *finish* tool (explicit completion).
        * ``"tool_call"`` — LLM invoked one or more regular tools.
        * ``"message"``   — LLM returned text only with no tool calls.
          The loop can then decide whether to send a reminder prompt or
          treat it as a terminal state.
        """
        # Optionally condense messages for the LLM call (full history is preserved)
        messages = context.messages
        if context.condenser is not None:
            messages = await context.condenser.condense(messages)

        # Prepare tools based on format (None for text-based formats)
        api_tools = self._format.prepare_tools(context.get_tool_schemas())

        # LLM call with validation retry
        response = None
        for attempt in range(1, self._max_empty_retries + 1):
            response = await context.llm.chat(
                messages=messages,
                tools=api_tools,
            )
            if self._is_valid_response(response):
                break
            logger.warning(
                "Invalid LLM response (attempt %d/%d): empty=%s, truncated=%s",
                attempt,
                self._max_empty_retries,
                not response.content and not response.tool_calls,
                response.finish_reason == "length",
            )

        # Parse tool calls using the configured format
        tool_calls = self._format.parse_response(response)

        if tool_calls:
            tool_call_dicts = [tc.to_dict() for tc in tool_calls]
            is_finish = any(
                tc.name == _FINISH_TOOL_NAME for tc in tool_calls
            )
            return Action(
                type="finish" if is_finish else "tool_call",
                content=response.content,
                thinking=response.thinking,
                tool_calls=tool_call_dicts,
                token_ids=response.completion_token_ids,
                logprobs=response.logprobs,
                usage=response.usage,
            )

        # LLM returned text without invoking any tool.
        return Action(
            type="message",
            content=response.content,
            thinking=response.thinking,
            token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
            usage=response.usage,
        )
