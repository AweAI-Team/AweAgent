"""Terminus 2 Agent — standard step()-based agent for Terminal Bench 2.0.

Uses ``TerminusJSONFormat`` (the 3rd ToolCallFormat) to translate the LLM's
raw JSON keystroke output into a synthetic ``ToolCall`` for the internal
``TmuxExecuteTool``.  This allows the agent to run inside the standard
``AgentLoop``, inheriting RL training, context condensing, stats tracking,
and step callbacks — without changing the LLM-facing prompt.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.llm.types import Message
from awe_agent.core.tool.protocol import Tool
from awe_agent.scaffold.terminus_2.tmux_session import TmuxSessionAdapter
from awe_agent.scaffold.terminus_2.tmux_tool import TmuxExecuteTool

if TYPE_CHECKING:
    from awe_agent.core.config.schema import AweAgentConfig
    from awe_agent.core.llm.format.protocol import ToolCallFormat
    from awe_agent.core.llm.format.terminus_json import TerminusJSONFormat

logger = logging.getLogger(__name__)

_DEFAULT_NO_TOOL_CALL_PROMPT = (
    "Your response could not be parsed as valid JSON. "
    "Please provide a valid JSON response with the required fields: "
    '"analysis", "plan", and "commands".'
)


class Terminus2Agent(Agent):
    """Terminal Bench 2.0 agent using the standard ``step()`` protocol.

    Interaction flow (per step):

    1. ``step()`` calls the LLM (no API-level tools).
    2. ``TerminusJSONFormat.parse_response()`` extracts keystrokes from
       the raw JSON text and wraps them as a synthetic ``ToolCall``.
    3. ``AgentLoop._execute_tools()`` dispatches to ``TmuxExecuteTool``,
       which sends keystrokes to tmux and returns terminal output.
    4. The observation (terminal output) is appended as the next user
       message (non-native-tools mode), and the loop continues.

    Double-confirmation for ``task_complete`` is handled within ``step()``:
    the first occurrence returns ``Action(type="tool_call")`` (the tool
    observation includes a confirmation prompt); the second occurrence
    returns ``Action(type="finish")``.
    """

    def __init__(
        self,
        session_name: str = "terminus-session",
        max_output_bytes: int = 10_000,
        max_empty_retries: int = 2,
    ) -> None:
        self._session_name = session_name
        self._max_output_bytes = max_output_bytes
        self._max_empty_retries = max_empty_retries

        # Lazy import to avoid circular dependency at module level.
        from awe_agent.core.llm.format import get_tool_format

        self._format: TerminusJSONFormat = get_tool_format("terminus_json")  # type: ignore[assignment]

        # Lazily initialised on first step().
        self._tmux: TmuxSessionAdapter | None = None
        self._tmux_tool: TmuxExecuteTool | None = None
        self._initialized: bool = False

        # Double-confirmation state.
        self._pending_completion: bool = False
        # Stores the last parse-error message for get_no_tool_call_prompt().
        self._last_parse_error: str = ""

    # ------------------------------------------------------------------
    # Agent protocol
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> Terminus2Agent:
        """Create from global config."""
        return cls()

    @classmethod
    def from_config_with_constraints(
        cls, config: AweAgentConfig, instance_constraints: Any
    ) -> Terminus2Agent:
        """Terminus 2 ignores search constraints."""
        return cls.from_config(config)

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        """Return an empty system prompt.

        The full prompt (instructions + JSON schema + terminal state) is
        delivered as a single user message, matching the Terminal Bench
        convention.  An empty system message is harmless for all major
        LLM providers.
        """
        return ""

    def get_tools(self) -> list[Tool]:
        """Return the internal tmux tool if initialized, else empty list.

        The tool list is populated lazily in ``step()`` once the tmux
        session is started.  ``AgentLoop._execute_tools()`` looks up
        tools by name on ``context.tools``, which is updated in-place.
        """
        if self._tmux_tool is not None:
            return [self._tmux_tool]
        return []

    def get_tool_call_format(self) -> ToolCallFormat | None:
        """Return the TerminusJSON format (text-based, non-native)."""
        return self._format

    def get_no_tool_call_prompt(self) -> str | None:
        """Return a parse-error specific or generic JSON retry prompt.

        Called by ``AgentLoop`` whenever ``step()`` returns
        ``Action(type="message")`` with no tool calls.  The prompt
        content is dynamic: if the last response had a parse error,
        the specific error is returned so the LLM can fix its output.
        """
        if self._last_parse_error:
            return self._last_parse_error
        return _DEFAULT_NO_TOOL_CALL_PROMPT

    async def step(self, context: AgentContext) -> Action:
        """Single-step decision: call LLM, parse JSON, return action.

        On the first invocation the tmux session is started and the
        initial terminal state is injected into the user message.
        """
        # -- Lazy init: start tmux and populate the initial prompt -----
        if not self._initialized:
            await self._initialize(context)

        # -- Condense messages if configured ---------------------------
        messages = context.messages
        if context.condenser is not None:
            messages = await context.condenser.condense(messages)

        # -- LLM call (no native tools) --------------------------------
        api_tools = self._format.prepare_tools(context.get_tool_schemas())

        llm_overrides: dict[str, Any] = {}
        if context.training is not None:
            llm_overrides["input_ids"] = context.training.get_input_ids()

        response = None
        for attempt in range(1, self._max_empty_retries + 1):
            response = await context.llm.chat(
                messages=messages,
                tools=api_tools,
                **llm_overrides,
            )
            if response.content:
                break
            if (
                context.training is not None
                and response.finish_status == "length"
            ):
                break
            logger.warning(
                "Empty LLM response (attempt %d/%d)",
                attempt,
                self._max_empty_retries,
            )

        # -- Parse response --------------------------------------------
        tool_calls = self._format.parse_response(response)
        parse_result = self._format.last_parse_result

        # -- Handle parse failure --------------------------------------
        if not tool_calls:
            error_msg = "No valid JSON found in response."
            if parse_result and parse_result.error:
                error_msg = f"Parsing error: {parse_result.error}"
                if parse_result.warning:
                    error_msg += f"\nWarnings: {parse_result.warning}"
            error_msg += (
                "\n\nPlease fix and provide valid JSON with "
                '"analysis", "plan", and "commands" fields.'
            )
            self._last_parse_error = error_msg
            return Action(
                type="message",
                content=response.content,
                thinking=response.thinking,
                token_ids=response.completion_token_ids,
                logprobs=response.logprobs,
                weight_version=response.weight_version,
                finish_status=response.finish_status,
                usage=response.usage,
            )

        # -- Parse succeeded -------------------------------------------
        self._last_parse_error = ""
        tool_call_dicts = [tc.to_dict() for tc in tool_calls]

        # Double-confirmation logic for task_complete.
        is_task_complete = (
            parse_result is not None and parse_result.is_task_complete
        )

        if is_task_complete and self._pending_completion:
            # Second confirmation -> finish.
            return Action(
                type="finish",
                content=response.content,
                thinking=response.thinking,
                tool_calls=tool_call_dicts,
                token_ids=response.completion_token_ids,
                logprobs=response.logprobs,
                weight_version=response.weight_version,
                finish_status=response.finish_status,
                usage=response.usage,
            )

        if is_task_complete:
            # First task_complete: execute commands normally; the tool
            # observation will include the confirmation prompt.
            self._pending_completion = True
        else:
            self._pending_completion = False

        return Action(
            type="tool_call",
            content=response.content,
            thinking=response.thinking,
            tool_calls=tool_call_dicts,
            token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
            weight_version=response.weight_version,
            finish_status=response.finish_status,
            usage=response.usage,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _initialize(self, context: AgentContext) -> None:
        """Start tmux, register the internal tool, and fill terminal_state."""
        workdir = context.task_info.get("workdir", "/workspace")

        self._tmux = TmuxSessionAdapter(
            session=context.session,
            session_name=self._session_name,
            workdir=workdir,
        )
        await self._tmux.start()

        self._tmux_tool = TmuxExecuteTool(
            self._tmux, max_output_bytes=self._max_output_bytes,
        )
        context.tools = [self._tmux_tool]

        # Inject the real terminal state into the initial user message.
        # The prompt template is passed via task_info by the Task, so the
        # scaffold layer does not depend on the tasks layer.
        initial_state = await self._tmux.get_incremental_output()
        instruction = context.task_info.get("instruction", "")
        prompt_template = context.task_info.get("prompt_template", "")
        full_prompt = prompt_template.format(
            instruction=instruction,
            terminal_state=initial_state,
        )
        for i, msg in enumerate(context.messages):
            if msg.role == "user":
                context.messages[i] = Message(
                    role="user", content=full_prompt,
                )
                break

        # Re-init training prompt tokens if in RL mode, because we
        # changed the user message content.
        if context.training is not None:
            msg_dicts = [m.to_dict() for m in context.messages]
            tool_schemas = context.get_tool_schemas() or None
            context.training.init_prompt(msg_dicts, tools=tool_schemas)

        self._initialized = True
