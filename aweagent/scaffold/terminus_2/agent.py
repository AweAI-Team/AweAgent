"""Terminus 2 Agent — standard step()-based agent for Terminal Bench 2.0.

Uses ``TerminusJSONFormat`` (the 3rd ToolCallFormat) to translate the LLM's
raw JSON keystroke output into a synthetic ``ToolCall`` for the internal
``TmuxExecuteTool``.  This allows the agent to run inside the standard
``AgentLoop``, inheriting RL training, context condensing, stats tracking,
and step callbacks — without changing the LLM-facing prompt.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.trajectory import Action
from aweagent.core.condenser.terminus_2 import (
    Terminus2Condenser,
    Terminus2ContextInput,
    Terminus2ContextResult,
)
from aweagent.core.llm.types import LLMResponse, Message
from aweagent.core.tool.protocol import Tool
from aweagent.scaffold.terminus_2.tmux_session import TmuxSessionAdapter
from aweagent.scaffold.terminus_2.tmux_tool import TmuxExecuteTool

if TYPE_CHECKING:
    from aweagent.core.condenser.protocol import Condenser
    from aweagent.core.config.schema import AweAgentConfig
    from aweagent.core.llm.format.protocol import ToolCallFormat
    from aweagent.core.llm.format.terminus_json import TerminusJSONFormat

logger = logging.getLogger(__name__)

_DEFAULT_NO_TOOL_CALL_PROMPT = (
    "Your response could not be parsed as valid JSON. "
    "Please provide a valid JSON response with the required fields: "
    '"analysis", "plan", and "commands".'
)
_TECHNICAL_DIFFICULTIES_RESPONSE = (
    "Technical difficulties. Please continue with the task."
)


def _get_model_output_limit(model_name: str) -> int | None:
    """Return LiteLLM's model-map output limit for Harbor retry feedback."""
    try:
        from litellm import get_model_info

        model_info = get_model_info(model_name)
        return model_info.get("max_output_tokens")
    except Exception as exc:
        logger.debug(
            "Failed to retrieve output limit for model %r: %s",
            model_name,
            exc,
        )
        return None


def _is_context_window_error(error: BaseException | str) -> bool:
    """Return whether a provider rejected Terminus 2's overlong context."""
    parts = [
        str(error),
        str(getattr(error, "body", "")),
        str(getattr(error, "message", "")),
        str(getattr(error, "error", "")),
    ]
    text = " ".join(part.lower() for part in parts if part)
    markers = (
        "context length exceeded",
        "context_length_exceeded",
        "requested token count exceeds",
        "maximum context length",
        "context length",
        "context window",
        "`inputs` tokens + `max_new_tokens`",
        "model's context length",
        "prompt is too long",
        "input is too long for requested model",
    )
    return any(marker in text for marker in markers)


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

    @staticmethod
    def manages_context_limits(condenser: Condenser | None) -> bool:
        """Return whether this agent owns context handling for ``condenser``."""
        return isinstance(condenser, Terminus2Condenser)

    def __init__(
        self,
        session_name: str = "terminus-session",
        max_output_bytes: int = 10_000,
        max_empty_retries: int = 1,
    ) -> None:
        self._session_name = session_name
        self._max_output_bytes = max_output_bytes
        self._max_empty_retries = max_empty_retries

        # Lazy import to avoid circular dependency at module level.
        from aweagent.core.llm.format import get_tool_format

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

        if self._tmux is None:
            raise RuntimeError("Terminus tmux session is not initialized")
        if not await self._tmux.is_session_alive():
            logger.info(
                "Terminus tmux session ended; finishing without another LLM call"
            )
            return Action(type="finish")

        # -- Condense messages if configured ---------------------------
        messages = context.messages
        context_manager = (
            context.condenser
            if isinstance(context.condenser, Terminus2Condenser)
            else None
        )
        if context_manager is not None and context.max_context_length is not None:
            proactive = await context_manager.maybe_condense(
                self._context_management_input(context)
            )
            if proactive.compacted:
                context.messages = proactive.messages
                messages = context.messages
                self._record_context_management(context, proactive)
        elif context.condenser is not None:
            messages = await context.condenser.condense(messages)

        # -- LLM call (no native tools) --------------------------------
        api_tools = self._format.prepare_tools(context.get_tool_schemas())

        llm_overrides: dict[str, Any] = {}
        if context.training is not None:
            llm_overrides["input_ids"] = context.training.get_input_ids()

        self._format.set_reasoning_format(context.llm.config.reasoning.format)
        response = None
        empty_attempt = 0
        recovered_context = False
        while empty_attempt < self._max_empty_retries:
            try:
                response = await context.llm.chat(
                    messages=messages,
                    tools=api_tools,
                    **llm_overrides,
                )
            except Exception as exc:
                if recovered_context:
                    logger.error("Even fallback chat failed: %s", exc)
                    response = LLMResponse(
                        content=_TECHNICAL_DIFFICULTIES_RESPONSE
                    )
                else:
                    can_recover = (
                        context_manager is not None
                        and context.max_context_length is not None
                        and _is_context_window_error(exc)
                    )
                    if not can_recover:
                        raise
                    recovery = await context_manager.recover_from_context_error(
                        self._context_management_input(context)
                    )
                    if not recovery.compacted:
                        raise
                    recovered_context = True
                    context.messages = recovery.messages
                    messages = context.messages
                    self._record_context_management(context, recovery)
                    continue

            if (
                context.training is None
                and response.finish_reason == "length"
            ):
                messages = self._output_length_retry_messages(
                    context,
                    messages,
                    response,
                )
                context.messages = messages
                recovered_context = False
                continue

            empty_attempt += 1
            if response.content:
                break
            if (
                context.training is not None
                and response.finish_status == "length"
            ):
                break
            logger.warning(
                "Empty LLM response (attempt %d/%d)",
                empty_attempt,
                self._max_empty_retries,
            )

        # -- Parse response --------------------------------------------
        tool_calls = self._format.parse_response(response)
        parse_result = self._format.last_parse_result

        # -- Handle parse failure --------------------------------------
        if not tool_calls:
            feedback = "ERROR: No valid JSON found in response."
            if parse_result and parse_result.error:
                feedback = f"ERROR: {parse_result.error}"
                if parse_result.warning:
                    feedback += f"\nWARNINGS: {parse_result.warning}"
            self._last_parse_error = (
                "Previous response had parsing errors:\n"
                f"{feedback}\n\n"
                "Please fix these issues and provide a proper JSON response."
            )
            return Action(
                type="message",
                content=response.content,
                reasoning_text=response.reasoning_text,
                reasoning_raw=response.reasoning_raw,
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
                reasoning_text=response.reasoning_text,
                reasoning_raw=response.reasoning_raw,
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
            reasoning_text=response.reasoning_text,
            reasoning_raw=response.reasoning_raw,
            tool_calls=tool_call_dicts,
            token_ids=response.completion_token_ids,
            logprobs=response.logprobs,
            weight_version=response.weight_version,
            finish_status=response.finish_status,
            usage=response.usage,
        )

    def _context_management_input(
        self,
        context: AgentContext,
    ) -> Terminus2ContextInput:
        if self._tmux is None:
            raise RuntimeError("Terminus tmux session is not initialized")
        if context.max_context_length is None:
            raise RuntimeError("Terminus context management requires max_context_length")
        messages = list(context.messages)
        committed_messages = messages if len(messages) <= 1 else messages[:-1]
        return Terminus2ContextInput(
            messages=committed_messages,
            llm=context.llm,
            model_name=context.llm.config.model,
            max_context_length=context.max_context_length,
            original_instruction=context.task_info.get("instruction", ""),
            terminal_state_provider=partial(
                self._tmux.capture_pane,
                capture_entire=False,
            ),
            reserved_output_tokens=self._reserved_output_tokens(context),
            chat_template_kwargs=self._chat_template_kwargs(context),
            # The OpenAI backend's auto mode (None) does not round-trip reasoning.
            preserve_reasoning=(context.llm.config.reasoning.preserve is True),
        )

    @staticmethod
    def _chat_template_kwargs(context: AgentContext) -> dict[str, Any]:
        """Copy chat-template options used by the actual model request."""
        value = context.llm.config.params.get("chat_template_kwargs")
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _reserved_output_tokens(context: AgentContext) -> int:
        """Read the maximum requested completion from the active LLM config."""
        for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
            value = context.llm.config.params.get(key)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
            ):
                return value
        return 0

    @staticmethod
    def _record_context_management(
        context: AgentContext,
        result: Terminus2ContextResult,
    ) -> None:
        if result.trace is None:
            return
        events = context.trajectory.metadata.setdefault("context_management", [])
        trace = result.trace
        events.append(
            {
                "index": len(events) + 1,
                "at_step": context.current_step,
                "trigger": result.trigger,
                "fallback_level": result.fallback_level,
                "boundary": "replace",
                "token_state": {
                    "context_limit": trace.context_limit,
                    "reserved_output_tokens": trace.reserved_output_tokens,
                    "effective_input_limit": trace.effective_input_limit,
                    "tokens_before": trace.tokens_before,
                    "free_tokens_before": trace.free_tokens_before,
                },
                "stages": [asdict(stage) for stage in trace.stages],
                "handoff_prompt": trace.handoff_prompt,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _output_length_retry_messages(
        self,
        context: AgentContext,
        messages: list[Message],
        response: LLMResponse,
    ) -> list[Message]:
        """Append Harbor's output-length recovery exchange."""
        truncated_response = response.content or ""
        self._format.parse_response(response)
        parse_result = self._format.last_parse_result

        warnings_text = ""
        if parse_result is not None and parse_result.warning:
            warnings_text = (
                "\n\nParser warnings from your truncated response:\n"
                f"{parse_result.warning}"
            )

        output_limit = _get_model_output_limit(context.llm.config.model)
        limit_str = (
            f"{output_limit} tokens"
            if output_limit is not None
            else "the maximum output length"
        )
        error_prompt = (
            "ERROR!! NONE of the actions you just requested were performed "
            f"because you exceeded {limit_str}. "
            f"Your outputs must be less than {limit_str}. Re-issue this request, "
            f"breaking it into chunks each of which is less than {limit_str}."
            f"{warnings_text}"
        )
        return [
            *messages,
            Message(role="assistant", content=truncated_response),
            Message(role="user", content=error_prompt),
        ]

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
        initial_state = self._tmux_tool.limit_output(
            await self._tmux.get_incremental_output()
        )
        instruction = context.task_info.get("instruction", "")
        prompt_template = self._get_prompt_template(context)
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

    def _get_prompt_template(self, context: AgentContext) -> str:
        """Return the prompt template used to build the initial user message."""
        return context.task_info.get("prompt_template", "")
