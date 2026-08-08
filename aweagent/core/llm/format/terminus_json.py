"""Terminus JSON format — 3rd ToolCallFormat for Terminal Bench 2.0.

The LLM outputs raw JSON text with keystrokes instead of calling tools.
This format parses the JSON and translates it into a synthetic ToolCall
for the internal ``tmux_execute`` tool, enabling standard AgentLoop
integration without changing the LLM-facing prompt.

Flow:
    LLM outputs: {"analysis": "...", "commands": [...], "task_complete": false}
    TerminusJSONFormat.parse_response() -> [ToolCall(name="tmux_execute", ...)]
    AgentLoop._execute_tools() -> TmuxExecuteTool.execute() -> terminal output
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from aweagent.core.llm.format.protocol import ToolCallFormat
from aweagent.core.llm.types import LLMResponse, ToolCall

if TYPE_CHECKING:
    from aweagent.scaffold.terminus_2.parser import ParseResult

# Internal tool name — used only by the framework, the LLM never sees it.
TMUX_EXECUTE_TOOL_NAME = "tmux_execute"


class TerminusJSONFormat(ToolCallFormat):
    """JSON keystroke format for Terminal Bench 2.0.

    The LLM outputs a structured JSON response containing ``analysis``,
    ``plan``, ``commands``, and an optional ``task_complete`` flag.  This
    format parses the JSON text and converts it into a single synthetic
    :class:`ToolCall` targeting the internal ``tmux_execute`` tool.

    Unlike ``OpenAIFunctionFormat`` or ``CodeActXMLFormat``, this format
    does *not* expose any tool descriptions to the LLM — the JSON schema
    is specified entirely in the user prompt (``json_plain.txt``).
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency:
        # format/__init__ -> terminus_json -> scaffold/terminus_2/__init__
        # -> agent -> format/__init__
        from aweagent.scaffold.terminus_2.parser import TerminusJSONParser

        self._parser = TerminusJSONParser()
        self._last_parse_result: ParseResult | None = None
        self._reasoning_format: str | None = None

    def set_reasoning_format(self, reasoning_format: str | None) -> None:
        self._reasoning_format = reasoning_format

    def needs_native_tools(self) -> bool:
        return False

    def format_text_observation(
        self,
        tool_name: str,
        observation: str,
    ) -> str:
        """Terminus sends raw terminal state back as the next user message."""
        return observation

    def prepare_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        # Tools are not sent to the LLM API.
        return None

    def get_system_prompt_suffix(self, tools: list[dict[str, Any]]) -> str:
        # The prompt is entirely agent-defined (json_plain.txt).
        return ""

    def parse_response(self, response: LLMResponse) -> list[ToolCall]:
        """Parse JSON keystroke response and wrap as a synthetic ToolCall.

        Returns a single-element list on success, or an empty list when
        parsing fails (which signals the agent's ``step()`` to handle the
        error via ``get_no_tool_call_prompt``).
        """
        content = response.content or ""
        self._last_parse_result = self._parser.parse_response(content)

        if self._last_parse_result.error:
            return []

        commands_data = [
            {"keystrokes": c.keystrokes, "duration": c.duration}
            for c in self._last_parse_result.commands
        ]
        args: dict[str, Any] = {
            "commands": commands_data,
            "task_complete": self._last_parse_result.is_task_complete,
        }
        # Only include warning when non-empty to keep arguments lean.
        warning = self._last_parse_result.warning
        if warning:
            args["warning"] = warning
        return [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:16]}",
                name=TMUX_EXECUTE_TOOL_NAME,
                arguments=json.dumps(args),
            )
        ]

    @property
    def last_parse_result(self) -> ParseResult | None:
        """The most recent parse result, for agent-side error inspection."""
        return self._last_parse_result
