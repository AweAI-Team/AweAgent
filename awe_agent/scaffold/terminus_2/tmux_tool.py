"""TmuxExecuteTool — internal tool bridging AgentLoop to TmuxSessionAdapter.

This tool is never exposed to the LLM.  It is created by ``Terminus2Agent``
on its first ``step()`` call and registered in ``AgentContext.tools`` so
that ``AgentLoop._execute_tools()`` can dispatch to it when executing the
synthetic ``tmux_execute`` ToolCall produced by ``TerminusJSONFormat``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from awe_agent.core.tool.protocol import Tool

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession
    from awe_agent.scaffold.terminus_2.tmux_session import TmuxSessionAdapter

logger = logging.getLogger(__name__)

_CONFIRMATION_TEXT = (
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)


class TmuxExecuteTool(Tool):
    """Execute keystrokes in a tmux session and return terminal output.

    This is a framework-internal tool: the LLM never sees its schema or
    description.  ``TerminusJSONFormat.parse_response()`` produces a
    synthetic ``ToolCall(name="tmux_execute", ...)`` whose arguments are
    dispatched here by ``AgentLoop._execute_tools()``.

    When ``is_task_complete`` is ``True`` in the arguments, the returned
    observation includes a double-confirmation prompt (aligned with
    the Terminal Bench double-confirmation flow).
    """

    def __init__(
        self,
        tmux_adapter: TmuxSessionAdapter,
        max_output_bytes: int = 10_000,
    ) -> None:
        self._tmux = tmux_adapter
        self._max_output_bytes = max_output_bytes

    @property
    def name(self) -> str:
        return "tmux_execute"

    @property
    def description(self) -> str:
        return "Send keystrokes to a tmux session and return terminal output."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keystrokes": {"type": "string"},
                            "duration": {"type": "number"},
                        },
                    },
                    "description": "Keystroke/duration pairs to send to tmux.",
                },
                "is_task_complete": {
                    "type": "boolean",
                    "description": "Whether the agent marked the task complete.",
                },
                "warning": {
                    "type": "string",
                    "description": "Parse warnings to relay to the LLM.",
                },
            },
            "required": ["commands"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        commands = params.get("commands", [])
        is_task_complete = params.get("is_task_complete", False)
        warning = params.get("warning", "")

        # Send keystrokes to tmux.
        errors: list[str] = []
        for cmd in commands:
            keystrokes = cmd.get("keystrokes", "")
            duration = min(cmd.get("duration", 1.0), 60.0)
            try:
                await self._tmux.send_keys(
                    keystrokes,
                    block=False,
                    min_timeout_sec=max(duration, 0.1),
                )
            except Exception as exc:
                logger.error("send_keys failed: %s", exc)
                errors.append(f"Error sending keystrokes: {exc}")

        # If all commands failed, return the errors immediately so the
        # LLM can adjust (e.g. wrong directory, broken pipe, etc.).
        if errors and len(errors) == len(commands):
            return "\n".join(errors)

        # Capture terminal output.
        terminal_output = await self._tmux.get_incremental_output()
        output = self._limit_output(terminal_output)

        # Prepend send_keys errors (partial failure) so the LLM is aware.
        if errors:
            output = "\n".join(errors) + "\n\n" + output

        # Prepend warnings if any.
        if warning:
            output = (
                f"Previous response had warnings:\n{warning}\n\n{output}"
            )

        # Append confirmation prompt on first task_complete signal.
        if is_task_complete:
            output = (
                f"Current terminal state:\n{output}\n\n{_CONFIRMATION_TEXT}"
            )

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _limit_output(self, output: str) -> str:
        """Truncate output to *max_output_bytes*, keeping head and tail."""
        encoded = output.encode("utf-8")
        if len(encoded) <= self._max_output_bytes:
            return output
        half = self._max_output_bytes // 2
        head = encoded[:half].decode("utf-8", errors="ignore")
        tail = encoded[-half:].decode("utf-8", errors="ignore")
        omitted = len(encoded) - 2 * half
        return f"{head}\n[... {omitted} bytes omitted ...]\n{tail}"
