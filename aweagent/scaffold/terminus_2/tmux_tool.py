"""TmuxExecuteTool — internal tool bridging AgentLoop to TmuxSessionAdapter.

This tool is never exposed to the LLM.  It is created by ``Terminus2Agent``
on its first ``step()`` call and registered in ``AgentContext.tools`` so
that ``AgentLoop._execute_tools()`` can dispatch to it when executing the
synthetic ``tmux_execute`` ToolCall produced by ``TerminusJSONFormat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aweagent.core.tool.protocol import Tool

if TYPE_CHECKING:
    from aweagent.core.runtime.protocol import RuntimeSession
    from aweagent.scaffold.terminus_2.tmux_session import TmuxSessionAdapter

_TIMEOUT_TEMPLATE = (
    "Previous command:\n"
    "{command}\n\n"
    "The previous command timed out after {timeout_sec} seconds\n\n"
    "It is possible that the command is not yet finished executing. If that is the "
    "case, then do nothing. It is also possible that you have entered an interactive "
    "shell and should continue sending keystrokes as normal.\n\n"
    "Here is the current state of the terminal:\n\n"
    "{terminal_state} "
)

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

    When ``task_complete`` is ``True`` in the arguments, the returned
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
        return "Send exact keystrokes to the task terminal and return terminal output."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "description": (
                        "Keystroke/duration pairs to send to tmux. "
                        "Use an empty keystrokes string with a positive duration to wait."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "keystrokes": {
                                "type": "string",
                                "description": (
                                    "Exact keystrokes to send. End shell commands "
                                    "with a newline (\\n) or they will not execute."
                                ),
                            },
                            "duration": {
                                "type": "number",
                                "description": (
                                    "Seconds to wait after sending these keystrokes. "
                                    "Prefer polling instead of waiting over 60 seconds."
                                ),
                            },
                        },
                        "required": ["keystrokes"],
                    },
                },
                "task_complete": {
                    "type": "boolean",
                    "description": (
                        "Set true only when the task is complete and ready for grading."
                    ),
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
        is_task_complete = params.get("task_complete", False)
        warning = params.get("warning", "")

        # Send keystrokes to tmux. Transport failures propagate; only command
        # timeouts become a recovery observation, matching Harbor.
        timeout_output: str | None = None
        for cmd in commands:
            keystrokes = cmd.get("keystrokes", "")
            duration = min(cmd.get("duration", 1.0), 60.0)
            try:
                await self._tmux.send_keys(
                    keystrokes,
                    block=False,
                    min_timeout_sec=duration,
                )
            except TimeoutError:
                terminal_state = self.limit_output(
                    await self._tmux.get_incremental_output()
                )
                timeout_output = _TIMEOUT_TEMPLATE.format(
                    command=keystrokes,
                    timeout_sec=duration,
                    terminal_state=terminal_state,
                )
                break

        # Capture terminal output.
        if timeout_output is not None:
            output = timeout_output
        else:
            terminal_output = await self._tmux.get_incremental_output()
            output = self.limit_output(terminal_output)

        # Completion confirmation takes precedence over parser warning feedback,
        # matching Harbor's observation construction order.
        if is_task_complete:
            output = (
                f"Current terminal state:\n{output}\n\n{_CONFIRMATION_TEXT}"
            )
        elif warning:
            output = (
                "Previous response had warnings:\n"
                f"WARNINGS: {warning}\n\n{output}"
            )

        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def limit_output(self, output: str) -> str:
        """Truncate output to *max_output_bytes*, keeping head and tail."""
        encoded = output.encode("utf-8")
        if len(encoded) <= self._max_output_bytes:
            return output
        half = self._max_output_bytes // 2
        head = encoded[:half].decode("utf-8", errors="ignore")
        tail = encoded[-half:].decode("utf-8", errors="ignore")
        omitted = (
            len(encoded)
            - len(head.encode("utf-8"))
            - len(tail.encode("utf-8"))
        )
        return (
            f"{head}\n[... output limited to {self._max_output_bytes} bytes; "
            f"{omitted} interior bytes omitted ...]\n{tail}"
        )
