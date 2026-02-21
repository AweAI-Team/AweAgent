"""ExecuteBashTool — execute shell commands in the runtime environment.

Migrated from swalm's CodeAct agent with enhanced descriptions,
timeout support, blocklist checking, and output truncation.
"""

from __future__ import annotations

import logging
import re
import time
from textwrap import dedent
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool

logger = logging.getLogger(__name__)

_MAX_OUTPUT_LENGTH = 32000


class ExecuteBashTool(Tool):
    """Execute bash commands in the runtime session.

    Features:
    - Configurable timeout with per-command override via parameters
    - Command blocklist for security (e.g., prevent git clone during eval)
    - Output truncation to avoid overwhelming the context
    - Exit code and execution time reporting
    """

    def __init__(
        self,
        timeout: int = 180,
        max_output_length: int = _MAX_OUTPUT_LENGTH,
        blocklist: list[str] | None = None,
        max_timeout: int = 600,
    ) -> None:
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._blocklist = [re.compile(p) for p in (blocklist or [])]
        self._max_timeout = max_timeout

    @property
    def name(self) -> str:
        return "execute_bash"

    @property
    def description(self) -> str:
        return dedent("""\
            Execute a bash command in the terminal.
            * One command at a time: You can only execute one bash command at a time. \
If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
            * Persistent session: Commands execute in a persistent shell session where \
environment variables, virtual environments, and working directory persist between commands.
            * Soft timeout: Commands have a soft timeout. Once that's reached, the command \
will be interrupted.
            * Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in \
shell scripts or commands in this environment. The runtime may not support them and can \
cause unusable shell sessions. If you want to run multi-line bash commands, write the \
commands to a file and then run it, instead.
            * For commands that may run indefinitely, run them in the background and \
redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
            * Directory verification: Before creating new directories or files, first \
verify the parent directory exists and is the correct location.
            * Directory management: Try to maintain working directory by using absolute \
paths and avoiding excessive use of `cd`.
            * Output truncation: If the output exceeds a maximum length, it will be \
truncated before being returned.""")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The bash command to execute. "
                        "Can be empty string to view additional logs when previous "
                        "exit code is `-1`. "
                        "Can be `C-c` (Ctrl+C) to interrupt the currently running process."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Sets a hard timeout in seconds for the command execution. "
                        "Optional — uses the default timeout if not specified."
                    ),
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        if session is None:
            return "Error: ExecuteBashTool requires a runtime session."

        command = params.get("command", "")
        if not command.strip():
            return "Error: empty command."

        # Check blocklist
        for pattern in self._blocklist:
            if pattern.match(command):
                return f"Error: command blocked by security policy: {command}"

        timeout = params.get("timeout", self._timeout)
        if isinstance(timeout, str):
            try:
                timeout = int(timeout)
            except ValueError:
                timeout = self._timeout
        timeout = min(timeout, self._max_timeout)

        start_time = time.monotonic()
        try:
            result = await session.execute(command, timeout=timeout)
        except TimeoutError:
            elapsed = time.monotonic() - start_time
            return (
                f"Command timed out after {elapsed:.1f} seconds and has been terminated.\n"
                "Consider using a longer timeout or running the command in the background."
            )
        except Exception as e:
            return f"Error executing command: {e}"

        elapsed = time.monotonic() - start_time

        output = result.output
        if len(output) > self._max_output_length:
            half = self._max_output_length // 2
            output = (
                output[:half]
                + f"\n\n... [{len(output) - self._max_output_length} characters truncated] ...\n\n"
                + output[-half:]
            )

        # Build structured response
        parts = []
        if output:
            parts.append(output)
        if elapsed >= 1.0:
            parts.append(f"[Execution time: {elapsed:.2f}s]")
        if result.exit_code != 0:
            parts.append(f"[Command finished with exit code {result.exit_code}]")

        return "\n".join(parts) or "(no output)"
