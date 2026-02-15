"""Bash tool — execute shell commands in the runtime environment."""

from __future__ import annotations

import logging
import re
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool

logger = logging.getLogger(__name__)

# Default max output length to avoid overwhelming the context
_MAX_OUTPUT_LENGTH = 32000


class BashTool(Tool):
    """Execute bash commands in the runtime session.

    Supports command blacklisting for security (e.g., prevent git clone during eval).
    """

    def __init__(
        self,
        timeout: int = 120,
        max_output_length: int = _MAX_OUTPUT_LENGTH,
        blacklist: list[str] | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_output_length = max_output_length
        self._blacklist = [re.compile(p) for p in (blacklist or [])]

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute a bash command in the working environment. "
            "Use for running code, installing packages, exploring the repo, etc."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
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
            return "Error: BashTool requires a runtime session."

        command = params.get("command", "")
        if not command.strip():
            return "Error: empty command."

        # Check blacklist
        for pattern in self._blacklist:
            if pattern.match(command):
                return f"Error: command blocked by security policy: {command}"

        result = await session.execute(command, timeout=self._timeout)
        output = result.output
        if len(output) > self._max_output_length:
            half = self._max_output_length // 2
            output = (
                output[:half]
                + f"\n\n... [{len(output) - self._max_output_length} characters truncated] ...\n\n"
                + output[-half:]
            )
        if result.exit_code != 0:
            output = f"[exit code: {result.exit_code}]\n{output}"
        return output or "(no output)"
