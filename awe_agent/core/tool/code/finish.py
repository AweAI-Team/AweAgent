"""FinishTool family — signal task completion with optional structured answers.

Provides multiple finish tool variants
for different task types (default, integer answer, fault localization, file submission).

Usage:
    from awe_agent.core.tool.code.finish import FINISH_TOOL_BUNDLES

    # Get a specific finish tool by task type
    finish_cls = FINISH_TOOL_BUNDLES["default"]
    finish_tool = finish_cls()

    # Or use directly
    from awe_agent.core.tool.code.finish import FinishTool
    tool = FinishTool()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from textwrap import dedent
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool


class AbstractFinishTool(Tool, ABC):
    """Base class for all finish tools.

    Subclasses define what kind of answer the agent should submit
    via ``signature``, ``get_tool_params()``, and ``parse_tool_call()``.
    """

    @property
    def name(self) -> str:
        return "finish"

    @property
    def description(self) -> str:
        base = (
            "Finish the interaction when the task is complete "
            "OR if the assistant cannot proceed further with the task."
        )
        sig = self.signature
        if sig:
            return f"{base}\n{sig}"
        return base

    @property
    @abstractmethod
    def signature(self) -> str:
        """Description of what kind of answer to submit."""
        ...

    @abstractmethod
    def get_tool_params(self) -> dict[str, Any]:
        """Return the JSON Schema parameters for the finish tool."""
        ...

    @abstractmethod
    def parse_tool_call(self, params: dict[str, Any]) -> Any:
        """Parse the raw params into a typed result."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        return self.get_tool_params()

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        return "The task is complete."

    def submit(self, params: dict[str, Any]) -> Any:
        """Parse and return the submitted answer, or None on failure."""
        try:
            return self.parse_tool_call(params)
        except (KeyError, ValueError):
            return None


# ── Finish Tool Registry ──────────────────────────────────────────────

FINISH_TOOL_BUNDLES: dict[str, type[AbstractFinishTool]] = {}


def _register_finish_tool(name: str):
    """Decorator to register a finish tool variant."""
    def wrapper(cls: type[AbstractFinishTool]) -> type[AbstractFinishTool]:
        FINISH_TOOL_BUNDLES[name] = cls
        return cls
    return wrapper


# ── Concrete Finish Tools ─────────────────────────────────────────────


@_register_finish_tool("default")
class FinishTool(AbstractFinishTool):
    """Default finish tool — signals completion with no answer submission."""

    @property
    def signature(self) -> str:
        return ""

    def get_tool_params(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def parse_tool_call(self, params: dict[str, Any]) -> None:
        return


@_register_finish_tool("submit_int")
class FinishWithIntTool(AbstractFinishTool):
    """Finish tool for submitting an integer answer."""

    @property
    def signature(self) -> str:
        return "Submit your final answer by using the parameter 'answer'."

    def get_tool_params(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "description": dedent("""\
                        The answer you submit.
                        You should only submit one INTEGER without any extra content."""),
                },
            },
            "required": ["answer"],
        }

    def parse_tool_call(self, params: dict[str, Any]) -> int:
        return int(params["answer"])


@_register_finish_tool("file_fl")
class FileFLFinishTool(AbstractFinishTool):
    """Finish tool for file-level fault localization."""

    @property
    def signature(self) -> str:
        return "Submit your final answer by using the parameter 'files'."

    def get_tool_params(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "files": {
                    "type": "string",
                    "description": dedent("""\
                        The bug files you think are the most likely to be the cause of the bug.
                        You should submit the file paths in the format of 'path/to/file', \
and use \\n to split multiple files.
                        Each file path should be a relative path to the root of the repository.
                        e.g.
                        file1.py
                        A/file2.py"""),
                },
            },
            "required": ["files"],
        }

    def parse_tool_call(self, params: dict[str, Any]) -> list[str]:
        return [f.strip() for f in params["files"].split("\n") if f.strip()]


@_register_finish_tool("line_fl")
class LineFLFinishTool(AbstractFinishTool):
    """Finish tool for line-level fault localization."""

    @property
    def signature(self) -> str:
        return "Submit your final answer by using the parameter 'lines'."

    def get_tool_params(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "lines": {
                    "type": "string",
                    "description": dedent("""\
                        The bug lines you think are the most likely to be the cause of the bug.
                        You should submit the line numbers in the format of 'file:line', \
and use \\n to split multiple files.
                        Each file should be a relative path to the root of the repository.
                        e.g.
                        file1.py:10,12,13,15
                        A/file2.py:20,24,26
                        B/file3.py:125-139"""),
                },
            },
            "required": ["lines"],
        }

    def parse_tool_call(self, params: dict[str, Any]) -> dict[str, list[int]]:
        lines_str = params["lines"]
        lines: dict[str, list[int]] = defaultdict(list)
        for line in lines_str.split("\n"):
            try:
                line = line.strip()
                if not line:
                    continue
                file_path, line_nums = line.split(":")
                for line_block in line_nums.split(","):
                    line_block = line_block.strip()
                    if "-" in line_block:
                        start, end = line_block.split("-")
                        lines[file_path.strip()] += list(
                            range(int(start.strip()), int(end.strip()) + 1)
                        )
                    else:
                        lines[file_path.strip()].append(int(line_block))
            except Exception:  # noqa: BLE001
                continue
        return dict(lines)


@_register_finish_tool("submit_file")
class SubmitFileFinishTool(AbstractFinishTool):
    """Finish tool for submitting a file path as the final deliverable."""

    @property
    def signature(self) -> str:
        return "Submit the file path of your final deliverable."

    def get_tool_params(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": dedent("""\
                        The absolute path to the file you want to submit as your final result.
                        The path MUST start with root '/'.
                        Do not provide a relative path."""),
                },
            },
            "required": ["file_path"],
        }

    def parse_tool_call(self, params: dict[str, Any]) -> str:
        return params["file_path"].strip()
