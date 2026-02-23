"""StrReplaceEditorTool — view, create, and edit files via str_replace."""

from __future__ import annotations

import logging
from textwrap import dedent
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool

logger = logging.getLogger(__name__)

# Maximum number of lines to show in view output before clipping
_MAX_VIEW_LINES = 500

# Snippet context lines to show around a replacement
_SNIPPET_CONTEXT_LINES = 4


class StrReplaceEditorTool(Tool):
    """View and edit files using the str_replace strategy.

    Supports four operations:
    - view: Display file contents with line numbers, or list directory contents
    - create: Create a new file (fails if the path already exists)
    - str_replace: Replace an exact string occurrence in a file
    - insert: Insert text after a specified line number
    """

    @property
    def name(self) -> str:
        return "str_replace_editor"

    @property
    def description(self) -> str:
        return dedent("""\
            Custom editing tool for viewing, creating and editing files.
            * State is persistent across command calls and discussions with the user.
            * If `path` is a file, `view` displays the result of applying `cat -n`. \
If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep.
            * The `create` command cannot be used if the specified `path` already exists as a file.
            * If a `command` generates a long output, it will be truncated and marked \
with `<response clipped>`.
            Notes for using the `str_replace` command:
            * The `old_str` parameter should match EXACTLY one or more consecutive lines \
from the original file. Be mindful of whitespaces!
            * If the `old_str` parameter is not unique in the file, the replacement will \
not be performed. Make sure to include enough context in `old_str` to make it unique.
            * The `new_str` parameter should contain the edited lines that should replace \
the `old_str`.
            * This tool can be used for creating and editing files in plain-text format.
            * Before using this tool:
              - 1. Use the view tool to understand the file's contents and context.
              - 2. Verify the directory path is correct (only applicable when creating new files): \
Use the view tool to verify the parent directory exists and is the correct location.
              - When making edits:
                - Ensure the edit results in idiomatic, correct code.
                - Do not leave the code in a broken state.
                - Always use absolute file paths (starting with /).
            * Remember: when making multiple file edits in a row to the same file, you \
should prefer to send all edits in a single message with multiple calls to this tool, \
rather than multiple messages with a single call each.""")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "description": "The command to run.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to file or directory, "
                        "e.g. `/repo/file.py` or `/repo`."
                    ),
                },
                "file_text": {
                    "type": "string",
                    "description": (
                        "Required parameter of `create` command, "
                        "with the content of the file to be created."
                    ),
                },
                "old_str": {
                    "type": "string",
                    "description": (
                        "Required parameter of `str_replace` command "
                        "containing the string in `path` to replace."
                    ),
                },
                "new_str": {
                    "type": "string",
                    "description": (
                        "Optional parameter of `str_replace` command containing "
                        "the new string (if not given, no string will be added). "
                        "Required parameter of `insert` command containing "
                        "the string to insert."
                    ),
                },
                "insert_line": {
                    "type": "integer",
                    "description": (
                        "Required parameter of `insert` command. "
                        "The `new_str` will be inserted AFTER the line "
                        "`insert_line` of `path`."
                    ),
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Optional parameter of `view` command when `path` "
                        "points to a file. e.g. [11, 12] will show lines 11 and 12. "
                        "Indexing at 1. `[start_line, -1]` shows all lines from "
                        "`start_line` to the end of the file."
                    ),
                },
            },
            "required": ["command", "path"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        if session is None:
            return "Error: StrReplaceEditorTool requires a runtime session."

        command = params.get("command", "")
        path = params.get("path", "")

        if not path:
            return "Error: 'path' parameter is required."

        if command == "view":
            return await self._view(session, path, params.get("view_range"))
        elif command == "create":
            return await self._create(session, path, params.get("file_text", ""))
        elif command == "str_replace":
            return await self._str_replace(
                session, path, params.get("old_str", ""), params.get("new_str", "")
            )
        elif command == "insert":
            return await self._insert(
                session, path, params.get("insert_line", 0), params.get("new_str", "")
            )
        else:
            return f"Error: unknown command '{command}'. Valid commands: view, create, str_replace, insert."

    # ── View ──────────────────────────────────────────────────────────

    async def _view(
        self, session: RuntimeSession, path: str, view_range: list[int] | None
    ) -> str:
        # Check if path is a directory or file
        check = await session.execute(f"test -d '{path}' && echo DIR || echo FILE")
        is_dir = check.stdout.strip() == "DIR"

        if is_dir:
            return await self._view_directory(session, path)
        return await self._view_file(session, path, view_range)

    async def _view_directory(self, session: RuntimeSession, path: str) -> str:
        """List non-hidden files and directories up to 2 levels deep."""
        result = await session.execute(
            f"find '{path}' -maxdepth 2 -not -path '*/\\.*' | head -200 | sort"
        )
        if not result.success:
            return f"Error listing directory {path}: {result.stderr}"
        output = result.stdout.strip()
        if not output:
            return f"Directory '{path}' is empty."
        return output

    async def _view_file(
        self, session: RuntimeSession, path: str, view_range: list[int] | None
    ) -> str:
        """View file contents with line numbers."""
        result = await session.execute(f"cat -n '{path}'")
        if not result.success:
            return f"Error viewing {path}: {result.stderr or result.output}"

        content = result.stdout
        if not content:
            return f"File '{path}' is empty."

        if view_range and len(view_range) == 2:
            lines = content.split("\n")
            start = max(0, view_range[0] - 1)
            end = len(lines) if view_range[1] == -1 else view_range[1]
            content = "\n".join(lines[start:end])
        else:
            # Clip long output
            lines = content.split("\n")
            if len(lines) > _MAX_VIEW_LINES:
                content = "\n".join(lines[:_MAX_VIEW_LINES]) + "\n<response clipped>"

        return content

    # ── Create ────────────────────────────────────────────────────────

    async def _create(self, session: RuntimeSession, path: str, content: str) -> str:
        if not content:
            return "Error: 'file_text' parameter is required for 'create' command."

        # Check if file already exists
        check = await session.execute(f"test -f '{path}' && echo EXISTS || echo OK")
        if check.stdout.strip() == "EXISTS":
            return (
                f"Error: file '{path}' already exists. "
                "Use 'str_replace' to edit existing files, or choose a different path."
            )

        # Ensure parent directory exists
        parent = path.rsplit("/", 1)[0] if "/" in path else "."
        await session.execute(f"mkdir -p '{parent}'")

        await session.upload_file(path, content.encode())
        return f"File created successfully at: {path}"

    # ── str_replace ───────────────────────────────────────────────────

    async def _str_replace(
        self, session: RuntimeSession, path: str, old_str: str, new_str: str
    ) -> str:
        if not old_str:
            return "Error: 'old_str' parameter is required for 'str_replace' command."

        try:
            file_content = (await session.download_file(path)).decode()
        except Exception as e:
            return f"Error reading {path}: {e}"

        if old_str not in file_content:
            return (
                f"Error: no match found for `old_str` in {path}. "
                "Check that the string matches EXACTLY, including whitespace and indentation."
            )

        count = file_content.count(old_str)
        if count > 1:
            return (
                f"Error: `old_str` found {count} times in {path}. "
                "Include more context in `old_str` to make it unique."
            )

        new_content = file_content.replace(old_str, new_str, 1)
        await session.upload_file(path, new_content.encode())

        # Show a snippet around the replacement for confirmation
        replacement_line = file_content[: file_content.index(old_str)].count("\n") + 1
        return self._format_replacement_result(
            path, new_content, replacement_line, old_str, new_str
        )

    def _format_replacement_result(
        self,
        path: str,
        new_content: str,
        replacement_line: int,
        old_str: str,
        new_str: str,
    ) -> str:
        """Format the result of a str_replace showing a context snippet."""
        lines = new_content.split("\n")
        new_str_lines = new_str.count("\n") + 1 if new_str else 0

        start = max(0, replacement_line - 1 - _SNIPPET_CONTEXT_LINES)
        end = min(len(lines), replacement_line - 1 + new_str_lines + _SNIPPET_CONTEXT_LINES)

        snippet_lines = []
        for i in range(start, end):
            snippet_lines.append(f"{i + 1:6}\t{lines[i]}")
        snippet = "\n".join(snippet_lines)

        return (
            f"The file {path} has been edited. Here's the result of running "
            f"`cat -n` on a snippet of {path}:\n{snippet}\n"
            "Review the changes and make sure they are as expected. "
            "Edit the file again if necessary."
        )

    # ── Insert ────────────────────────────────────────────────────────

    async def _insert(
        self, session: RuntimeSession, path: str, line_num: int, text: str
    ) -> str:
        if not text:
            return "Error: 'new_str' parameter is required for 'insert' command."

        try:
            file_content = (await session.download_file(path)).decode()
        except Exception as e:
            return f"Error reading {path}: {e}"

        lines = file_content.split("\n")
        if line_num < 0 or line_num > len(lines):
            return (
                f"Error: insert_line {line_num} is out of range "
                f"(valid range: 0 to {len(lines)})."
            )

        new_lines = text.split("\n")
        lines[line_num:line_num] = new_lines
        new_content = "\n".join(lines)
        await session.upload_file(path, new_content.encode())

        # Show a snippet around the insertion
        start = max(0, line_num - _SNIPPET_CONTEXT_LINES)
        end = min(len(lines), line_num + len(new_lines) + _SNIPPET_CONTEXT_LINES)
        snippet_lines = []
        for i in range(start, end):
            snippet_lines.append(f"{i + 1:6}\t{lines[i]}")
        snippet = "\n".join(snippet_lines)

        return (
            f"The file {path} has been edited. Here's the result of running "
            f"`cat -n` on a snippet of the edited file:\n{snippet}\n"
            "Review the changes and make sure they are as expected (correct indentation, "
            "no duplicate lines, etc). Edit the file again if necessary."
        )
