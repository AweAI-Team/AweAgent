"""File editor tool — view, create, and edit files via str_replace."""

from __future__ import annotations

from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool


class FileEditorTool(Tool):
    """View and edit files using str_replace strategy.

    Supports: view, create, str_replace, insert operations.
    """

    @property
    def name(self) -> str:
        return "editor"

    @property
    def description(self) -> str:
        return (
            "View, create, or edit files. "
            "Commands: view (read file), create (create new file), "
            "str_replace (replace exact string), insert (insert at line)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "description": "The operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file.",
                },
                "file_text": {
                    "type": "string",
                    "description": "Content for 'create' command.",
                },
                "old_str": {
                    "type": "string",
                    "description": "String to replace (for 'str_replace').",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (for 'str_replace' / 'insert').",
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Line number to insert after (for 'insert').",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Line range [start, end] for 'view'.",
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
            return "Error: FileEditorTool requires a runtime session."

        command = params.get("command", "")
        path = params.get("path", "")

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
            return f"Error: unknown command '{command}'"

    async def _view(
        self, session: RuntimeSession, path: str, view_range: list[int] | None
    ) -> str:
        result = await session.execute(f"cat -n '{path}'")
        if not result.success:
            return f"Error viewing {path}: {result.stderr}"
        content = result.stdout
        if view_range and len(view_range) == 2:
            lines = content.split("\n")
            start, end = view_range[0] - 1, view_range[1]
            content = "\n".join(lines[max(0, start):end])
        return content

    async def _create(self, session: RuntimeSession, path: str, content: str) -> str:
        await session.upload_file(path, content.encode())
        return f"File created: {path}"

    async def _str_replace(
        self, session: RuntimeSession, path: str, old_str: str, new_str: str
    ) -> str:
        try:
            file_content = (await session.download_file(path)).decode()
        except Exception as e:
            return f"Error reading {path}: {e}"

        if old_str not in file_content:
            return f"Error: old_str not found in {path}. Check exact match including whitespace."

        count = file_content.count(old_str)
        if count > 1:
            return (
                f"Error: old_str found {count} times in {path}. "
                "Provide more context to make it unique."
            )

        new_content = file_content.replace(old_str, new_str, 1)
        await session.upload_file(path, new_content.encode())
        return f"Replacement applied in {path}."

    async def _insert(
        self, session: RuntimeSession, path: str, line_num: int, text: str
    ) -> str:
        try:
            file_content = (await session.download_file(path)).decode()
        except Exception as e:
            return f"Error reading {path}: {e}"

        lines = file_content.split("\n")
        if line_num < 0 or line_num > len(lines):
            return f"Error: line {line_num} out of range (0-{len(lines)})"

        new_lines = text.split("\n")
        lines[line_num:line_num] = new_lines
        await session.upload_file(path, "\n".join(lines).encode())
        return f"Inserted {len(new_lines)} lines at line {line_num} in {path}."
