"""Core types for the Runtime layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of a command execution in a runtime session."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    @property
    def output(self) -> str:
        """Combined stdout + stderr for convenience."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(self.stderr)
        return "\n".join(parts)

    @property
    def success(self) -> bool:
        return self.exit_code == 0


@dataclass
class FileInfo:
    """Metadata about a file in the runtime."""

    path: str
    size: int = 0
    is_dir: bool = False


@dataclass
class RuntimeSessionInfo:
    """Information about a created runtime session."""

    session_id: str
    endpoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
