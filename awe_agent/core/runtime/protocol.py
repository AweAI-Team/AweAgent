"""Runtime Protocol — the interface all runtime backends must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.types import ExecutionResult


class RuntimeSession(ABC):
    """Abstract session representing a single running environment.

    Provides command execution, file operations, and git helpers.
    All operations are async for high concurrency.
    """

    @abstractmethod
    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a shell command."""
        ...

    @abstractmethod
    async def upload_file(self, remote_path: str, content: bytes) -> None:
        """Upload a file to the container."""
        ...

    @abstractmethod
    async def download_file(self, remote_path: str) -> bytes:
        """Download a file from the container."""
        ...

    @abstractmethod
    async def list_files(self, path: str, recursive: bool = False) -> list[str]:
        """List files in a directory."""
        ...

    async def get_patch(self, cwd: str, base_commit: str | None = None) -> str:
        """Get git diff as patch. Default implementation uses shell commands."""
        if base_commit:
            result = await self.execute(f"git diff {base_commit}", cwd=cwd)
        else:
            result = await self.execute("git diff", cwd=cwd)
        return result.stdout

    async def apply_patch(self, cwd: str, patch: str) -> ExecutionResult:
        """Apply a patch. Tries multiple strategies for robustness."""
        # Upload patch
        await self.upload_file("/tmp/_awe_agent.patch", patch.encode())

        # Try git apply first
        result = await self.execute(
            "git apply --verbose /tmp/_awe_agent.patch", cwd=cwd
        )
        if result.success:
            return result

        # Fallback: git apply with reject
        result = await self.execute(
            "git apply --verbose --reject /tmp/_awe_agent.patch", cwd=cwd
        )
        if result.success:
            return result

        # Final fallback: patch command
        return await self.execute(
            "patch --batch --fuzz=5 -p1 < /tmp/_awe_agent.patch", cwd=cwd
        )

    @abstractmethod
    async def close(self) -> None:
        """Release all resources associated with this session."""
        ...

    async def __aenter__(self) -> RuntimeSession:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


class Runtime(ABC):
    """Abstract runtime factory. Creates sessions for different backends."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @abstractmethod
    async def create_session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> RuntimeSession:
        """Create a new runtime session (container)."""
        ...

    @asynccontextmanager
    async def session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[RuntimeSession]:
        """Context manager for runtime sessions. Ensures cleanup."""
        sess = await self.create_session(image, **kwargs)
        try:
            yield sess
        finally:
            await sess.close()
