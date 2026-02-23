"""Runtime Protocol — the interface all runtime backends must satisfy."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import AsyncIterator

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.types import ExecutionResult

logger = logging.getLogger(__name__)


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
        """Apply a patch. Tries 6 strategies for robustness."""
        await self.upload_file("/tmp/_awe_agent.patch", patch.encode())

        strategies = [
            ("git apply --verbose /tmp/_awe_agent.patch", False),
            ("git apply --verbose --ignore-space-change --ignore-whitespace /tmp/_awe_agent.patch", False),
            ("patch --batch --fuzz=5 -p1 -i /tmp/_awe_agent.patch", False),
            ("git apply --verbose --reject /tmp/_awe_agent.patch", True),
            ("git apply --verbose --reject --ignore-space-change --ignore-whitespace /tmp/_awe_agent.patch", True),
            ("git apply --verbose --reject --ignore-space-change --ignore-whitespace --allow-empty /tmp/_awe_agent.patch", True),
        ]

        last_result = None
        for cmd, is_reject in strategies:
            result = await self.execute(cmd, cwd=cwd)
            if result.success:
                return result
            if is_reject and result.exit_code == 1:
                # --reject partial success (rejected hunks written to .rej)
                return ExecutionResult(stdout=result.stdout, stderr=result.stderr, exit_code=0)
            last_result = result

        return last_result or ExecutionResult(stderr="All patch strategies failed", exit_code=1)

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
        """Context manager for runtime sessions. Ensures cleanup.

        Enforces ``config.timeout`` as a hard session TTL at the protocol
        level.  Individual backends (e.g. Docker) may enforce tighter
        per-operation timeouts internally; this acts as a final safety net.
        """
        sess = await self.create_session(image, **kwargs)
        try:
            timeout = self.config.timeout
            if timeout and timeout > 0:
                async with asyncio.timeout(timeout):
                    yield sess
            else:
                yield sess
        except TimeoutError:
            logger.error(
                "Session TTL expired after %ds — force-closing session",
                self.config.timeout,
            )
            raise
        finally:
            await sess.close()
