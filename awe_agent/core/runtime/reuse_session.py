"""Runtime adapter that reuses an existing session instead of creating a new one.

Used when evaluation must run in the SAME container as the agent (e.g. Terminal Bench),
where there is no patch to apply — the agent modifies the container state directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.protocol import Runtime, RuntimeSession


class _NonClosingSessionWrapper(RuntimeSession):
    """Wraps a session but does not close it (session is managed elsewhere)."""

    def __init__(self, session: RuntimeSession) -> None:
        self._session = session

    async def execute(self, *args, **kwargs):
        return await self._session.execute(*args, **kwargs)

    async def upload_file(self, *args, **kwargs):
        return await self._session.upload_file(*args, **kwargs)

    async def download_file(self, *args, **kwargs):
        return await self._session.download_file(*args, **kwargs)

    async def list_files(self, *args, **kwargs):
        return await self._session.list_files(*args, **kwargs)

    async def close(self) -> None:
        pass  # No-op — do not close the borrowed session


class RuntimeWithExistingSession(Runtime):
    """Runtime that yields an existing session instead of creating a new one.

    Used when evaluating in the same container as the agent (e.g. Terminal Bench).
    """

    def __init__(self, session: RuntimeSession) -> None:
        self._session = session
        self.config = RuntimeConfig()

    async def create_session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> RuntimeSession:
        return _NonClosingSessionWrapper(self._session)

    @asynccontextmanager
    async def session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> AsyncIterator[RuntimeSession]:
        sess = await self.create_session(image, **kwargs)
        try:
            yield sess
        finally:
            await sess.close()
