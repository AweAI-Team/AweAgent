from __future__ import annotations

import pytest

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.local import LocalRuntime, LocalRuntimeSession
from awe_agent.core.task.runner import runtime_registry


def test_local_runtime_is_registered():
    assert runtime_registry.get("local") is LocalRuntime


@pytest.mark.asyncio
async def test_local_runtime_executes_in_configured_workdir(tmp_path):
    runtime = LocalRuntime(RuntimeConfig(backend="local", workdir=str(tmp_path)))

    async with runtime.session() as session:
        assert isinstance(session, LocalRuntimeSession)
        result = await session.execute("pwd")

    assert result.success
    assert result.stdout.strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_local_runtime_file_operations(tmp_path):
    runtime = LocalRuntime(RuntimeConfig(backend="local", workdir=str(tmp_path)))
    file_path = tmp_path / "nested" / "note.txt"

    async with runtime.session() as session:
        await session.upload_file(str(file_path), b"hello")
        assert await session.download_file(str(file_path)) == b"hello"
        listed = await session.list_files(str(tmp_path), recursive=True)

    assert str(file_path) in listed
