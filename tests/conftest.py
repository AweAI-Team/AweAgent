"""Shared fixtures for AweAgent tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.runtime.types import ExecutionResult


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class MockRuntimeSession(RuntimeSession):
    """In-memory mock runtime session for testing."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}
        self._default_result = ExecutionResult(stdout="", stderr="", exit_code=0)

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        self.commands.append(command)
        # Simple simulation: git diff returns empty patch
        if "git diff" in command:
            return ExecutionResult(stdout="", stderr="", exit_code=0)
        return self._default_result

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        self.files[remote_path] = content

    async def download_file(self, remote_path: str) -> bytes:
        if remote_path in self.files:
            return self.files[remote_path]
        raise FileNotFoundError(remote_path)

    async def list_files(self, path: str, recursive: bool = False) -> list[str]:
        return [p for p in self.files if p.startswith(path)]

    async def close(self) -> None:
        pass


@pytest.fixture
def mock_session() -> MockRuntimeSession:
    return MockRuntimeSession()


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(backend="openai", model="gpt-4o-mini", base_url="http://test")


def make_llm_response(
    content: str = "I'll fix the bug.",
    tool_calls: list[ToolCall] | None = None,
    thinking: str = "",
) -> LLMResponse:
    """Helper to create mock LLM responses."""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        thinking=thinking,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
    )
