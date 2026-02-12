"""Tests for built-in tools."""

from __future__ import annotations

import pytest

from awe_agent.core.runtime.types import ExecutionResult
from awe_agent.core.tool.builtin.bash import BashTool
from awe_agent.core.tool.builtin.editor import FileEditorTool
from awe_agent.core.tool.builtin.search import LinkSummaryTool, SearchTool
from awe_agent.core.tool.builtin.think import ThinkTool
from tests.conftest import MockRuntimeSession


# ── BashTool ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bash_requires_session():
    tool = BashTool()
    result = await tool.execute({"command": "echo hello"})
    assert "requires a runtime session" in result


@pytest.mark.asyncio
async def test_bash_empty_command():
    tool = BashTool()
    session = MockRuntimeSession()
    result = await tool.execute({"command": ""}, session=session)
    assert "empty command" in result.lower()


@pytest.mark.asyncio
async def test_bash_blacklist():
    tool = BashTool(blacklist=[r".*rm -rf.*"])
    session = MockRuntimeSession()
    result = await tool.execute({"command": "rm -rf /"}, session=session)
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_bash_execute():
    tool = BashTool()
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="hello world", exit_code=0)
    result = await tool.execute({"command": "echo hello world"}, session=session)
    assert "hello world" in result
    assert "echo hello world" in session.commands


@pytest.mark.asyncio
async def test_bash_truncation():
    tool = BashTool(max_output_length=100)
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="x" * 500, exit_code=0)
    result = await tool.execute({"command": "cat bigfile"}, session=session)
    assert "truncated" in result


@pytest.mark.asyncio
async def test_bash_nonzero_exit_code():
    tool = BashTool()
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="", stderr="not found", exit_code=1)
    result = await tool.execute({"command": "false"}, session=session)
    assert "exit code: 1" in result


def test_bash_schema():
    tool = BashTool()
    schema = tool.schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "bash"
    assert "command" in schema["function"]["parameters"]["properties"]


# ── FileEditorTool ───────────────────────────────────────────────────────────


def test_editor_schema():
    tool = FileEditorTool()
    schema = tool.schema
    assert schema["function"]["name"] == "editor"
    params = schema["function"]["parameters"]["properties"]
    assert "command" in params
    assert "path" in params


@pytest.mark.asyncio
async def test_editor_view():
    tool = FileEditorTool()
    session = MockRuntimeSession()
    session.files["/test.py"] = b"print('hello')\n"
    session._default_result = ExecutionResult(
        stdout="print('hello')\n", exit_code=0
    )
    result = await tool.execute(
        {"command": "view", "path": "/test.py"},
        session=session,
    )
    assert "print" in result


@pytest.mark.asyncio
async def test_editor_create():
    tool = FileEditorTool()
    session = MockRuntimeSession()
    result = await tool.execute(
        {"command": "create", "path": "/new.py", "file_text": "x = 1\n"},
        session=session,
    )
    assert "created" in result.lower() or "/new.py" in result


# ── ThinkTool ────────────────────────────────────────────────────────────────


def test_think_schema():
    tool = ThinkTool()
    assert tool.name == "think"
    assert "thought" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_think_execute():
    tool = ThinkTool()
    result = await tool.execute({"thought": "I need to fix the parser."})
    assert "recorded" in result.lower()


# ── SearchTool ───────────────────────────────────────────────────────────────


def test_search_schema():
    tool = SearchTool()
    assert tool.name == "search"
    assert "query" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_search_default():
    tool = SearchTool()
    result = await tool.execute({"query": "python asyncio timeout"})
    assert "python asyncio timeout" in result


@pytest.mark.asyncio
async def test_search_empty_query():
    tool = SearchTool()
    result = await tool.execute({"query": ""})
    assert "error" in result.lower()


# ── LinkSummaryTool ──────────────────────────────────────────────────────────


def test_link_summary_schema():
    tool = LinkSummaryTool()
    assert tool.name == "link_summary"
    assert "url" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_link_summary_requires_session():
    tool = LinkSummaryTool()
    result = await tool.execute({"url": "https://example.com"})
    assert "requires a runtime session" in result.lower()
