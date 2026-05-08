"""Tests for code tools (bash, editor, think, finish)."""

from __future__ import annotations

import pytest

from awe_agent.core.runtime.types import ExecutionResult
from awe_agent.core.tool.code.bash import ExecuteBashTool
from awe_agent.core.tool.code.editor import StrReplaceEditorTool
from awe_agent.core.tool.code.think import ThinkTool
from awe_agent.core.tool.code.finish import (
    FINISH_TOOL_BUNDLES,
    FinishTool,
    FinishWithIntTool,
    FileFLFinishTool,
    LineFLFinishTool,
    SubmitFileFinishTool,
)
from awe_agent.core.tool.search import WebFetchTool, WebSearchTool
from tests.conftest import MockRuntimeSession


# ── ExecuteBashTool ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bash_requires_session():
    tool = ExecuteBashTool()
    result = await tool.execute({"command": "echo hello"})
    assert "requires a runtime session" in result


@pytest.mark.asyncio
async def test_bash_empty_command():
    tool = ExecuteBashTool()
    session = MockRuntimeSession()
    result = await tool.execute({"command": ""}, session=session)
    assert "empty command" in result.lower()


@pytest.mark.asyncio
async def test_bash_blocklist():
    tool = ExecuteBashTool(blocklist=[r".*rm -rf.*"])
    session = MockRuntimeSession()
    result = await tool.execute({"command": "rm -rf /"}, session=session)
    assert "blocked" in result.lower()


@pytest.mark.asyncio
async def test_bash_execute():
    tool = ExecuteBashTool()
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="hello world", exit_code=0)
    result = await tool.execute({"command": "echo hello world"}, session=session)
    assert "hello world" in result
    assert "echo hello world" in session.commands


@pytest.mark.asyncio
async def test_bash_truncation():
    tool = ExecuteBashTool(max_output_length=100)
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="x" * 500, exit_code=0)
    result = await tool.execute({"command": "cat bigfile"}, session=session)
    assert "truncated" in result


@pytest.mark.asyncio
async def test_bash_nonzero_exit_code():
    tool = ExecuteBashTool()
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="", stderr="not found", exit_code=1)
    result = await tool.execute({"command": "false"}, session=session)
    assert "exit code" in result.lower()


def test_bash_schema():
    tool = ExecuteBashTool()
    schema = tool.schema
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "execute_bash"
    assert "command" in schema["function"]["parameters"]["properties"]
    assert "timeout" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_bash_custom_timeout():
    """Test that per-command timeout parameter is accepted."""
    tool = ExecuteBashTool(timeout=60)
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="ok", exit_code=0)
    result = await tool.execute({"command": "sleep 1", "timeout": 10}, session=session)
    assert "ok" in result


# ── StrReplaceEditorTool ──────────────────────────────────────────────


def test_editor_schema():
    tool = StrReplaceEditorTool()
    schema = tool.schema
    assert schema["function"]["name"] == "str_replace_editor"
    params = schema["function"]["parameters"]["properties"]
    assert "command" in params
    assert "path" in params
    assert "view_range" in params
    assert "file_text" in params
    assert "old_str" in params
    assert "new_str" in params
    assert "insert_line" in params


@pytest.mark.asyncio
async def test_editor_view():
    tool = StrReplaceEditorTool()
    session = MockRuntimeSession()
    session.files["/test.py"] = b"print('hello')\n"
    session._default_result = ExecutionResult(
        stdout="     1\tprint('hello')\n", exit_code=0
    )
    result = await tool.execute(
        {"command": "view", "path": "/test.py"},
        session=session,
    )
    assert "print" in result


@pytest.mark.asyncio
async def test_editor_create():
    tool = StrReplaceEditorTool()
    session = MockRuntimeSession()
    # First call checks if dir, second checks if file exists
    session._default_result = ExecutionResult(stdout="FILE", exit_code=0)
    result = await tool.execute(
        {"command": "create", "path": "/new.py", "file_text": "x = 1\n"},
        session=session,
    )
    assert "created" in result.lower() or "/new.py" in result


@pytest.mark.asyncio
async def test_editor_create_existing_file():
    """Create should fail if the file already exists."""
    tool = StrReplaceEditorTool()
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="EXISTS", exit_code=0)
    result = await tool.execute(
        {"command": "create", "path": "/existing.py", "file_text": "x = 1\n"},
        session=session,
    )
    assert "already exists" in result


@pytest.mark.asyncio
async def test_editor_str_replace():
    """str_replace should replace the exact match and show snippet."""
    tool = StrReplaceEditorTool()
    session = MockRuntimeSession()
    session.files["/test.py"] = b"def hello():\n    return 'hello'\n"
    result = await tool.execute(
        {
            "command": "str_replace",
            "path": "/test.py",
            "old_str": "return 'hello'",
            "new_str": "return 'world'",
        },
        session=session,
    )
    assert "has been edited" in result
    assert b"return 'world'" in session.files["/test.py"]


@pytest.mark.asyncio
async def test_editor_str_replace_not_unique():
    """str_replace should fail if old_str matches multiple times."""
    tool = StrReplaceEditorTool()
    session = MockRuntimeSession()
    session.files["/test.py"] = b"x = 1\nx = 1\n"
    result = await tool.execute(
        {
            "command": "str_replace",
            "path": "/test.py",
            "old_str": "x = 1",
            "new_str": "x = 2",
        },
        session=session,
    )
    assert "2 times" in result


# ── ThinkTool ─────────────────────────────────────────────────────────


def test_think_schema():
    tool = ThinkTool()
    assert tool.name == "think"
    assert "content" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_think_execute():
    tool = ThinkTool()
    result = await tool.execute({"content": "I need to fix the parser."})
    assert "recorded" in result.lower()


@pytest.mark.asyncio
async def test_think_history():
    """Think tool should accumulate thought history."""
    tool = ThinkTool()
    await tool.execute({"content": "First thought"})
    await tool.execute({"content": "Second thought"})
    assert len(tool.think_history) == 2
    assert tool.think_history[0] == "First thought"
    assert tool.think_history[1] == "Second thought"


# ── FinishTool ────────────────────────────────────────────────────────


def test_finish_tool_registry():
    """All finish tool variants should be registered."""
    assert "default" in FINISH_TOOL_BUNDLES
    assert "submit_int" in FINISH_TOOL_BUNDLES
    assert "file_fl" in FINISH_TOOL_BUNDLES
    assert "line_fl" in FINISH_TOOL_BUNDLES
    assert "submit_file" in FINISH_TOOL_BUNDLES


def test_finish_default_schema():
    tool = FinishTool()
    assert tool.name == "finish"
    schema = tool.schema
    assert schema["function"]["name"] == "finish"


@pytest.mark.asyncio
async def test_finish_default_execute():
    tool = FinishTool()
    result = await tool.execute({})
    assert "complete" in result.lower()


def test_finish_int_submit():
    tool = FinishWithIntTool()
    assert tool.submit({"answer": 42}) == 42
    assert tool.submit({"answer": "7"}) == 7
    assert tool.submit({}) is None


def test_finish_file_fl_submit():
    tool = FileFLFinishTool()
    result = tool.submit({"files": "file1.py\nA/file2.py\n"})
    assert result == ["file1.py", "A/file2.py"]


def test_finish_line_fl_submit():
    tool = LineFLFinishTool()
    result = tool.submit({"lines": "file1.py:10,12\nfile2.py:5-8"})
    assert result == {
        "file1.py": [10, 12],
        "file2.py": [5, 6, 7, 8],
    }


def test_finish_submit_file():
    tool = SubmitFileFinishTool()
    result = tool.submit({"file_path": " /tmp/result.txt "})
    assert result == "/tmp/result.txt"


# ── WebSearchTool ─────────────────────────────────────────────────────


def test_web_search_schema():
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "query" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_web_search_default():
    tool = WebSearchTool()
    result = await tool.execute({"query": "python asyncio timeout"})
    # Without bandai_mcp_host, returns no results
    assert "python asyncio timeout" in result


@pytest.mark.asyncio
async def test_web_search_empty_query():
    tool = WebSearchTool()
    result = await tool.execute({"query": ""})
    assert "error" in result.lower()


# ── WebFetchTool ──────────────────────────────────────────────────────


def test_web_fetch_schema():
    tool = WebFetchTool()
    assert tool.name == "web_fetch"
    assert "url" in tool.parameters["properties"]
    assert "prompt" in tool.parameters["properties"]


@pytest.mark.asyncio
async def test_web_fetch_empty_url():
    tool = WebFetchTool()
    result = await tool.execute({"url": "", "prompt": "test"})
    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_web_fetch_empty_prompt():
    tool = WebFetchTool()
    result = await tool.execute({"url": "https://example.com", "prompt": ""})
    assert "error" in result.lower()
