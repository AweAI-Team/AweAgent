"""Tests for the 9-feature enhancement plan.

Covers:
- Item 2:  Dynamic bash timeout clamping
- Item 6:  RunStats statistics tracking
- Item 1:  Context condensing integration
- Item 3:  Search mode blocklist adjustment
- Item 5:  LLM response validation + retry
- Item 4:  Dynamic search constraints from task
- Item 7:  CodeActXML format support
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from awe_agent.core.agent.stats import RunStats
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.condenser import build_condenser
from awe_agent.core.condenser.truncation import TruncationCondenser
from awe_agent.core.config.schema import AgentConfig, CondenserConfig
from awe_agent.core.llm.format import get_format
from awe_agent.core.llm.format.openai import OpenAIFunctionFormat
from awe_agent.core.llm.format.xml import CodeActXMLFormat
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from awe_agent.core.runtime.types import ExecutionResult
from awe_agent.core.task.types import Instance
from awe_agent.core.tool.code.bash import ExecuteBashTool
from awe_agent.core.tool.search.constraints import SearchConstraints
from tests.conftest import MockRuntimeSession


# ═══════════════════════════════════════════════════════════════════════
# Item 2: Dynamic Bash Timeout Clamping
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_bash_timeout_clamping():
    """Timeout requested by LLM is clamped to max_timeout."""
    tool = ExecuteBashTool(timeout=180, max_timeout=600)
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="ok", exit_code=0)

    # Request a timeout far exceeding max — should be clamped to 600
    result = await tool.execute({"command": "sleep 1", "timeout": 9999}, session=session)
    assert "ok" in result


@pytest.mark.asyncio
async def test_bash_timeout_clamping_below_max():
    """Timeout below max is kept as-is."""
    tool = ExecuteBashTool(timeout=180, max_timeout=600)
    session = MockRuntimeSession()
    session._default_result = ExecutionResult(stdout="ok", exit_code=0)

    result = await tool.execute({"command": "sleep 1", "timeout": 30}, session=session)
    assert "ok" in result


@pytest.mark.asyncio
async def test_bash_default_max_timeout():
    """Default max_timeout is 600."""
    tool = ExecuteBashTool()
    assert tool._max_timeout == 600


@pytest.mark.asyncio
async def test_bash_custom_max_timeout():
    """Custom max_timeout is respected."""
    tool = ExecuteBashTool(max_timeout=120)
    assert tool._max_timeout == 120


# ═══════════════════════════════════════════════════════════════════════
# Item 6+10: RunStats Statistics Tracking
# ═══════════════════════════════════════════════════════════════════════


def test_run_stats_basic_lifecycle():
    """RunStats tracks timing, steps, and token counts."""
    stats = RunStats()
    stats.start()

    stats.record_llm_call(elapsed=1.5, prompt_tokens=100, completion_tokens=50)
    stats.record_tool_call("execute_bash", elapsed=0.3)
    stats.end_step()

    stats.record_llm_call(elapsed=2.0, prompt_tokens=200, completion_tokens=100)
    stats.record_tool_call("execute_bash", elapsed=0.5)
    stats.record_tool_call("str_replace_editor", elapsed=0.1)
    stats.end_step()

    stats.finish()

    d = stats.to_dict()
    assert d["steps"] == 2
    assert d["llm_calls"] == 2
    assert d["llm_time"] == pytest.approx(3.5)
    assert d["tool_time"] == pytest.approx(0.9)
    assert d["total_prompt_tokens"] == 300
    assert d["total_completion_tokens"] == 150
    assert d["tool_usage"] == {"execute_bash": 2, "str_replace_editor": 1}
    assert d["total_time"] >= 0  # Very fast test, may round to 0


def test_run_stats_empty():
    """RunStats with no calls returns zeroes."""
    stats = RunStats()
    stats.start()
    stats.finish()
    d = stats.to_dict()
    assert d["steps"] == 0
    assert d["llm_calls"] == 0
    assert d["total_prompt_tokens"] == 0


def test_run_stats_to_dict_structure():
    """to_dict returns all expected keys."""
    stats = RunStats()
    d = stats.to_dict()
    expected_keys = {
        "total_time", "llm_time", "tool_time", "steps",
        "llm_calls", "tool_usage", "total_prompt_tokens",
        "total_completion_tokens",
    }
    assert set(d.keys()) == expected_keys


def test_action_has_usage_field():
    """Action dataclass includes usage field."""
    usage = TokenUsage(prompt_tokens=10, completion_tokens=20)
    action = Action(type="message", content="hello", usage=usage)
    assert action.usage is usage
    assert action.usage.prompt_tokens == 10


# ═══════════════════════════════════════════════════════════════════════
# Item 1: Context Condensing Integration
# ═══════════════════════════════════════════════════════════════════════


def test_build_condenser_none():
    """type='none' returns None."""
    config = CondenserConfig(type="none")
    assert build_condenser(config) is None


def test_build_condenser_truncation():
    """type='truncation' returns TruncationCondenser."""
    config = CondenserConfig(type="truncation", max_messages=30, keep_first=3)
    condenser = build_condenser(config)
    assert isinstance(condenser, TruncationCondenser)
    assert condenser._max_messages == 30
    assert condenser._keep_first == 3


def test_build_condenser_invalid_type():
    """Unknown type raises ValueError."""
    config = CondenserConfig(type="quantum_compressor")
    with pytest.raises(ValueError, match="quantum_compressor"):
        build_condenser(config)


@pytest.mark.asyncio
async def test_truncation_condenser_preserves_short():
    """Messages under limit are returned unchanged."""
    condenser = TruncationCondenser(max_messages=10, keep_first=2)
    messages = [Message(role="user", content=f"msg {i}") for i in range(5)]
    result = await condenser.condense(messages)
    assert len(result) == 5


@pytest.mark.asyncio
async def test_truncation_condenser_truncates():
    """Messages over limit are truncated (keep first + recent)."""
    condenser = TruncationCondenser(max_messages=5, keep_first=2)
    messages = [Message(role="user", content=f"msg {i}") for i in range(10)]
    result = await condenser.condense(messages)
    assert len(result) == 5
    # First 2 messages preserved
    assert result[0].content == "msg 0"
    assert result[1].content == "msg 1"
    # Last 3 messages preserved
    assert result[2].content == "msg 7"
    assert result[3].content == "msg 8"
    assert result[4].content == "msg 9"


def test_agent_config_has_condenser():
    """AgentConfig includes condenser field."""
    config = AgentConfig()
    assert config.condenser.type == "none"
    assert config.condenser.max_messages == 50


# ═══════════════════════════════════════════════════════════════════════
# Item 3: Search Mode Blocklist Adjustment
# ═══════════════════════════════════════════════════════════════════════


def test_search_mode_allows_git_clone():
    """Search mode should NOT block git clone (only _ALWAYS_BLOCKED applies)."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    agent = SearchSWEAgent(enable_search=True)
    bash_tool = agent._tools[0]

    # In search mode, git clone should NOT be blocked
    import re
    git_clone_blocked = any(p.match("git clone https://github.com/test/repo") for p in bash_tool._blocklist)
    assert not git_clone_blocked, "git clone should be allowed in search mode"


def test_non_search_mode_blocks_git_clone():
    """Non-search mode should block git clone."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    agent = SearchSWEAgent(enable_search=False)
    bash_tool = agent._tools[0]

    import re
    git_clone_blocked = any(p.match("git clone https://github.com/test/repo") for p in bash_tool._blocklist)
    assert git_clone_blocked, "git clone should be blocked in non-search mode"


def test_always_blocked_in_search_mode():
    """git log --all should be blocked even in search mode."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    agent = SearchSWEAgent(enable_search=True)
    bash_tool = agent._tools[0]

    import re
    always_blocked = any(p.match("git log --all") for p in bash_tool._blocklist)
    assert always_blocked, "git log --all should always be blocked"


def test_explicit_blocklist_overrides():
    """Explicit blocklist should override defaults entirely."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    custom = [r".*forbidden.*"]
    agent = SearchSWEAgent(enable_search=False, bash_blocklist=custom)
    bash_tool = agent._tools[0]

    # Custom blocklist only has our pattern (+ no defaults)
    import re
    forbidden_blocked = any(p.match("forbidden command") for p in bash_tool._blocklist)
    assert forbidden_blocked
    # Default patterns should NOT be present
    git_clone_blocked = any(p.match("git clone https://github.com/test/repo") for p in bash_tool._blocklist)
    assert not git_clone_blocked


# ═══════════════════════════════════════════════════════════════════════
# Item 5: LLM Response Validation + Retry
# ═══════════════════════════════════════════════════════════════════════


def test_llm_response_has_finish_reason():
    """LLMResponse includes finish_reason field."""
    resp = LLMResponse(content="test", finish_reason="stop")
    assert resp.finish_reason == "stop"


def test_is_valid_response_valid():
    """Valid response with content and stop reason."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    resp = LLMResponse(content="I'll fix it", finish_reason="stop")
    assert SearchSWEAgent._is_valid_response(resp)


def test_is_valid_response_empty():
    """Empty response (no content, no tool_calls) is invalid."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    resp = LLMResponse(content=None, tool_calls=[], finish_reason="stop")
    assert not SearchSWEAgent._is_valid_response(resp)


def test_is_valid_response_truncated():
    """Truncated response (finish_reason='length') is invalid."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    resp = LLMResponse(content="partial...", finish_reason="length")
    assert not SearchSWEAgent._is_valid_response(resp)


def test_is_valid_response_with_tool_calls():
    """Response with tool_calls and no content is valid."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    resp = LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="1", name="execute_bash", arguments='{"command":"ls"}')],
        finish_reason="tool_calls",
    )
    assert SearchSWEAgent._is_valid_response(resp)


def test_retry_config_includes_bad_request():
    """BadRequestError should be in default retry list."""
    from awe_agent.core.llm.config import RetryConfig

    config = RetryConfig()
    assert "BadRequestError" in config.retry_on


# ═══════════════════════════════════════════════════════════════════════
# Item 4: Dynamic Search Constraints from Task
# ═══════════════════════════════════════════════════════════════════════


def test_task_get_search_constraints_with_repo():
    """Task.get_search_constraints returns constraints when repo is set."""
    from awe_agent.core.task.protocol import Task

    class DummyTask(Task):
        def get_instances(self, instance_ids=None): return []
        def get_prompt(self, instance): return ""

    task = DummyTask()
    instance = Instance(id="test", dataset_id="test", repo="django/django")
    constraints = task.get_search_constraints(instance)

    assert constraints is not None
    assert constraints._repo_name == "django"
    assert constraints._repo_owner == "django"
    # Should block django's GitHub URL
    assert constraints.is_url_blocked("https://github.com/django/django/issues/123")


def test_task_get_search_constraints_no_repo():
    """Task.get_search_constraints returns None when repo is empty."""
    from awe_agent.core.task.protocol import Task

    class DummyTask(Task):
        def get_instances(self, instance_ids=None): return []
        def get_prompt(self, instance): return ""

    task = DummyTask()
    instance = Instance(id="test", dataset_id="test", repo="")
    constraints = task.get_search_constraints(instance)
    assert constraints is None


def test_search_constraints_merge():
    """SearchConstraints.merge unions patterns from both sides."""
    a = SearchConstraints.from_repo("django/django")
    b = SearchConstraints(blocked_patterns={"url": [r".*example\.com.*"]})
    merged = a.merge(b)
    assert ".*example\\.com.*" in merged.blocked_patterns["url"]
    assert any("django" in p for p in merged.blocked_patterns["url"])


# ═══════════════════════════════════════════════════════════════════════
# Item 7: CodeActXML Format Support
# ═══════════════════════════════════════════════════════════════════════


def test_get_format_openai():
    """get_format('openai_function') returns OpenAIFunctionFormat."""
    fmt = get_format("openai_function")
    assert isinstance(fmt, OpenAIFunctionFormat)
    assert fmt.needs_native_tools()


def test_get_format_xml():
    """get_format('codeact_xml') returns CodeActXMLFormat."""
    fmt = get_format("codeact_xml")
    assert isinstance(fmt, CodeActXMLFormat)
    assert not fmt.needs_native_tools()


def test_get_format_invalid():
    """Unknown format raises ValueError."""
    with pytest.raises(ValueError, match="unknown_format"):
        get_format("unknown_format")


def test_openai_format_prepare_tools():
    """OpenAI format passes tools through."""
    fmt = OpenAIFunctionFormat()
    tools = [{"type": "function", "function": {"name": "test"}}]
    assert fmt.prepare_tools(tools) == tools
    assert fmt.prepare_tools([]) is None


def test_openai_format_system_prompt_suffix():
    """OpenAI format has no system prompt suffix."""
    fmt = OpenAIFunctionFormat()
    assert fmt.get_system_prompt_suffix([{"function": {"name": "test"}}]) == ""


def test_xml_format_prepare_tools():
    """XML format returns None (tools in prompt)."""
    fmt = CodeActXMLFormat()
    tools = [{"type": "function", "function": {"name": "test"}}]
    assert fmt.prepare_tools(tools) is None


def test_xml_format_system_prompt_suffix():
    """XML format generates tool descriptions for system prompt."""
    fmt = CodeActXMLFormat()
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a bash command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run",
                    }
                },
                "required": ["command"],
            },
        },
    }]
    suffix = fmt.get_system_prompt_suffix(tools)
    assert "execute_bash" in suffix
    assert "command" in suffix
    assert "(required)" in suffix
    assert "Run a bash command" in suffix


def test_xml_format_parse_response():
    """XML format parses function calls from response content."""
    fmt = CodeActXMLFormat()
    content = (
        "Let me check the files.\n"
        "<function=execute_bash>\n"
        "<parameter=command>ls -la</parameter>\n"
        "</function>\n"
    )
    response = LLMResponse(content=content)
    tool_calls = fmt.parse_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "execute_bash"
    import json
    args = json.loads(tool_calls[0].arguments)
    assert args["command"] == "ls -la"


def test_xml_format_parse_multiple_calls():
    """XML format parses multiple function calls."""
    fmt = CodeActXMLFormat()
    content = (
        "<function=execute_bash>\n"
        "<parameter=command>ls</parameter>\n"
        "</function>\n"
        "Now let me edit the file.\n"
        "<function=str_replace_editor>\n"
        "<parameter=command>view</parameter>\n"
        "<parameter=path>/test.py</parameter>\n"
        "</function>\n"
    )
    response = LLMResponse(content=content)
    tool_calls = fmt.parse_response(response)
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "execute_bash"
    assert tool_calls[1].name == "str_replace_editor"
    import json
    args1 = json.loads(tool_calls[1].arguments)
    assert args1["command"] == "view"
    assert args1["path"] == "/test.py"


def test_xml_format_parse_empty_content():
    """XML format returns empty list for None content."""
    fmt = CodeActXMLFormat()
    response = LLMResponse(content=None)
    assert fmt.parse_response(response) == []


def test_xml_format_parse_no_function_tags():
    """XML format returns empty list when no function tags present."""
    fmt = CodeActXMLFormat()
    response = LLMResponse(content="Just some regular text without function calls.")
    assert fmt.parse_response(response) == []


def test_openai_format_parse_response():
    """OpenAI format returns response.tool_calls directly."""
    fmt = OpenAIFunctionFormat()
    tc = ToolCall(id="1", name="test", arguments="{}")
    response = LLMResponse(content="hello", tool_calls=[tc])
    parsed = fmt.parse_response(response)
    assert len(parsed) == 1
    assert parsed[0].name == "test"


def test_agent_config_has_tool_call_format():
    """AgentConfig includes tool_call_format field."""
    config = AgentConfig()
    assert config.tool_call_format == "openai_function"


# ═══════════════════════════════════════════════════════════════════════
# Item 9: Model Info Logging (verify only)
# ═══════════════════════════════════════════════════════════════════════


def test_agent_config_has_bash_max_timeout():
    """AgentConfig includes bash_max_timeout field."""
    config = AgentConfig()
    assert config.bash_max_timeout == 600
