"""Tests for the IterResearch scaffold: agent catalog, Markovian loop, prompts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.config.schema import AweAgentConfig
from awe_agent.core.llm.types import LLMResponse, TokenUsage
from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool
from awe_agent.scaffold.iter_research.agent import IterResearchAgent, _resolve_tool_names
from awe_agent.scaffold.iter_research.loop import IterResearchLoop
from awe_agent.scaffold.iter_research.prompts import (
    check_report_action,
    extract_tags,
    render_template,
)


@pytest.fixture
def mock_llm():
    return AsyncMock()


class _FakeTool(Tool):
    """A tool whose observation is produced by a callable; records its calls."""

    def __init__(self, name: str, responder: Callable[[dict[str, Any]], str]) -> None:
        self._name = name
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"fake {self._name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self, params: dict[str, Any], session: RuntimeSession | None = None
    ) -> str:
        self.calls.append(params)
        return self._responder(params)


def _report_tool_call(name: str, arguments: dict[str, Any]) -> str:
    import json

    return (
        "<report>### Status Report\nworking...</report>\n"
        f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'
    )


def _report_answer(answer: str) -> str:
    return f"<report>### Status Report\ndone</report>\n<answer>\n{answer}\n</answer>"


def _ctx(mock_llm: Any, tools: list[Tool], **kwargs: Any) -> AgentContext:
    kwargs.setdefault("task_info", {"skip_patch_extraction": True, "dataset_id": "browsecomp"})
    kwargs.setdefault("max_steps", 5)
    return AgentContext(llm=mock_llm, tools=tools, **kwargs)


# ── Agent catalog / construction ───────────────────────────────────────────


def test_default_tools_and_external_names():
    agent = IterResearchAgent()
    assert [t.name for t in agent.get_tools()] == [
        "web_search",
        "web_fetch",
        "python_interpreter",
    ]
    # The model-facing (external) names, not the AweAgent internal names.
    assert set(agent.tool_specs) == {"google_search", "Visit", "PythonInterpreter"}
    assert "google_search" in agent.tool_str
    assert "Visit" in agent.tool_str
    assert "PythonInterpreter" in agent.tool_str
    # scholar was removed entirely.
    assert "google_scholar" not in agent.tool_str


def test_default_sampling_and_knobs_are_faithful():
    agent = IterResearchAgent()
    assert agent.sampling == {"temperature": 0.6, "top_p": 0.95, "presence_penalty": 1.5}
    assert agent.max_format_retries == 5
    assert agent.observation_max_tokens == 32000
    assert agent.tokenizer_encoding == "o200k_base"


def test_empty_system_prompt():
    # All instruction lives in the per-turn user message the loop reconstructs.
    assert IterResearchAgent().get_system_prompt({"dataset_id": "browsecomp"}) == ""


def test_create_loop_returns_iter_research_loop():
    agent = IterResearchAgent()
    loop = agent.create_loop(_ctx(None, agent.get_tools()))
    assert isinstance(loop, IterResearchLoop)


def test_from_config_overrides_via_extra():
    config = AweAgentConfig(
        agent={"type": "iter_research"},
        extra={
            "iter_research": {
                "sampling": {"temperature": 0.9},
                "max_format_retries": 2,
                "observation_max_tokens": 1000,
                "tokenizer_encoding": "cl100k_base",
            }
        },
    )
    agent = IterResearchAgent.from_config(config)
    # Sampling override merges over the faithful defaults.
    assert agent.sampling == {"temperature": 0.9, "top_p": 0.95, "presence_penalty": 1.5}
    assert agent.max_format_retries == 2
    assert agent.observation_max_tokens == 1000
    assert agent.tokenizer_encoding == "cl100k_base"


def test_from_config_warns_on_token_cap(caplog):
    config = AweAgentConfig(
        llm={"params": {"max_tokens": 4096}}, agent={"type": "iter_research"}
    )
    with caplog.at_level("WARNING"):
        IterResearchAgent.from_config(config)
    assert any("uncapped generation" in r.message for r in caplog.records)


def test_resolve_tool_names_default_and_explicit():
    assert _resolve_tool_names(AweAgentConfig(agent={"type": "iter_research"})) == [
        "web_search",
        "web_fetch",
        "python_interpreter",
    ]
    config = AweAgentConfig(
        agent={"type": "iter_research", "tools": ["web_search", "python_interpreter"]}
    )
    assert _resolve_tool_names(config) == ["web_search", "python_interpreter"]


def test_from_config_with_constraints_builds_agent():
    config = AweAgentConfig(
        agent={"type": "iter_research"},
        security={"blocked_search_patterns": {"url": [r".*example\.com.*"]}},
    )
    from awe_agent.core.tool.search import SearchConstraints

    agent = IterResearchAgent.from_config_with_constraints(
        config, SearchConstraints(blocked_patterns={"title": [r".*leak.*"]})
    )
    assert [t.name for t in agent.get_tools()] == [
        "web_search",
        "web_fetch",
        "python_interpreter",
    ]


# ── Loop: terminal answer + scoring contract ────────────────────────────────


@pytest.mark.asyncio
async def test_terminal_answer_populates_final_answer(mock_llm):
    async def chat(messages, tools=None, **kwargs):
        return LLMResponse(
            content=_report_answer("The capital is Paris."),
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )

    mock_llm.chat = chat
    agent = IterResearchAgent()
    result = await agent.create_loop(_ctx(mock_llm, agent.get_tools())).run("Capital of France?")

    assert result.finish_reason == "finish"
    assert result.metadata["final_answer"] == "The capital is Paris."
    assert result.metadata["answer_provenance"]["agent_submitted_final_answer"] is True
    assert len(result.trajectory.steps) == 1


@pytest.mark.asyncio
async def test_text_only_call_passes_sampling_and_no_tools(mock_llm):
    seen: dict[str, Any] = {}

    async def chat(messages, tools=None, **kwargs):
        seen["tools"] = tools
        seen["kwargs"] = kwargs
        return LLMResponse(content=_report_answer("A"))

    mock_llm.chat = chat
    agent = IterResearchAgent()
    await agent.create_loop(_ctx(mock_llm, agent.get_tools())).run("Q?")

    assert seen["tools"] is None  # schemas live in the prompt, not native FC
    assert seen["kwargs"]["temperature"] == 0.6
    assert seen["kwargs"]["top_p"] == 0.95
    assert seen["kwargs"]["presence_penalty"] == 1.5
    assert "max_tokens" not in seen["kwargs"]  # no token cap


# ── Loop: Markovian state reconstruction ─────────────────────────────────────


@pytest.mark.asyncio
async def test_markovian_reset_rebuilds_single_user_message(mock_llm):
    windows: list[list[Any]] = []

    async def chat(messages, tools=None, **kwargs):
        windows.append(messages)
        if len(windows) == 1:
            return LLMResponse(
                content=_report_tool_call("google_search", {"query": ["eiffel tower"]})
            )
        return LLMResponse(content=_report_answer("Paris"))

    mock_llm.chat = chat
    search = _FakeTool("web_search", lambda p: "SEARCH_OBSERVATION_TEXT")
    agent = IterResearchAgent()
    result = await agent.create_loop(_ctx(mock_llm, [search])).run("Where is the Eiffel Tower?")

    assert result.metadata["final_answer"] == "Paris"
    # Tool was dispatched via the google_search -> web_search alias with raw args.
    assert search.calls == [{"query": ["eiffel tower"]}]

    # Turn 0: a single fresh user message (the initial instruction).
    assert len(windows[0]) == 1 and windows[0][0].role == "user"
    assert "Where is the Eiffel Tower?" in windows[0][0].content
    # Turn 1: the window was DISCARDED and rebuilt to one user message carrying
    # the prior report, action, and observation — never an accumulating history.
    assert len(windows[1]) == 1 and windows[1][0].role == "user"
    reconstructed = windows[1][0].content
    assert "SEARCH_OBSERVATION_TEXT" in reconstructed
    assert "Last Tool Response" in reconstructed
    assert "eiffel tower" in reconstructed  # the last action is echoed back


@pytest.mark.asyncio
async def test_visit_fans_out_over_all_urls(mock_llm):
    async def chat(messages, tools=None, **kwargs):
        if not hasattr(chat, "done"):
            chat.done = True
            return LLMResponse(
                content=_report_tool_call(
                    "Visit", {"url": ["http://a", "http://b"], "goal": "find X"}
                )
            )
        return LLMResponse(content=_report_answer("answer"))

    mock_llm.chat = chat
    fetch = _FakeTool("web_fetch", lambda p: f"FETCHED[{p['url']}]")
    agent = IterResearchAgent()
    await agent.create_loop(_ctx(mock_llm, [fetch])).run("Q?")

    # Every URL is visited; goal maps to web_fetch's prompt.
    assert fetch.calls == [
        {"url": "http://a", "prompt": "find X"},
        {"url": "http://b", "prompt": "find X"},
    ]


@pytest.mark.asyncio
async def test_forced_last_turn_switches_prompt(mock_llm):
    windows: list[str] = []

    async def chat(messages, tools=None, **kwargs):
        windows.append(messages[0].content)
        # Never answer; always call a tool, so the budget runs out.
        return LLMResponse(content=_report_tool_call("google_search", {"query": ["x"]}))

    mock_llm.chat = chat
    search = _FakeTool("web_search", lambda p: "obs")
    agent = IterResearchAgent()
    result = await agent.create_loop(_ctx(mock_llm, [search], max_steps=2)).run("Q?")

    # max_turn=2 → is_last_turn set at turn==0; turn 1's prompt is the last-turn one.
    assert "you MUST give a final response" not in windows[0]
    assert "you MUST give a final response" in windows[1]
    assert result.finish_reason == "max_steps"
    assert result.metadata["final_answer"] == ""


# ── Loop: format-retry behavior ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_format_retry_regenerates_until_valid(mock_llm):
    calls = {"n": 0}

    async def chat(messages, tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(content="no tags at all")  # invalid: missing <report>
        return LLMResponse(content=_report_answer("recovered"))

    mock_llm.chat = chat
    agent = IterResearchAgent()
    result = await agent.create_loop(_ctx(mock_llm, agent.get_tools())).run("Q?")

    assert calls["n"] == 2  # retried once within the same turn
    assert result.metadata["final_answer"] == "recovered"
    assert len(result.trajectory.steps) == 1  # still a single turn


@pytest.mark.asyncio
async def test_format_retry_exhausts_and_returns_last(mock_llm):
    calls = {"n": 0}

    async def chat(messages, tools=None, **kwargs):
        calls["n"] += 1
        return LLMResponse(content="always invalid, no report tag")

    mock_llm.chat = chat
    agent = IterResearchAgent(max_format_retries=3)
    result = await agent.create_loop(_ctx(mock_llm, agent.get_tools(), max_steps=1)).run("Q?")

    assert calls["n"] == 3  # exhausted the retries for the single turn
    # Tolerant: the last (invalid) response is still used as the terminal turn.
    assert result.finish_reason == "finish"


@pytest.mark.asyncio
async def test_format_retry_appends_correction(mock_llm):
    """A failed format check turns the retry into a repair turn: the window is
    re-sent with the malformed attempt + an explicit correction (not resent as-is)."""
    seen: list[list[Any]] = []
    calls = {"n": 0}

    async def chat(messages, tools=None, **kwargs):
        seen.append(list(messages))
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(content="no tags at all")  # invalid: missing <report>
        return LLMResponse(content=_report_answer("recovered"))

    mock_llm.chat = chat
    agent = IterResearchAgent()
    await agent.create_loop(_ctx(mock_llm, agent.get_tools())).run("Q?")

    assert len(seen[0]) == 1  # first attempt: just the Markovian user window
    # retry: window + assistant(failed attempt) + user(correction)
    assert len(seen[1]) == 3
    assert seen[1][1].role == "assistant" and seen[1][1].content == "no tags at all"
    assert seen[1][2].role == "user"
    assert "<report>" in seen[1][2].content  # the correction names the required tag


# ── Loop internals: dispatch, truncation, reasoning fallback ─────────────────


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_and_missing_tool(mock_llm):
    agent = IterResearchAgent()
    loop = agent.create_loop(_ctx(mock_llm, []))  # no tools in context

    assert await loop._dispatch({"name": "nope", "arguments": {}}) == "No such tool (nope)."
    # Known alias, but the tool is absent from ctx.tools.
    msg = await loop._dispatch({"name": "google_search", "arguments": {"query": ["x"]}})
    assert "is not available" in msg


def test_observation_truncation_caps_length():
    agent = IterResearchAgent(observation_max_tokens=50)
    loop = agent.create_loop(_ctx(None, agent.get_tools()))
    long_text = "x" * 5000
    out = loop._truncate_observation(long_text)
    assert 0 < len(out) < len(long_text)
    # Short observations are untouched.
    assert loop._truncate_observation("short") == "short"


def test_assistant_text_falls_back_to_reasoning():
    agent = IterResearchAgent()
    loop = agent.create_loop(_ctx(None, agent.get_tools()))

    # content carries the tags → use content.
    r1 = LLMResponse(content="<report>r</report><answer>A</answer>", reasoning_text="cot")
    assert loop._assistant_text(r1) == "<report>r</report><answer>A</answer>"

    # content lacks tags but the reasoning channel carries them → use reasoning.
    r2 = LLMResponse(content="thin shell", reasoning_text="<report>r</report><answer>A</answer>")
    assert loop._assistant_text(r2) == "<report>r</report><answer>A</answer>"

    assert loop._assistant_text(None) == ""


# ── Prompt helpers ───────────────────────────────────────────────────────────


def test_extract_tags():
    assert extract_tags("<a> hi </a>", "a") == "hi"
    assert extract_tags("nothing here", "a") == ""


def test_check_report_action_rules():
    import json

    ok_tool = "<report>r</report><tool_call>" + json.dumps(
        {"name": "x", "arguments": {}}
    ) + "</tool_call>"
    assert check_report_action(ok_tool)[0] is True
    assert check_report_action("<report>r</report><answer>a</answer>")[0] is True

    # Missing report.
    assert check_report_action("<answer>a</answer>")[0] is False
    # Tool call that isn't valid JSON.
    assert check_report_action("<report>r</report><tool_call>{bad}</tool_call>")[0] is False
    # Report only, no answer/action.
    assert check_report_action("<report>r</report>")[0] is False
    # Tool call missing the required 'arguments' key.
    assert check_report_action(
        '<report>r</report><tool_call>{"name": "x"}</tool_call>'
    )[0] is False


def test_render_template_is_single_pass():
    template = "{question} | {observation} | {tools}"
    out = render_template(
        template,
        {
            "{question}": "Q",
            "{observation}": "contains a literal {tools} token",
            "{tools}": "TOOLS",
        },
    )
    # The {tools} that arrived *inside* the observation must NOT be re-expanded.
    assert out == "Q | contains a literal {tools} token | TOOLS"
