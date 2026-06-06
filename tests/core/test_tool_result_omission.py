from __future__ import annotations

import pytest

from aweagent.core.condenser import build_condenser
from aweagent.core.condenser.tool_result_omission import OMITTED_TOOL_RESULT
from aweagent.core.config.schema import CondenserConfig
from aweagent.core.llm.types import Message, ToolCall


@pytest.mark.asyncio
async def test_tool_result_omission_keeps_recent_native_turns():
    condenser = build_condenser(
        CondenserConfig(type="tool_result_omission", keep_recent_tool_results=1)
    )
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="question"),
        Message(
            role="assistant",
            content="searching",
            tool_calls=[ToolCall(id="tc1", name="web_search", arguments="{}")],
        ),
        Message(role="tool", content="old result", tool_call_id="tc1", name="web_search"),
        Message(
            role="assistant",
            content="fetching",
            tool_calls=[
                ToolCall(id="tc2", name="web_fetch", arguments="{}"),
                ToolCall(id="tc3", name="web_fetch", arguments="{}"),
            ],
        ),
        Message(role="tool", content="new result 1", tool_call_id="tc2", name="web_fetch"),
        Message(role="tool", content="new result 2", tool_call_id="tc3", name="web_fetch"),
    ]

    condensed = await condenser.condense(messages)

    assert condensed[0].content == "system"
    assert condensed[1].content == "question"
    assert condensed[2].tool_calls == messages[2].tool_calls
    assert condensed[3].content == OMITTED_TOOL_RESULT
    assert condensed[5].content == "new result 1"
    assert condensed[6].content == "new result 2"


@pytest.mark.asyncio
async def test_tool_result_omission_zero_omits_all_results():
    condenser = build_condenser(
        CondenserConfig(type="tool_result_omission", keep_recent_tool_results=0)
    )
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="question"),
        Message(role="assistant", content="<function=think></function>"),
        Message(role="user", content="OBSERVATION:\n[think]\nold"),
    ]

    condensed = await condenser.condense(messages)

    assert condensed[1].content == "question"
    assert condensed[2].content == "<function=think></function>"
    assert condensed[3].content == OMITTED_TOOL_RESULT


@pytest.mark.asyncio
async def test_tool_result_omission_minus_one_keeps_all_results():
    condenser = build_condenser(
        CondenserConfig(type="tool_result_omission", keep_recent_tool_results=-1)
    )
    messages = [
        Message(role="user", content="question"),
        Message(role="assistant", content="search"),
        Message(role="tool", content="full result", tool_call_id="tc1", name="web_search"),
    ]

    condensed = await condenser.condense(messages)

    assert condensed[2].content == "full result"
