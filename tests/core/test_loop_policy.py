from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop
from aweagent.core.agent.policy import CompositePolicy, LoopPolicy, NoOpPolicy
from aweagent.core.agent.trajectory import Trajectory
from aweagent.core.llm.types import LLMResponse, Message


def _loop(llm) -> AgentLoop:
    ctx = AgentContext(llm=llm, tools=[], task_info={"skip_patch_extraction": True})
    ctx.messages = [Message(role="user", content="q")]
    ctx.trajectory = Trajectory()
    return AgentLoop(object(), ctx)


def test_default_policy_is_noop_and_never_retries():
    loop = _loop(AsyncMock())
    assert isinstance(loop.policy, NoOpPolicy)
    assert loop.policy.should_retry(0, object(), object()) is False


class _RecordingPolicy(LoopPolicy):
    def __init__(self, name: str, retry: bool, log: list[str]) -> None:
        self._name = name
        self._retry = retry
        self._log = log

    def should_retry(self, attempt, result, ctx) -> bool:
        return self._retry

    async def finalize(self, loop, result, ctx):
        self._log.append(self._name)
        return result


@pytest.mark.asyncio
async def test_composite_policy_retries_if_any_and_finalizes_in_order():
    log: list[str] = []
    composite = CompositePolicy(
        _RecordingPolicy("a", False, log),
        _RecordingPolicy("b", True, log),
    )
    assert composite.should_retry(0, None, None) is True

    none_retry = CompositePolicy(
        _RecordingPolicy("x", False, log),
        _RecordingPolicy("y", False, log),
    )
    assert none_retry.should_retry(0, None, None) is False

    await composite.finalize(None, "result", None)
    assert log == ["a", "b"]


@pytest.mark.asyncio
async def test_append_extraction_turn_commits_and_records():
    seen: dict = {}

    async def chat(messages, tools=None, **kwargs):
        seen["tools"] = tools
        seen["count"] = len(messages)
        return LLMResponse(content="answer text")

    llm = AsyncMock()
    llm.chat = chat
    loop = _loop(llm)

    caller_view = list(loop.ctx.messages)
    action = await loop.append_extraction_turn(
        caller_view, "state the answer", tools=None, commit=True,
    )

    assert action.content == "answer text"
    assert seen["tools"] is None
    assert seen["count"] == 2  # the original user message + the instruction
    assert len(caller_view) == 1  # caller's list is never mutated
    assert loop.ctx.messages[-1].role == "assistant"
    assert loop.ctx.messages[-1].content == "answer text"
    assert len(loop.ctx.trajectory.steps) == 1


@pytest.mark.asyncio
async def test_append_extraction_turn_without_commit_leaves_messages_untouched():
    async def chat(messages, tools=None, **kwargs):
        return LLMResponse(content="probe reply")

    llm = AsyncMock()
    llm.chat = chat
    loop = _loop(llm)

    action = await loop.append_extraction_turn(
        list(loop.ctx.messages), "probe", commit=False,
    )

    assert action.content == "probe reply"
    assert len(loop.ctx.messages) == 1  # no assistant message committed
    assert len(loop.ctx.trajectory.steps) == 1  # but the turn is still recorded
