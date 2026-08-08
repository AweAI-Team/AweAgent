"""Terminus 2 tmux-session lifecycle tests."""

from __future__ import annotations

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import Message
from aweagent.scaffold.terminus_2.agent import Terminus2Agent


class UnexpectedLLM:
    def __init__(self) -> None:
        self.config = LLMConfig(model="test-model")
        self.calls = 0

    async def chat(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("LLM must not be called after the tmux session exits")


class DeadTmux:
    async def is_session_alive(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_dead_tmux_finishes_before_requesting_the_llm() -> None:
    llm = UnexpectedLLM()
    agent = Terminus2Agent()
    agent._initialized = True
    agent._tmux = DeadTmux()  # type: ignore[assignment]
    context = AgentContext(  # type: ignore[arg-type]
        llm=llm,
        messages=[Message(role="user", content="task")],
    )

    action = await agent.step(context)

    assert action.type == "finish"
    assert action.content is None
    assert llm.calls == 0
