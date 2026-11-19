"""A stopped generation must not erase an already-created code submission."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop
from aweagent.core.agent.protocol import Agent
from aweagent.core.llm.types import LLMResponse, Message, ToolCall


class InterruptedAgent(Agent):
    def get_system_prompt(self, task_info):
        return "Implement the specification."

    def get_tools(self):
        return []

    async def step(self, context):
        raise RuntimeError("generation unavailable")


@pytest.mark.asyncio
async def test_generation_failure_preserves_patch_and_error():
    session = SimpleNamespace(get_patch=AsyncMock(return_value="saved implementation"))
    ctx = AgentContext(llm=None, session=session, task_info={"base_commit": "before"})
    result = await AgentLoop(InterruptedAgent(), ctx).run("Build it")
    assert result.patch == "saved implementation"
    assert result.finish_reason == "error"
    assert result.error == "generation unavailable"
    assert "stats" in result.metadata
    session.get_patch.assert_awaited_once_with("/testbed", "before", language="python")


@pytest.mark.asyncio
async def test_operator_cancellation_does_not_collect_or_grade():
    session = SimpleNamespace(get_patch=AsyncMock())
    agent = InterruptedAgent()
    agent.step = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await AgentLoop(agent, AgentContext(llm=None, session=session)).run("Build it")
    session.get_patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_committed_extraction_saves_instruction_and_tool_only_reply():
    call = ToolCall(id="answer", name="finish", arguments='{"answer":"Paris"}')
    llm = SimpleNamespace(chat=AsyncMock(return_value=LLMResponse(tool_calls=[call])))
    ctx = AgentContext(llm=llm, messages=[Message(role="user", content="Question")])
    await AgentLoop(InterruptedAgent(), ctx).append_extraction_turn(
        list(ctx.messages), "Submit the answer", tools=[], commit=True,
    )
    assert [m.role for m in ctx.messages] == ["user", "user", "assistant"]
    assert ctx.messages[-2].content == "Submit the answer"
    assert ctx.messages[-1].tool_calls == [call]


@pytest.mark.asyncio
async def test_git_failure_is_not_an_empty_submission():
    from aweagent.core.runtime.protocol import RuntimeSession
    from aweagent.core.runtime.types import ExecutionResult

    session = SimpleNamespace(
        _update_gitignore=AsyncMock(),
        execute=AsyncMock(return_value=ExecutionResult(
            stdout="", stderr="fatal: bad revision", exit_code=128,
        )),
    )
    with pytest.raises(RuntimeError, match="bad revision"):
        await RuntimeSession.get_patch(session, "/testbed", "before")


def test_saved_trajectory_preserves_reminder_and_collection_diagnostics():
    from aweagent.core.agent.loop import AgentResult
    from aweagent.core.agent.trajectory import Action, Trajectory
    from aweagent.core.task.runner import _build_trajectory_record
    from aweagent.core.task.types import TaskResult

    trajectory = Trajectory()
    step = trajectory.add_step(0, Action(type="message", content="draft"))
    step.metadata["invalid_message_retry"] = {"reason": "no_tool_call"}
    result = TaskResult(instance_id="task", agent_result=AgentResult(
        trajectory=trajectory, error="generation failed",
        metadata={"patch_collection_error": "diff failed"},
    ))
    record = _build_trajectory_record(result)
    assert record["trajectory"][0]["metadata"] == step.metadata
    assert record["agent_error"] == "generation failed"
    assert record["patch_collection_error"] == "diff failed"
