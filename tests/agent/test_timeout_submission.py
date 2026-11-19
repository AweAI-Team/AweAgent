"""Deadline expiry still submits the implementation, without another agent run."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aweagent.core.agent.protocol import Agent
from aweagent.core.llm.config import LLMConfig
from aweagent.core.task.protocol import Task
from aweagent.core.task.runner import TaskRunner
from aweagent.core.task.types import EvalResult, Instance


class CodeTask(Task):
    def get_instances(self, ids=None):
        return []

    def get_prompt(self, instance):
        return "Build it"

    def get_task_info(self, instance):
        return {"workdir": "/testbed", "base_commit": "before"}

    def requires_git_snapshot(self):
        return False


class WaitingAgent(Agent):
    def get_system_prompt(self, task_info):
        return "Implement"

    def get_tools(self):
        return []

    async def step(self, context):
        await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("patch_error", [False, True])
async def test_timeout_collects_before_closing_session(tmp_path, monkeypatch, patch_error):
    monkeypatch.setattr("aweagent.core.task.runner.LLMClient", lambda _: None)
    session = SimpleNamespace(get_patch=AsyncMock(return_value="implementation"))
    if patch_error:
        session.get_patch.side_effect = RuntimeError("diff failed")
    task = CodeTask()
    task.prepare_session = AsyncMock()
    task.get_setup_commands = lambda _: []
    runner = TaskRunner(task, lambda **_: WaitingAgent(), LLMConfig(model="fake"), None,
                        evaluator=SimpleNamespace(requires_same_session=False),
                        output_path=tmp_path, agent_timeout_override=0.01)
    @asynccontextmanager
    async def opened(instance):
        yield session
    runner._instance_session = opened
    runner._evaluate = AsyncMock(return_value=EvalResult(score=0.5))
    result = await runner._run_instance(Instance(id="sample", dataset_id="example"))
    assert result.agent_result.finish_reason == "timeout"
    session.get_patch.assert_awaited_once_with("/testbed", "before", language="python")
    if patch_error:
        runner._evaluate.assert_not_awaited()
        assert "diff failed" in result.agent_result.metadata["patch_collection_error"]
    else:
        runner._evaluate.assert_awaited_once()
        assert result.agent_result.patch == "implementation"
        assert result.eval_result.score == 0.5
