"""Tests for core types (Message, LLMResponse, ExecutionResult, Instance)."""

from __future__ import annotations

from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall
from awe_agent.core.runtime.types import ExecutionResult, FileInfo, RuntimeSessionInfo
from awe_agent.core.task.types import EvalResult, Instance, TaskResult


# ── Message ──────────────────────────────────────────────────────────────────


def test_message_creation():
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_message_to_dict():
    msg = Message(role="assistant", content="Hi there")
    d = msg.to_dict()
    assert d["role"] == "assistant"
    assert d["content"] == "Hi there"


def test_message_from_dict():
    d = {"role": "system", "content": "You are helpful."}
    msg = Message.from_dict(d)
    assert msg.role == "system"
    assert msg.content == "You are helpful."


def test_message_with_tool_call():
    msg = Message(
        role="assistant",
        content="Let me check.",
        tool_calls=[ToolCall(id="tc1", name="bash", arguments='{"command": "ls"}')],
    )
    d = msg.to_dict()
    assert len(d["tool_calls"]) == 1


def test_message_tool_response():
    msg = Message(role="tool", content="output", tool_call_id="tc1", name="bash")
    d = msg.to_dict()
    assert d["tool_call_id"] == "tc1"


# ── LLMResponse ──────────────────────────────────────────────────────────────


def test_llm_response():
    resp = LLMResponse(
        content="Fixed the bug.",
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    assert resp.content == "Fixed the bug."
    assert resp.usage.total_tokens == 150


def test_llm_response_with_thinking():
    resp = LLMResponse(
        content="The fix.",
        thinking="I need to change line 42.",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
    )
    assert resp.thinking == "I need to change line 42."


def test_llm_response_with_tokens():
    resp = LLMResponse(
        content="answer",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=20),
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5, 6],
        logprobs=[-0.1, -0.2, -0.3],
    )
    assert resp.completion_token_ids == [4, 5, 6]
    assert resp.logprobs == [-0.1, -0.2, -0.3]


# ── ExecutionResult ──────────────────────────────────────────────────────────


def test_execution_result_success():
    r = ExecutionResult(stdout="hello", exit_code=0)
    assert r.success is True
    assert r.output == "hello"


def test_execution_result_failure():
    r = ExecutionResult(stdout="", stderr="error msg", exit_code=1)
    assert r.success is False
    assert "error msg" in r.output


def test_execution_result_combined_output():
    r = ExecutionResult(stdout="out", stderr="err", exit_code=0)
    assert "out" in r.output
    assert "err" in r.output


# ── Instance ─────────────────────────────────────────────────────────────────


def test_instance_defaults():
    inst = Instance(id="test-1", dataset_id="swe_bench")
    assert inst.workdir == "/testbed"
    assert inst.language == "python"
    assert inst.setup_commands == []


def test_instance_with_metadata():
    inst = Instance(
        id="test-2",
        dataset_id="beyond_swe",
        metadata={"task_type": "doc2repo"},
    )
    assert inst.metadata["task_type"] == "doc2repo"


# ── EvalResult ───────────────────────────────────────────────────────────────


def test_eval_result_accepted():
    r = EvalResult(accepted=True, score=1.0)
    assert r.accepted is True


def test_eval_result_failed():
    r = EvalResult(accepted=False, score=0.0, details={"reason": "tests_failed"})
    assert r.accepted is False
    assert r.details["reason"] == "tests_failed"


# ── TaskResult ───────────────────────────────────────────────────────────────


def test_task_result_success():
    r = TaskResult(
        instance_id="test-1",
        eval_result=EvalResult(accepted=True, score=1.0),
    )
    assert r.success is True


def test_task_result_failure():
    r = TaskResult(instance_id="test-1", error="container crashed")
    assert r.success is False
