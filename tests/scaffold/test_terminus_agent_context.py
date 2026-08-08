from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

import aweagent.scaffold.terminus_2.agent as terminus_agent_module
from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentLoop
from aweagent.core.condenser.terminus_2 import Terminus2Condenser
from aweagent.core.llm.config import LLMConfig
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage
from aweagent.scaffold.terminus_2.agent import Terminus2Agent


class FakeTmux:
    async def is_session_alive(self) -> bool:
        return True

    async def capture_pane(self, capture_entire: bool = False) -> str:
        return "current terminal"


class StaticLLM:
    def __init__(self, responses: list[str]) -> None:
        self.config = LLMConfig(model="custom/qwen")
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        **overrides: object,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(content=self.responses.pop(0))


class OutputLengthLLM:
    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.config = LLMConfig(model="custom/qwen")
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        **overrides: object,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

VALID_COMMAND_RESPONSE = json.dumps(
    {
        "analysis": "continue",
        "plan": "inspect",
        "commands": [{"keystrokes": "pwd\n", "duration": 0.1}],
        "task_complete": False,
    }
)


def test_context_window_error_classification() -> None:
    provider_error = RuntimeError("Bad request")
    provider_error.body = {  # type: ignore[attr-defined]
        "code": "context_length_exceeded"
    }

    assert terminus_agent_module._is_context_window_error(
        RuntimeError("maximum context length exceeded")
    )
    assert terminus_agent_module._is_context_window_error(provider_error)
    assert not terminus_agent_module._is_context_window_error(
        RuntimeError("rate limit exceeded")
    )


def make_initialized_agent_context(
    llm: StaticLLM,
    *,
    max_steps: int = 1,
) -> tuple[Terminus2Agent, AgentContext]:
    agent = Terminus2Agent()
    agent._initialized = True
    agent._tmux = FakeTmux()  # type: ignore[assignment]
    context = AgentContext(
        llm=llm,  # type: ignore[arg-type]
        messages=[Message(role="user", content="original task")],
        task_info={"instruction": "original task"},
        max_steps=max_steps,
    )
    return agent, context


def test_context_management_forwards_model_request_settings() -> None:
    llm = StaticLLM([VALID_COMMAND_RESPONSE])
    llm.config.params["max_tokens"] = 81_920
    llm.config.params["chat_template_kwargs"] = {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    llm.config.reasoning.preserve = True
    agent, context = make_initialized_agent_context(llm)
    context.max_context_length = 262_144

    data = agent._context_management_input(context)

    assert data.reserved_output_tokens == 81_920
    assert data.chat_template_kwargs == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert data.preserve_reasoning is True


class RecoveringLLM:
    def __init__(self) -> None:
        self.config = LLMConfig(model="custom/qwen")
        self.responses: list[str | Exception] = [
            RuntimeError("maximum context length exceeded"),
            "summary",
            "questions",
            "answers",
            json.dumps(
                {
                    "analysis": "continue",
                    "plan": "inspect",
                    "commands": [{"keystrokes": "pwd\n", "duration": 0.1}],
                    "task_complete": False,
                }
            ),
        ]
        self.calls: list[list[Message]] = []

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        **overrides: object,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return LLMResponse(
            content=response,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        )


@pytest.mark.asyncio
async def test_terminus_agent_recovers_context_error_and_retries_main_call() -> None:
    llm = RecoveringLLM()
    agent = Terminus2Agent()
    agent._initialized = True
    agent._tmux = FakeTmux()  # type: ignore[assignment]
    context = AgentContext(
        llm=llm,  # type: ignore[arg-type]
        messages=[
            Message(role="user", content="original task"),
            Message(role="assistant", content="previous response"),
            Message(role="user", content="pending terminal observation"),
        ],
        task_info={"instruction": "original task"},
        max_context_length=131_072,
        condenser=Terminus2Condenser(token_counter=lambda model, messages: 1),
    )

    action = await agent.step(context)

    assert action.type == "tool_call"
    assert len(llm.calls) == 5
    assert llm.calls[1][:-1] == [
        Message(role="user", content="original task"),
        Message(role="assistant", content="previous response"),
    ]
    assert all(
        message.content != "pending terminal observation"
        for message in llm.calls[1]
    )
    assert [message.role for message in context.messages] == [
        "user",
        "user",
        "assistant",
        "user",
    ]
    events = context.trajectory.metadata["context_management"]
    assert events[0]["trigger"] == "context_error"
    assert events[0]["fallback_level"] == "full"


@pytest.mark.asyncio
async def test_context_management_excludes_pending_prompt_but_main_call_keeps_it() -> None:
    llm = StaticLLM([VALID_COMMAND_RESPONSE])
    agent, context = make_initialized_agent_context(llm)
    original_messages = [
        Message(role="user", content="original task"),
        Message(role="assistant", content="previous response"),
        Message(role="user", content="pending terminal observation"),
    ]
    counted_messages: list[list[Message]] = []

    def record_counter(model: str, messages: list[Message]) -> int:
        counted_messages.append(list(messages))
        return 1

    context.messages = list(original_messages)
    context.max_context_length = 131_072
    context.condenser = Terminus2Condenser(token_counter=record_counter)

    action = await agent.step(context)

    assert action.type == "tool_call"
    assert counted_messages == [original_messages[:-1]]
    assert llm.calls == [original_messages]
    assert "context_management" not in context.trajectory.metadata


@pytest.mark.asyncio
async def test_context_management_keeps_initial_prompt_for_token_counting() -> None:
    llm = StaticLLM([VALID_COMMAND_RESPONSE])
    agent, context = make_initialized_agent_context(llm)
    initial_prompt = context.messages[0]
    counted_messages: list[list[Message]] = []

    def record_counter(model: str, messages: list[Message]) -> int:
        counted_messages.append(list(messages))
        return 1

    context.max_context_length = 131_072
    context.condenser = Terminus2Condenser(token_counter=record_counter)

    action = await agent.step(context)

    assert action.type == "tool_call"
    assert counted_messages == [[initial_prompt]]
    assert llm.calls == [[initial_prompt]]


@pytest.mark.asyncio
async def test_output_length_retries_with_exact_harbor_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        SimpleNamespace(
            get_model_info=lambda model: {
                "model": model,
                "max_output_tokens": 8_192,
            }
        ),
    )
    truncated = "preface\n{}"
    llm = OutputLengthLLM(
        [
            LLMResponse(content=truncated, finish_reason="length"),
            LLMResponse(content=VALID_COMMAND_RESPONSE, finish_reason="stop"),
        ]
    )
    agent, context = make_initialized_agent_context(llm)  # type: ignore[arg-type]

    action = await agent.step(context)

    error_prompt = (
        "ERROR!! NONE of the actions you just requested were performed "
        "because you exceeded 8192 tokens. "
        "Your outputs must be less than 8192 tokens. Re-issue this request, "
        "breaking it into chunks each of which is less than 8192 tokens."
        "\n\nParser warnings from your truncated response:\n"
        "- Extra text detected before JSON object"
    )
    retry_messages = [
        Message(role="user", content="original task"),
        Message(role="assistant", content=truncated),
        Message(role="user", content=error_prompt),
    ]
    assert action.type == "tool_call"
    assert llm.calls == [
        [Message(role="user", content="original task")],
        retry_messages,
    ]
    assert context.messages == retry_messages


@pytest.mark.asyncio
async def test_output_length_starts_new_repeated_context_recovery() -> None:
    truncated = "truncated output"
    context_error = RuntimeError("maximum context length exceeded")
    llm = OutputLengthLLM(
        [
            context_error,
            LLMResponse(content="first summary", finish_reason="stop"),
            LLMResponse(content="first questions", finish_reason="stop"),
            LLMResponse(content="first answers", finish_reason="stop"),
            LLMResponse(content=truncated, finish_reason="length"),
            context_error,
            LLMResponse(content="second summary", finish_reason="stop"),
            LLMResponse(content="second questions", finish_reason="stop"),
            LLMResponse(content="second answers", finish_reason="stop"),
            LLMResponse(content=VALID_COMMAND_RESPONSE, finish_reason="stop"),
        ]
    )
    agent, context = make_initialized_agent_context(llm)  # type: ignore[arg-type]
    context.max_context_length = 131_072
    context.condenser = Terminus2Condenser(
        token_counter=lambda model, messages: 1
    )

    action = await agent.step(context)

    assert action.type == "tool_call"
    assert len(llm.calls) == 10
    events = context.trajectory.metadata["context_management"]
    assert [event["trigger"] for event in events] == [
        "context_error",
        "context_error",
    ]


@pytest.mark.asyncio
async def test_post_recovery_failure_uses_technical_difficulties_response() -> None:
    llm = OutputLengthLLM(
        [
            RuntimeError("maximum context length exceeded"),
            LLMResponse(content="summary", finish_reason="stop"),
            LLMResponse(content="questions", finish_reason="stop"),
            LLMResponse(content="answers", finish_reason="stop"),
            RuntimeError("provider unavailable"),
        ]
    )
    agent, context = make_initialized_agent_context(llm)  # type: ignore[arg-type]
    context.max_context_length = 131_072
    context.condenser = Terminus2Condenser(
        token_counter=lambda model, messages: 1
    )

    action = await agent.step(context)

    assert action.type == "message"
    assert action.content == (
        "Technical difficulties. Please continue with the task."
    )
    assert agent.get_no_tool_call_prompt() == (
        "Previous response had parsing errors:\n"
        "ERROR: No valid JSON found in response\n"
        "WARNINGS: - No valid JSON object found\n\n"
        "Please fix these issues and provide a proper JSON response."
    )


@pytest.mark.asyncio
async def test_agent_loop_appends_exact_harbor_parse_error_as_user_message() -> None:
    agent, context = make_initialized_agent_context(StaticLLM(["{}"] * 5))

    await AgentLoop(agent, context).run("original task")

    assert context.messages[-1] == Message(
        role="user",
        content=(
            "Previous response had parsing errors:\n"
            "ERROR: Missing required fields: analysis, plan, commands\n\n"
            "Please fix these issues and provide a proper JSON response."
        ),
    )
