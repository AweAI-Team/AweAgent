from __future__ import annotations

import sys
import tomllib
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import ModuleType

import pytest

import aweagent.core.condenser.terminus_2 as terminus_condenser
from aweagent.core.condenser.terminus_2 import (
    Terminus2Condenser,
    Terminus2ContextInput,
)
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage


class FakeLLM:
    def __init__(
        self,
        responses: list[str | None | Exception | LLMResponse],
    ) -> None:
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
        if isinstance(response, LLMResponse):
            return response
        return LLMResponse(
            content=response,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class FixedCounter:
    def __init__(self, tokens: int) -> None:
        self.tokens = tokens

    def __call__(self, model_name: str, messages: list[Message]) -> int:
        return self.tokens


class PerMessageCounter:
    def __init__(self, tokens_per_message: int) -> None:
        self.tokens_per_message = tokens_per_message

    def __call__(self, model_name: str, messages: list[Message]) -> int:
        return len(messages) * self.tokens_per_message


def make_terminal_provider(screen: str) -> Callable[[], Awaitable[str]]:
    async def provider() -> str:
        return screen

    return provider


def make_input(
    llm: FakeLLM,
    *,
    messages: list[Message] | None = None,
    limit: int = 131_072,
    screen: str = "terminal-screen",
    reserved_output_tokens: int | None = None,
    preserve_reasoning: bool = True,
) -> Terminus2ContextInput:
    return Terminus2ContextInput(
        messages=messages or [Message(role="user", content="original task")],
        llm=llm,  # type: ignore[arg-type]
        model_name="custom/qwen",
        max_context_length=limit,
        original_instruction="original task",
        terminal_state_provider=make_terminal_provider(screen),
        reserved_output_tokens=(
            0 if reserved_output_tokens is None else reserved_output_tokens
        ),
        preserve_reasoning=preserve_reasoning,
    )


@pytest.mark.parametrize("tokenizer_path", ["", " \t "])
def test_terminus_condenser_normalizes_blank_tokenizer_path(
    tokenizer_path: str,
) -> None:
    condenser = Terminus2Condenser(
        tokenizer_path=tokenizer_path,
        token_counter=lambda model, messages: 1,
    )

    assert condenser.tokenizer_path is None


def install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    auto_tokenizer: type,
) -> None:
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = auto_tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)


def test_project_declares_harbor_dependencies_in_terminus_extra() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    terminus_dependencies = project["project"]["optional-dependencies"][
        "terminus2"
    ]

    assert "litellm==1.86.2" not in dependencies
    assert "transformers==5.12.1" not in dependencies
    assert "litellm==1.86.2" in terminus_dependencies
    assert "transformers==5.12.1" in terminus_dependencies


@pytest.mark.parametrize(
    ("preserve_reasoning", "expected_reasoning"),
    [(False, None), (True, "reason")],
)
def test_default_counter_matches_reasoning_preservation(
    monkeypatch: pytest.MonkeyPatch,
    preserve_reasoning: bool,
    expected_reasoning: str | None,
) -> None:
    counter_calls: list[list[dict[str, object]]] = []

    def load_counter() -> Callable[..., int]:
        def count(*, model: str, messages: list[dict[str, object]]) -> int:
            counter_calls.append(messages)
            return 7

        return count

    monkeypatch.setattr(
        terminus_condenser,
        "_load_litellm_token_counter",
        load_counter,
    )
    condenser = Terminus2Condenser()
    messages = [
        Message(
            role="assistant",
            content="answer",
            reasoning_raw="reason",
        )
    ]

    count = condenser.count_messages(
        "custom/qwen",
        messages,
        preserve_reasoning=preserve_reasoning,
    )

    assert count == 7
    assert counter_calls[0][0].get("reasoning_raw") == expected_reasoning
    assert messages[0].reasoning_raw == "reason"


def test_local_counter_caches_tokenizer_and_uses_chat_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template_calls: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            **kwargs: object,
        ) -> object:
            template_calls.append((messages, kwargs))
            if kwargs.get("return_dict") is False:
                return [1, 2, 3, 4]
            return {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
            }

    class FakeAutoTokenizer:
        load_calls: list[tuple[str, dict[str, object]]] = []

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> FakeTokenizer:
            cls.load_calls.append((path, kwargs))
            return FakeTokenizer()

    install_fake_transformers(monkeypatch, FakeAutoTokenizer)
    terminus_condenser._load_local_tokenizer.cache_clear()

    first = Terminus2Condenser(tokenizer_path=str(tmp_path))
    second = Terminus2Condenser(tokenizer_path=str(tmp_path / "."))
    messages = [
        Message(role="user", content="task"),
        Message(
            role="assistant",
            content="answer",
            reasoning_raw="reason",
        ),
    ]

    assert first.count_messages("ignored", messages) == 4
    assert second.count_messages("ignored", messages) == 4
    assert FakeAutoTokenizer.load_calls == [
        (str(tmp_path.resolve()), {"local_files_only": True})
    ]
    assert template_calls[0][0][1]["reasoning_content"] == "reason"
    assert template_calls[0][1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
    }


def test_local_counter_rejects_missing_tokenizer_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing-model"

    with pytest.raises(ValueError) as exc_info:
        Terminus2Condenser(tokenizer_path=str(missing))

    assert str(missing.resolve()) in str(exc_info.value)


def test_local_counter_reports_chat_template_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BrokenTokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> list[int]:
            raise ValueError("broken template")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> BrokenTokenizer:
            return BrokenTokenizer()

    install_fake_transformers(monkeypatch, FakeAutoTokenizer)
    terminus_condenser._load_local_tokenizer.cache_clear()
    condenser = Terminus2Condenser(tokenizer_path=str(tmp_path))

    with pytest.raises(RuntimeError) as exc_info:
        condenser.count_messages(
            "ignored",
            [Message(role="user", content="task")],
        )

    assert str(tmp_path.resolve()) in str(exc_info.value)
    assert "broken template" in str(exc_info.value)


@pytest.mark.asyncio
async def test_proactive_compaction_only_below_free_token_threshold() -> None:
    counter = FixedCounter(tokens=123_000)
    llm = FakeLLM(["unused"])
    condenser = Terminus2Condenser(
        proactive_threshold=8000,
        token_counter=counter,
    )

    result = await condenser.maybe_condense(make_input(llm))
    assert result.compacted is False
    assert llm.calls == []

    counter.tokens = 123_073
    llm.responses = ["summary", "questions", "answers"]
    result = await condenser.maybe_condense(make_input(llm))

    assert result.compacted is True
    assert result.trigger == "proactive"
    assert result.fallback_level == "full"


@pytest.mark.asyncio
async def test_proactive_compaction_reserves_requested_output_tokens() -> None:
    llm = FakeLLM(["summary", "questions", "answers"])
    condenser = Terminus2Condenser(
        proactive_threshold=8_000,
        token_counter=FixedCounter(tokens=172_225),
    )

    result = await condenser.maybe_condense(
        make_input(
            llm,
            limit=262_144,
            reserved_output_tokens=81_920,
        )
    )

    assert result.compacted is True
    assert result.trace is not None
    assert result.trace.context_limit == 262_144
    assert result.trace.reserved_output_tokens == 81_920
    assert result.trace.effective_input_limit == 180_224
    assert result.trace.free_tokens_before == 7_999


@pytest.mark.asyncio
async def test_full_summary_uses_three_official_subagent_contexts() -> None:
    llm = FakeLLM(["summary text", "question text", "answer text"])
    messages = [
        Message(role="user", content="original task"),
        Message(role="assistant", content="first action"),
        Message(role="user", content="first observation"),
    ]
    condenser = Terminus2Condenser(token_counter=FixedCounter(130_000))

    result = await condenser.maybe_condense(
        make_input(llm, messages=messages, screen="VISIBLE SCREEN")
    )

    assert len(llm.calls) == 3
    assert llm.calls[0][:-1] == messages
    assert "comprehensive summary" in str(llm.calls[0][-1].content)
    assert len(llm.calls[1]) == 1
    assert "summary text" in str(llm.calls[1][0].content)
    assert "VISIBLE SCREEN" in str(llm.calls[1][0].content)
    assert llm.calls[2][: len(messages)] == messages
    assert llm.calls[2][-2].role == "assistant"
    assert llm.calls[2][-2].content == "summary text"
    assert "question text" in str(llm.calls[2][-1].content)

    assert [message.role for message in result.messages] == [
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert result.messages[0] == messages[0]
    assert result.messages[2].content == "question text"
    assert "answer text" in str(result.messages[3].content)
    assert len(result.usage) == 3


@pytest.mark.asyncio
async def test_proactive_output_length_preserves_original_messages() -> None:
    messages = [
        Message(role="user", content="original task"),
        Message(role="assistant", content="existing progress"),
    ]
    llm = FakeLLM(
        [
            LLMResponse(content="partial summary", finish_reason="length"),
            "unused questions",
            "unused answers",
        ]
    )
    condenser = Terminus2Condenser(token_counter=FixedCounter(130_000))

    result = await condenser.maybe_condense(make_input(llm, messages=messages))

    assert result.compacted is False
    assert result.messages == messages
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_reactive_repeated_output_length_uses_terminal_fallback() -> None:
    llm = FakeLLM(
        [
            LLMResponse(content="partial full summary", finish_reason="length"),
            LLMResponse(content="partial short summary", finish_reason="length"),
            "unused questions",
        ]
    )
    condenser = Terminus2Condenser(token_counter=FixedCounter(1))

    result = await condenser.recover_from_context_error(
        make_input(llm, screen="terminal state")
    )

    assert result.compacted is True
    assert result.fallback_level == "terminal"
    assert result.messages[-1].content == (
        "original task\n\nCurrent state: terminal state"
    )
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_full_summary_does_not_use_reasoning_as_stage_content() -> None:
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                reasoning_text="reasoning-only summary",
                finish_reason="stop",
            ),
            "questions",
            "answers",
        ]
    )
    condenser = Terminus2Condenser(token_counter=FixedCounter(130_000))

    result = await condenser.maybe_condense(make_input(llm))

    assert "reasoning-only summary" not in str(llm.calls[1][0].content)
    assert all(
        "reasoning-only summary" not in str(message.content)
        for message in result.messages
    )
    assert result.trace is not None
    assert result.trace.stages[0].response_content == ""
    assert result.trace.stages[0].response_reasoning == "reasoning-only summary"


@pytest.mark.asyncio
async def test_full_summary_records_compact_stage_trace() -> None:
    responses = [
        LLMResponse(
            content="summary text",
            reasoning_text="summary reasoning",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=11, completion_tokens=12, total_tokens=23),
        ),
        LLMResponse(
            content="question text",
            reasoning_text="question reasoning",
            usage=TokenUsage(prompt_tokens=21, completion_tokens=22, total_tokens=43),
        ),
        LLMResponse(
            content="answer text",
            reasoning_text="answer reasoning",
            usage=TokenUsage(prompt_tokens=31, completion_tokens=32, total_tokens=63),
        ),
    ]
    condenser = Terminus2Condenser(token_counter=FixedCounter(130_000))

    result = await condenser.maybe_condense(make_input(FakeLLM(responses)))

    assert result.trace is not None
    assert result.trace.context_limit == 131_072
    assert result.trace.tokens_before == 130_000
    assert result.trace.free_tokens_before == 1_072
    assert result.trace.reserved_output_tokens == 0
    assert result.trace.effective_input_limit == 131_072
    assert [stage.name for stage in result.trace.stages] == [
        "summary",
        "questions",
        "answers",
    ]
    assert result.trace.stages[0].response_content == "summary text"
    assert result.trace.stages[0].response_reasoning == "summary reasoning"
    assert result.trace.stages[0].finish_reason == "stop"
    assert result.trace.stages[0].usage == {
        "prompt_tokens": 11,
        "completion_tokens": 12,
        "total_tokens": 23,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
    }
    assert "comprehensive summary" in result.trace.stages[0].prompt
    assert "question text" in result.trace.stages[2].prompt
    assert result.trace.handoff_prompt == result.messages[-1].content


@pytest.mark.asyncio
async def test_reactive_recovery_unwinds_pairs_until_4000_tokens_are_free() -> None:
    llm = FakeLLM(["summary", "questions", "answers"])
    messages = [
        Message(role="user", content="original task"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="o1"),
        Message(role="assistant", content="a2"),
        Message(role="user", content="o2"),
    ]
    condenser = Terminus2Condenser(
        recovery_target_free_tokens=4000,
        token_counter=PerMessageCounter(3000),
    )

    result = await condenser.recover_from_context_error(
        make_input(llm, messages=messages, limit=10_000)
    )

    assert result.compacted is True
    assert result.trigger == "context_error"
    assert llm.calls[0][0] == messages[0]
    assert llm.calls[0][1].role == "user"
    assert len(llm.calls[0]) == 2


@pytest.mark.asyncio
async def test_reactive_recovery_uses_short_summary_after_full_failure() -> None:
    llm = FakeLLM([RuntimeError("summary failed"), "short continuation"])
    condenser = Terminus2Condenser(token_counter=FixedCounter(1))

    result = await condenser.recover_from_context_error(
        make_input(llm, screen="screen-state")
    )

    assert result.compacted is True
    assert result.fallback_level == "short"
    assert len(llm.calls) == 2
    assert "Briefly continue this task" in str(llm.calls[1][-1].content)
    assert result.messages[-1].content == (
        "original task\n\nSummary: short continuation"
    )


@pytest.mark.asyncio
async def test_reactive_recovery_uses_terminal_tail_after_all_llm_failures() -> None:
    screen = "x" * 1200
    llm = FakeLLM(
        [RuntimeError("summary failed"), RuntimeError("short summary failed")]
    )
    condenser = Terminus2Condenser(token_counter=FixedCounter(1))

    result = await condenser.recover_from_context_error(
        make_input(llm, screen=screen)
    )

    assert result.compacted is True
    assert result.fallback_level == "terminal"
    assert result.messages == [
        Message(role="user", content="original task"),
        Message(
            role="user",
            content=f"original task\n\nCurrent state: {screen[-1000:]}",
        ),
    ]


@pytest.mark.asyncio
async def test_disabled_summarization_never_compacts() -> None:
    llm = FakeLLM(["unused"])
    condenser = Terminus2Condenser(
        enable_summarize=False,
        token_counter=FixedCounter(131_072),
    )

    proactive = await condenser.maybe_condense(make_input(llm))
    reactive = await condenser.recover_from_context_error(make_input(llm))

    assert proactive.compacted is False
    assert reactive.compacted is False
    assert llm.calls == []
