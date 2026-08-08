"""Harbor-compatible semantic context condenser for Terminus 2."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aweagent.core.condenser.protocol import Condenser
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage

if TYPE_CHECKING:
    from aweagent.core.llm.client import LLMClient

logger = logging.getLogger(__name__)

TokenCounter = Callable[[str, list[Message]], int]


@dataclass(frozen=True)
class Terminus2ContextInput:
    """Dependencies and state needed for Terminus 2 context compaction."""

    messages: list[Message]
    llm: LLMClient
    model_name: str
    max_context_length: int
    original_instruction: str
    terminal_state_provider: Callable[[], Awaitable[str]]
    reserved_output_tokens: int = 0
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    preserve_reasoning: bool = True


@dataclass(frozen=True)
class Terminus2ContextStageTrace:
    """One successful LLM stage contributing to a Terminus 2 handoff."""

    name: str
    prompt: str
    response_content: str
    response_reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Terminus2ContextTrace:
    """Audit record for one successful Terminus 2 context replacement."""

    stages: list[Terminus2ContextStageTrace]
    handoff_prompt: str
    context_limit: int
    tokens_before: int
    free_tokens_before: int
    reserved_output_tokens: int = 0
    effective_input_limit: int | None = None


@dataclass(frozen=True)
class Terminus2ContextResult:
    """Messages and diagnostics produced by Terminus 2 compaction."""

    messages: list[Message]
    compacted: bool
    trigger: str | None = None
    fallback_level: str | None = None
    usage: list[TokenUsage] = field(default_factory=list)
    trace: Terminus2ContextTrace | None = None


@cache
def _load_litellm_token_counter() -> Callable[..., int]:
    """Load LiteLLM once, before concurrent task execution begins."""
    from litellm.utils import token_counter

    return token_counter


def _default_token_counter(model_name: str, messages: list[Message]) -> int:
    """Count messages with Harbor's LiteLLM token counter."""
    message_dicts = [message.to_full_dict() for message in messages]
    token_counter = _load_litellm_token_counter()
    return int(token_counter(model=model_name, messages=message_dicts))


@cache
def _load_local_tokenizer(resolved_path: str) -> Any:
    """Load and process-cache a tokenizer without allowing network access."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            resolved_path,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load local tokenizer from {resolved_path}: {exc}"
        ) from exc


def _message_for_chat_template(message: Message) -> dict[str, Any]:
    """Serialize a message using fields understood by local chat templates."""
    result = message.to_dict()
    if message.role == "assistant" and isinstance(message.reasoning_raw, str):
        result["reasoning_content"] = message.reasoning_raw
    return result


class _LocalTokenizerCounter:
    """Count a complete prompt with a cached local model chat template."""

    def __init__(self, tokenizer_path: str) -> None:
        path = Path(tokenizer_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Local tokenizer path is not a directory: {path}")
        self.path = str(path)
        self.tokenizer = _load_local_tokenizer(self.path)

    def __call__(
        self,
        model_name: str,
        messages: list[Message],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        del model_name
        payload = [_message_for_chat_template(message) for message in messages]
        try:
            token_ids = self.tokenizer.apply_chat_template(
                payload,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **(chat_template_kwargs or {}),
            )
            if not isinstance(token_ids, (list, tuple)):
                raise TypeError(
                    "Local tokenizer chat template did not return "
                    "a token-ID sequence"
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to count tokens with local tokenizer {self.path}: {exc}"
            ) from exc
        return len(token_ids)


class Terminus2Condenser(Condenser):
    """Implement Terminus 2 proactive and context-error summarization."""

    def __init__(
        self,
        *,
        enable_summarize: bool = True,
        proactive_threshold: int = 8000,
        recovery_target_free_tokens: int = 4000,
        tokenizer_path: str | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        if tokenizer_path is not None and not tokenizer_path.strip():
            tokenizer_path = None
        self.enable_summarize = enable_summarize
        self.proactive_threshold = proactive_threshold
        self.recovery_target_free_tokens = recovery_target_free_tokens
        self.tokenizer_path = tokenizer_path
        self._local_token_counter: _LocalTokenizerCounter | None = None
        if token_counter is not None:
            self._token_counter = token_counter
        elif tokenizer_path:
            local_counter = _LocalTokenizerCounter(tokenizer_path)
            self.tokenizer_path = local_counter.path
            self._local_token_counter = local_counter
            self._token_counter = local_counter
        else:
            # LiteLLM's first import may refresh its model map and can take tens
            # of seconds. Do this synchronously while the runner is being built,
            # before TaskRunner starts concurrent async task coroutines.
            _load_litellm_token_counter()
            self._token_counter = _default_token_counter

    async def condense(self, messages: list[Message]) -> list[Message]:
        """Keep legacy condenser calls non-destructive.

        Semantic compaction requires the richer context passed to
        :meth:`maybe_condense` or :meth:`recover_from_context_error`.
        """
        return list(messages)

    def count_messages(
        self,
        model_name: str,
        messages: list[Message],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        preserve_reasoning: bool = True,
    ) -> int:
        counted_messages = (
            messages
            if preserve_reasoning
            else [replace(message, reasoning_raw=None) for message in messages]
        )
        if self._local_token_counter is not None:
            return self._local_token_counter(
                model_name,
                counted_messages,
                chat_template_kwargs=chat_template_kwargs,
            )
        return self._token_counter(model_name, counted_messages)

    @staticmethod
    def _effective_input_limit(data: Terminus2ContextInput) -> int:
        """Return input capacity after reserving the requested completion."""
        return max(
            0,
            data.max_context_length - data.reserved_output_tokens,
        )

    async def maybe_condense(
        self,
        data: Terminus2ContextInput,
    ) -> Terminus2ContextResult:
        messages = list(data.messages)
        if not self.enable_summarize:
            return Terminus2ContextResult(messages=messages, compacted=False)

        tokens_before = self.count_messages(
            data.model_name,
            messages,
            chat_template_kwargs=data.chat_template_kwargs,
            preserve_reasoning=data.preserve_reasoning,
        )
        free_tokens = self._effective_input_limit(data) - tokens_before
        if free_tokens >= self.proactive_threshold:
            return Terminus2ContextResult(messages=messages, compacted=False)

        try:
            compacted, usage, trace = await self._full_summary(
                data,
                messages,
                tokens_before,
            )
        except Exception as exc:
            logger.error("Proactive Terminus 2 summarization failed: %s", exc)
            return Terminus2ContextResult(messages=messages, compacted=False)
        return Terminus2ContextResult(
            messages=compacted,
            compacted=True,
            trigger="proactive",
            fallback_level="full",
            usage=usage,
            trace=trace,
        )

    async def recover_from_context_error(
        self,
        data: Terminus2ContextInput,
    ) -> Terminus2ContextResult:
        messages = list(data.messages)
        if not self.enable_summarize:
            return Terminus2ContextResult(messages=messages, compacted=False)

        tokens_before = self.count_messages(
            data.model_name,
            messages,
            chat_template_kwargs=data.chat_template_kwargs,
            preserve_reasoning=data.preserve_reasoning,
        )
        messages = self._unwind_messages(data, messages)
        try:
            compacted, usage, trace = await self._full_summary(
                data,
                messages,
                tokens_before,
            )
            return Terminus2ContextResult(
                messages=compacted,
                compacted=True,
                trigger="context_error",
                fallback_level="full",
                usage=usage,
                trace=trace,
            )
        except Exception as exc:
            logger.debug("Full Terminus 2 summary failed: %s", exc)

        terminal_state = await data.terminal_state_provider()
        limited_screen = terminal_state[-1000:] if terminal_state else ""
        short_prompt = (
            f"Briefly continue this task: {data.original_instruction}\n\n"
            f"Current state: {limited_screen}\n\nNext steps (2-3 sentences):"
        )
        try:
            response = await self._call(data, [], short_prompt)
            summary_prompt = (
                f"{data.original_instruction}\n\nSummary: {self._content(response)}"
            )
            return Terminus2ContextResult(
                messages=[*messages, Message(role="user", content=summary_prompt)],
                compacted=True,
                trigger="context_error",
                fallback_level="short",
                usage=self._usage(response),
                trace=self._trace(
                    data,
                    tokens_before,
                    [self._stage_trace("short_summary", short_prompt, response)],
                    summary_prompt,
                ),
            )
        except Exception as exc:
            logger.error("Short Terminus 2 summary failed: %s", exc)

        fallback_prompt = (
            f"{data.original_instruction}\n\nCurrent state: {limited_screen}"
        )
        return Terminus2ContextResult(
            messages=[*messages, Message(role="user", content=fallback_prompt)],
            compacted=True,
            trigger="context_error",
            fallback_level="terminal",
            trace=self._trace(
                data,
                tokens_before,
                [],
                fallback_prompt,
            ),
        )

    def _unwind_messages(
        self,
        data: Terminus2ContextInput,
        messages: list[Message],
    ) -> list[Message]:
        while len(messages) > 1:
            free_tokens = self._effective_input_limit(data) - self.count_messages(
                data.model_name,
                messages,
                chat_template_kwargs=data.chat_template_kwargs,
                preserve_reasoning=data.preserve_reasoning,
            )
            if free_tokens >= self.recovery_target_free_tokens:
                break
            messages = messages[:-2]
        return messages

    async def _full_summary(
        self,
        data: Terminus2ContextInput,
        messages: list[Message],
        tokens_before: int,
    ) -> tuple[list[Message], list[TokenUsage], Terminus2ContextTrace]:
        if not messages:
            handoff_prompt = data.original_instruction
            return (
                [Message(role="user", content=handoff_prompt)],
                [],
                self._trace(data, tokens_before, [], handoff_prompt),
            )

        summary_prompt = f"""You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {data.original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""
        summary_response = await self._call(data, messages, summary_prompt)
        summary = self._content(summary_response)
        summary_stage = self._stage_trace(
            "summary",
            summary_prompt,
            summary_response,
        )

        current_screen = await data.terminal_state_provider()
        question_prompt = f"""You are picking up work from a previous AI agent on this task:

**Original Task:** {data.original_instruction}

**Summary from Previous Agent:**
{summary}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""
        questions_response = await self._call(data, [], question_prompt)
        questions = self._content(questions_response)
        questions_stage = self._stage_trace(
            "questions",
            question_prompt,
            questions_response,
        )

        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each of "
            "them one by one in detail:\n\n"
            + questions
        )
        answer_history = [
            *messages,
            Message(role="user", content=summary_prompt),
            Message(role="assistant", content=summary),
        ]
        answers_response = await self._call(
            data,
            answer_history,
            answer_request_prompt,
        )
        answers = self._content(answers_response)
        answers_stage = self._stage_trace(
            "answers",
            answer_request_prompt,
            answers_response,
        )
        handoff_prompt = (
            "Here are the answers the other agent provided.\n\n"
            + answers
            + "\n\nContinue working on this task from where the previous agent left off."
            " You can no longer ask questions. Please follow the spec to interact "
            "with the terminal."
        )
        compacted = [
            messages[0],
            Message(role="user", content=question_prompt),
            Message(role="assistant", content=questions),
            Message(role="user", content=handoff_prompt),
        ]
        usage = [
            *self._usage(summary_response),
            *self._usage(questions_response),
            *self._usage(answers_response),
        ]
        trace = self._trace(
            data,
            tokens_before,
            [summary_stage, questions_stage, answers_stage],
            handoff_prompt,
        )
        return compacted, usage, trace

    @staticmethod
    async def _call(
        data: Terminus2ContextInput,
        history: list[Message],
        prompt: str,
    ) -> LLMResponse:
        response = await data.llm.chat(
            messages=[*history, Message(role="user", content=prompt)],
            tools=None,
        )
        if response.finish_reason == "length":
            usage = response.usage
            logger.warning(
                "Terminus 2 summarization response was truncated: "
                "finish_reason=length, prompt_tokens=%s, "
                "completion_tokens=%s, total_tokens=%s",
                usage.prompt_tokens if usage is not None else None,
                usage.completion_tokens if usage is not None else None,
                usage.total_tokens if usage is not None else None,
            )
            raise RuntimeError(
                "Terminus 2 summarization hit the maximum output length."
            )
        return response

    @staticmethod
    def _content(response: LLMResponse) -> str:
        return response.content or ""

    @staticmethod
    def _stage_trace(
        name: str,
        prompt: str,
        response: LLMResponse,
    ) -> Terminus2ContextStageTrace:
        return Terminus2ContextStageTrace(
            name=name,
            prompt=prompt,
            response_content=response.content or "",
            response_reasoning=response.reasoning_text,
            finish_reason=response.finish_reason,
            usage=asdict(response.usage) if response.usage is not None else {},
        )

    @staticmethod
    def _trace(
        data: Terminus2ContextInput,
        tokens_before: int,
        stages: list[Terminus2ContextStageTrace],
        handoff_prompt: str,
    ) -> Terminus2ContextTrace:
        effective_input_limit = Terminus2Condenser._effective_input_limit(data)
        return Terminus2ContextTrace(
            stages=stages,
            handoff_prompt=handoff_prompt,
            context_limit=data.max_context_length,
            tokens_before=tokens_before,
            free_tokens_before=effective_input_limit - tokens_before,
            reserved_output_tokens=data.reserved_output_tokens,
            effective_input_limit=effective_input_limit,
        )

    @staticmethod
    def _usage(response: LLMResponse) -> list[TokenUsage]:
        return [response.usage] if response.usage is not None else []
