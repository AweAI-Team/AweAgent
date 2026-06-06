"""IterResearchLoop — the Markovian state-reconstruction execution engine.

IterResearch does not accumulate a chat transcript. Each turn the model emits a
``<report>`` (its rewritten working memory) plus a ``<tool_call>`` or a final
``<answer>``; after a tool runs, the entire window is **discarded** and the next
turn's single user message is reconstructed from {question, tools, last report,
last action, last observation}. This keeps the context bounded (~one report +
one observation) regardless of horizon, which is what lets the harness sustain
hundreds of tool calls.

This is incompatible with the stock :class:`AgentLoop` (which appends to a
growing ``ctx.messages``), so :class:`~aweagent.scaffold.iter_research.agent.IterResearchAgent`
returns this loop from ``create_loop``. The loop still reuses the framework's
primitives — ``ctx.llm`` for generation, ``ctx.get_tool`` for dispatch,
``ctx.trajectory`` for recording, and an :class:`AgentResult` for the return —
so it plugs into the runner, the trajectory exporter, and the BrowseComp judge
(which reads ``metadata["final_answer"]``) unchanged.

Faithful-reproduction details: a ``<report>`` is required every turn; format is
re-validated with up to ``max_format_retries`` regenerations; observations are
truncated to ``observation_max_tokens`` (two-stage: cheap char gate, then exact
token count); the model is told to finalize one turn before the budget ends
(``turn == max_turn - 2``); sampling has no ``max_tokens`` cap so long reports
and answers are never clipped mid-tag.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.loop import AgentResult
from aweagent.core.agent.trajectory import Action, Trajectory
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage
from aweagent.scaffold.iter_research.prompts import (
    check_report_action,
    extract_tags,
    initial_instruction_prompt,
    instruction_prompt,
    last_instruction_prompt,
    observation_prompt,
    render_template,
)

if TYPE_CHECKING:
    from aweagent.core.agent.context import AgentContext

logger = logging.getLogger(__name__)

# Sentinel for lazily-loaded tiktoken encoder (None means "tried and failed").
_UNSET: Any = object()

_FORMAT_TAGS = ("<report>", "<tool_call>", "<answer>")

# Sent back to the model on a format-retry. Re-sending the identical window at a
# non-zero temperature tends to reproduce the same malformed output (a thinking
# model occasionally spends its turn in the reasoning channel and emits content
# with no <report>). An explicit correction turns the retry into a repair step.
_FORMAT_CORRECTION = (
    "Your previous reply was rejected: its content did not contain a valid "
    "`<report>` section. Reply again now in EXACTLY this format: a "
    "`<report>...</report>` block (your full, updated working memory) followed by "
    "EITHER one `<tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>` OR a "
    "final `<answer>...</answer>`. These tags must appear in your visible reply, "
    "not only in your private reasoning."
)


def _has_format_tags(text: str) -> bool:
    return any(tag in text for tag in _FORMAT_TAGS)


class IterResearchLoop:
    """Run the IterResearch Markovian loop and return a scoreable AgentResult.

    The loop is duck-typed against the runner: it only needs an async
    ``run(task_prompt) -> AgentResult`` method. Configuration is pulled from the
    owning agent (tool schemas, sampling, retry/truncation knobs).
    """

    def __init__(self, agent: Any, context: AgentContext) -> None:
        self.agent = agent
        self.ctx = context
        self._tool_str: str = agent.tool_str
        self._specs: dict[str, Any] = agent.tool_specs  # external name -> _ToolSpec
        self._sampling: dict[str, Any] = dict(agent.sampling)
        self._max_format_retries: int = agent.max_format_retries
        self._obs_max_tokens: int = agent.observation_max_tokens
        self._encoding_name: str = agent.tokenizer_encoding
        self._encoder: Any = _UNSET

    async def run(self, task_prompt: str) -> AgentResult:
        ctx = self.ctx
        question = task_prompt
        date = datetime.now().strftime("%Y-%m-%d")
        max_turn = max(int(ctx.max_steps), 1)

        # The Markovian window: exactly the messages sent to the LLM this turn.
        first_user = render_template(
            initial_instruction_prompt,
            {"{question}": question, "{tools}": self._tool_str, "{date_to_use}": date},
        )
        window: list[Message] = [Message(role="user", content=first_user)]

        # Flat, append-only transcript — kept current so a runner timeout still
        # yields a partial trajectory/messages snapshot.
        ctx.messages = [Message(role="user", content=first_user)]
        ctx.trajectory = Trajectory()

        is_last_turn = False
        final_answer: str | None = None
        finish_reason = "max_steps"
        answer_provenance: dict[str, Any] = {}
        total_usage = TokenUsage()

        for turn in range(max_turn):
            response = await self._generate(window)
            if response is None:
                finish_reason = "error"
                answer_provenance = {"error": "llm_generation_failed", "turn": turn}
                break

            text = self._assistant_text(response)
            self._accumulate_usage(total_usage, response.usage)

            report = extract_tags(text, "report")
            tool_call_str = extract_tags(text, "tool_call")
            answer_str = extract_tags(text, "answer")
            tool_call = self._parse_tool_call(tool_call_str)
            has_tool_call = tool_call is not None

            action = Action(
                type="tool_call" if has_tool_call else "finish",
                content=text,
                reasoning_text=response.reasoning_text,
                reasoning_raw=response.reasoning_raw,
                tool_calls=(
                    [
                        {
                            "name": tool_call.get("name", ""),
                            "arguments": tool_call.get("arguments", {}),
                        }
                    ]
                    if has_tool_call
                    else []
                ),
                finish_status=response.finish_status,
                usage=response.usage,
                llm_response_raw=response.raw,
            )
            ctx.messages.append(
                Message(role="assistant", content=text, reasoning_raw=response.reasoning_raw)
            )
            step = ctx.trajectory.add_step(
                step=turn,
                action=action,
                reasoning_text=action.reasoning_text,
                llm_response_raw=action.llm_response_raw,
            )

            # ── Terminal: no tool call → the answer lives in this turn ───────
            if not has_tool_call:
                final_answer = answer_str if answer_str else text
                finish_reason = "finish"
                answer_provenance = {
                    "agent_submitted_final_answer": True,
                    "forced_final_answer": is_last_turn,
                    "turn": turn,
                }
                break

            # ── Dispatch the tool and record the observation ─────────────────
            tool_name = str(tool_call.get("name", ""))
            observation = await self._dispatch(tool_call)
            step.observations = [observation]
            ctx.messages.append(Message(role="tool", name=tool_name, content=observation))

            # One turn before the budget ends, force a final answer next turn.
            if turn == max_turn - 2:
                is_last_turn = True

            # ── Markovian reset: rebuild one user message, discard the rest ──
            window = [
                Message(
                    role="user",
                    content=self._build_next_prompt(
                        question=question,
                        date=date,
                        report=report,
                        action=tool_call_str,
                        observation=observation,
                        is_last=is_last_turn,
                    ),
                )
            ]

        if final_answer is None:
            final_answer = ""
            answer_provenance.setdefault("agent_submitted_final_answer", False)
            answer_provenance.setdefault("reason", "max_turn_reached_without_answer")

        ctx.trajectory.metadata["iter_research"] = {
            "turns": len(ctx.trajectory.steps),
            "date": date,
        }

        return AgentResult(
            trajectory=ctx.trajectory,
            messages=list(ctx.messages),
            finish_reason=finish_reason,
            metadata={
                "final_answer": final_answer,
                "answer_provenance": answer_provenance,
                "stats": {
                    "turns": len(ctx.trajectory.steps),
                    "prompt_tokens": total_usage.prompt_tokens,
                    "completion_tokens": total_usage.completion_tokens,
                },
            },
        )

    # ── Generation ───────────────────────────────────────────────────────

    async def _generate(self, window: list[Message]) -> LLMResponse | None:
        """Call the LLM (text-only) and re-generate until the format validates.

        Tool schemas are carried in the prompt text (the ``{tools}`` block), not
        via native function calling, so ``tools=None``. Returns the last
        response even if it never validated (matches the reference's tolerance).
        """
        response: LLMResponse | None = None
        work = window
        for attempt in range(self._max_format_retries):
            try:
                response = await self.ctx.llm.chat(
                    messages=work, tools=None, **self._sampling
                )
            except Exception as exc:  # noqa: BLE001 - transient/endpoint errors → retry
                logger.warning(
                    "IterResearch LLM call failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_format_retries,
                    exc,
                )
                work = window  # transient failure, not a format issue — resend as-is
                continue
            text = self._assistant_text(response)
            ok, reason = check_report_action(text)
            if ok:
                return response
            logger.info(
                "IterResearch format retry %d/%d: %s",
                attempt + 1,
                self._max_format_retries,
                reason,
            )
            # Repair turn: show the model its malformed attempt plus an explicit
            # correction, so the retry actively fixes the format instead of
            # resampling the identical window. Transient (never persisted to
            # ctx.messages or the next Markovian window).
            work = [
                *window,
                Message(role="assistant", content=text or "(your previous reply was empty)"),
                Message(role="user", content=_FORMAT_CORRECTION),
            ]
        return response

    def _assistant_text(self, response: LLMResponse | None) -> str:
        """Text to parse tags from: prefer ``content``; fall back to reasoning.

        Reasoning-split models (e.g. DeepSeek) route chain-of-thought to
        ``reasoning_text`` and the answer to ``content``; the ``<report>``/
        ``<tool_call>`` structure belongs in ``content``. As a safety net, if
        ``content`` carries none of the format tags but the reasoning channel
        does, parse the reasoning channel instead.
        """
        if response is None:
            return ""
        content = response.content or ""
        if _has_format_tags(content):
            return content
        reasoning = response.reasoning_text or ""
        if reasoning and _has_format_tags(reasoning):
            return reasoning
        return content

    @staticmethod
    def _parse_tool_call(tool_call_str: str) -> dict[str, Any] | None:
        if not tool_call_str:
            return None
        import json

        try:
            parsed = json.loads(tool_call_str)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    # ── Tool dispatch ──────────────────────────────────────────────────────

    async def _dispatch(self, tool_call: dict[str, Any]) -> str:
        name = str(tool_call.get("name", ""))
        spec = self._specs.get(name)
        if spec is None:
            return f"No such tool ({name})."
        tool = self.ctx.get_tool(spec.internal)
        if tool is None:
            return f"Error: tool '{spec.internal}' is not available."
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            return "ToolParseError!"
        try:
            param_list = spec.adapt(arguments)
            parts = [
                await tool.execute(params, session=self.ctx.session)
                for params in param_list
            ]
        except Exception as exc:  # noqa: BLE001 - tool errors surface as observations
            logger.warning("IterResearch tool %r failed: %s", name, exc)
            return "ToolParseError!"
        if not parts:
            return "No information available."
        return "\n\n".join(str(p) for p in parts)

    # ── Prompt reconstruction ──────────────────────────────────────────────

    def _build_next_prompt(
        self,
        *,
        question: str,
        date: str,
        report: str,
        action: str,
        observation: str,
        is_last: bool,
    ) -> str:
        obs_block = observation_prompt.replace("{tool_response}", observation)
        obs_block = self._truncate_observation(obs_block)
        template = last_instruction_prompt if is_last else instruction_prompt
        return render_template(
            template,
            {
                "{question}": question,
                "{report}": report,
                "{action}": action,
                "{observation}": obs_block,
                "{tools}": self._tool_str,
                "{date_to_use}": date,
            },
        )

    def _truncate_observation(self, text: str) -> str:
        limit = self._obs_max_tokens
        if len(text) <= limit:  # cheap char gate (same threshold as the token cap)
            return text
        encoder = self._get_encoder()
        if encoder is None:
            # Offline / vocab unavailable: rough char-based fallback (~4 chars/token).
            return text[: limit * 4]
        tokens = encoder.encode(text)
        if len(tokens) > limit:
            return encoder.decode(tokens[:limit])
        return text

    def _get_encoder(self) -> Any:
        if self._encoder is _UNSET:
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding(self._encoding_name)
            except Exception as exc:  # noqa: BLE001 - degrade to char truncation
                logger.warning(
                    "tiktoken encoding %r unavailable (%s); using char-based truncation",
                    self._encoding_name,
                    exc,
                )
                self._encoder = None
        return self._encoder

    @staticmethod
    def _accumulate_usage(total: TokenUsage, usage: Any) -> None:
        if usage is None:
            return
        total.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        total.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        total.total_tokens += getattr(usage, "total_tokens", 0) or 0
