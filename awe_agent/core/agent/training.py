"""TrainingState — token-level accumulation for RL training.

During RL rollout with SGLang, the agent loop operates in *continuation mode*:
each LLM call sends the full ``input_ids = prompt_token_ids + response_token_ids``
to the ``/generate`` endpoint rather than a message list.  The response contains
newly generated tokens and their log-probabilities, plus a ``weight_version``
that tracks model staleness.

TrainingState is the single accumulator that lives inside :class:`AgentContext`.
It is ``None`` during inference, so all training logic is zero-cost when unused.

Data flow (per step)::

    ┌─────────────┐  token_ids + logprobs   ┌────────────────────┐
    │  SGLang      │ ──────────────────────▶ │  append_model_     │
    │  /generate   │                         │  tokens(mask=1)    │
    └─────────────┘                          └────────────────────┘
                                                      │
    ┌─────────────┐  observation text        ┌────────▼───────────┐
    │  Tool exec   │ ──────────────────────▶ │  append_observation│
    │  (env)       │                         │  _tokens(mask=0)   │
    └─────────────┘                          └────────────────────┘

The ``loss_mask`` distinguishes model-generated tokens (``1``) from environment
observation tokens (``0``), ensuring that the RL loss is only computed on the
policy's own outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Token-level state accumulated across a multi-turn RL rollout.

    Injected into :class:`AgentContext` when training mode is active.
    The agent loop calls the ``append_*`` helpers after each LLM response
    and tool execution; the final ``to_rl_data()`` output is consumed by
    the Slime training framework.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer with
            ``apply_chat_template`` and ``__call__`` (encode) support.
        max_new_tokens: Token budget for the full sequence (prompt + response).
            Used to detect overflow before calling the LLM.
    """

    tokenizer: Any

    # Token budget — mirrors SGLang's sampling_params.max_new_tokens.
    # When ``len(prompt_token_ids) + len(response_token_ids)`` approaches this
    # limit, the SGLang backend will return early to avoid OOM.
    max_new_tokens: int = 32768

    # ── Accumulated state ─────────────────────────────────────────────
    prompt_token_ids: list[int] = field(default_factory=list)
    response_token_ids: list[int] = field(default_factory=list)
    response_text: str = ""
    loss_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    weight_versions: list[str] = field(default_factory=list)

    # Terminal status: "stop" (explicit finish), "length" (token budget
    # exhausted), or "abort" (unrecoverable error).
    finish_status: str = "stop"

    # ── Initialisation ────────────────────────────────────────────────

    def init_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: Any | None = None,
    ) -> None:
        """Tokenize the initial conversation into ``prompt_token_ids``.

        Must be called once, after the system and user messages are set,
        before the first ``agent.step()`` call.

        Args:
            messages: Conversation messages as plain dicts
                (``[{"role": "system", ...}, {"role": "user", ...}]``).
            tools: Optional tool schemas passed to ``apply_chat_template``
                for models that embed tool descriptions in the template.
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.prompt_token_ids = self.tokenizer(
            text, add_special_tokens=False,
        )["input_ids"]

    # ── Token accumulation ────────────────────────────────────────────

    def append_model_tokens(
        self,
        token_ids: list[int],
        logprobs: list[float],
        weight_version: str = "",
    ) -> None:
        """Record model-generated tokens (``loss_mask = 1``).

        Called by the agent loop after each LLM response.

        Args:
            token_ids: Newly generated token IDs from SGLang.
            logprobs: Corresponding log-probabilities (same length).
            weight_version: Model weight version string for staleness
                detection.  Empty string is ignored.
        """
        self.response_token_ids.extend(token_ids)
        self.loss_mask.extend([1] * len(token_ids))
        self.rollout_log_probs.extend(logprobs)
        if weight_version:
            self.weight_versions.append(weight_version)

    def append_observation_tokens(
        self,
        observation_message: dict[str, Any],
        is_final: bool = False,
    ) -> None:
        """Tokenize and record an environment observation (``loss_mask = 0``).

        Replicates the exact tokenization pattern used by the RL
        training pipeline: the observation message is formatted via
        ``apply_chat_template``, then — unless this is the final step —
        the assistant generation header is appended so that the next
        continuation call produces tokens in the correct context.

        Args:
            observation_message: A single message dict with at minimum
                ``role`` and ``content`` keys.  Typically
                ``{"role": "tool", "content": "...", "tool_call_id": "..."}``.
            is_final: If ``True``, omit the trailing assistant header
                (the agent is done, no more generation follows).
        """
        # Format the observation using the tokenizer's chat template.
        # Passing a single-message list produces just that message's tokens
        # (e.g. ``<|im_start|>tool\n{content}<|im_end|>\n``).
        formatted = self.tokenizer.apply_chat_template(
            [observation_message],
            tokenize=False,
            add_generation_prompt=False,
        )
        text = f"\n{formatted}"

        if not is_final:
            # Append the assistant generation header so the next /generate
            # call continues seamlessly.  This is model-specific; Qwen-style
            # chat templates use ``<|im_start|>assistant\n``.
            text += self._assistant_header()

        obs_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]

        self.response_text += text
        self.response_token_ids.extend(obs_ids)
        self.loss_mask.extend([0] * len(obs_ids))
        self.rollout_log_probs.extend([0.0] * len(obs_ids))

    # ── Queries ───────────────────────────────────────────────────────

    def get_input_ids(self) -> list[int]:
        """Full token sequence for the next SGLang continuation call.

        Returns:
            ``prompt_token_ids + response_token_ids``
        """
        return self.prompt_token_ids + self.response_token_ids

    def remaining_budget(self) -> int:
        """Number of tokens the model can still generate.

        Returns a negative value when the budget is exhausted — the SGLang
        backend uses this to short-circuit and return a ``length`` status.
        """
        return self.max_new_tokens - len(self.get_input_ids()) - 128

    # ── Export ─────────────────────────────────────────────────────────

    def to_rl_data(self) -> dict[str, Any]:
        """Export accumulated data in the format expected by Slime.

        The returned dict maps directly to ``Sample`` fields::

            sample.tokens           = rl_data["prompt_token_ids"] + rl_data["response_token_ids"]
            sample.response_length  = len(rl_data["response_token_ids"])
            sample.loss_mask        = rl_data["loss_mask"]
            sample.rollout_log_probs = rl_data["rollout_log_probs"]
            sample.weight_versions  = rl_data["weight_versions"]
        """
        return {
            "prompt_token_ids": list(self.prompt_token_ids),
            "response_token_ids": list(self.response_token_ids),
            "response_text": self.response_text,
            "loss_mask": list(self.loss_mask),
            "rollout_log_probs": list(self.rollout_log_probs),
            "weight_versions": list(self.weight_versions),
            "status": self.finish_status,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _assistant_header(self) -> str:
        """Derive the assistant generation header from the tokenizer.

        Uses ``apply_chat_template`` on a minimal assistant-start message
        and extracts the header prefix.  Falls back to the Qwen/ChatML
        default ``<|im_start|>assistant\\n`` if detection fails.
        """
        try:
            # Generate a template with a dummy assistant message and extract
            # the header that precedes the content.
            sentinel = "__SENTINEL__"
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": sentinel}],
                tokenize=False,
                add_generation_prompt=False,
            )
            idx = rendered.find(sentinel)
            if idx > 0:
                return rendered[:idx]
        except Exception:
            pass
        # Fallback: Qwen / ChatML style
        return "<|im_start|>assistant\n"
