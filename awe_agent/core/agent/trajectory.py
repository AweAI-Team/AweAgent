"""Trajectory — RL-friendly data structures for recording agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """An action taken by the agent."""

    type: str  # "tool_call" | "message" | "finish"
    content: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_text: str | None = None  # Plain-text reasoning for logs/training
    reasoning_raw: Any = None  # Raw provider payload for multi-turn round-trip

    # RL data (populated when using SGLang backend)
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    weight_version: str | None = None
    finish_status: str | None = None  # "stop" | "length"

    # Token usage from LLM response (for stats tracking)
    usage: Any = None
    # Raw LLM response object (for debugging / full-fidelity export)
    llm_response_raw: Any = None


@dataclass
class TrajectoryStep:
    """A single step in the agent trajectory. Contains all data needed for RL."""

    step: int
    action: Action
    observations: list[str] = field(default_factory=list)
    reward: float | None = None
    # Raw LLM response for debugging
    llm_response_raw: Any = None
    # Plain-text reasoning preserved even if API discards it across turns
    reasoning_text: str | None = None


@dataclass
class Trajectory:
    """Complete agent trajectory. Can be exported for RL training."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    final_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        step: int,
        action: Action,
        observations: list[str] | None = None,
        **kwargs: Any,
    ) -> TrajectoryStep:
        ts = TrajectoryStep(
            step=step,
            action=action,
            observations=observations or [],
            **kwargs,
        )
        self.steps.append(ts)
        return ts

    def to_messages(self) -> list[dict[str, Any]]:
        """Export trajectory as a conversation message list."""
        messages: list[dict[str, Any]] = []
        for step in self.steps:
            # Assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if step.action.content:
                assistant_msg["content"] = step.action.content
            if step.reasoning_text or step.action.reasoning_text:
                assistant_msg["reasoning_text"] = step.reasoning_text or step.action.reasoning_text
            if step.action.reasoning_raw is not None:
                assistant_msg["reasoning_raw"] = step.action.reasoning_raw
            if step.action.tool_calls:
                assistant_msg["tool_calls"] = step.action.tool_calls
            messages.append(assistant_msg)

            # Tool observations
            for obs in step.observations:
                messages.append({"role": "tool", "content": obs})
        return messages

    def to_training_format(self) -> dict[str, Any]:
        """Export for RL training (Slime compatible).

        Returns dict with:
            prompt_token_ids: all prompt tokens
            response_token_ids: all response tokens
            logprobs: log probabilities for response tokens
            reward: final reward
            loss_mask: which tokens to include in loss
        """
        all_response_tokens: list[int] = []
        all_logprobs: list[float] = []

        for step in self.steps:
            if step.action.token_ids:
                all_response_tokens.extend(step.action.token_ids)
            if step.action.logprobs:
                all_logprobs.extend(step.action.logprobs)

        # Collect reasoning for training (interleaved reasoning must not be lost)
        reasoning_per_step: list[dict[str, Any]] = []
        for step in self.steps:
            entry: dict[str, Any] = {"step": step.step}
            if step.action.reasoning_text:
                entry["reasoning_text"] = step.action.reasoning_text
            if step.action.reasoning_raw is not None:
                entry["reasoning_raw"] = step.action.reasoning_raw
            if len(entry) > 1:  # has more than just step number
                reasoning_per_step.append(entry)

        result: dict[str, Any] = {
            "response_token_ids": all_response_tokens,
            "logprobs": all_logprobs,
            "reward": self.final_reward,
            "num_steps": len(self.steps),
            "metadata": self.metadata,
        }
        if reasoning_per_step:
            result["reasoning"] = reasoning_per_step
        return result
