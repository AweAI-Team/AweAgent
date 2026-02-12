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
    thinking: str | None = None

    # RL data (populated when using SGLang backend)
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None


@dataclass
class TrajectoryStep:
    """A single step in the agent trajectory. Contains all data needed for RL."""

    step: int
    action: Action
    observations: list[str] = field(default_factory=list)
    reward: float | None = None
    # Raw LLM response for debugging
    llm_response_raw: Any = None


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
            if step.action.tool_calls:
                assistant_msg["tool_calls"] = step.action.tool_calls
            messages.append(assistant_msg)

            # Tool observations
            for obs in step.observations:
                messages.append({"role": "tool", "content": obs})
        return messages

    def to_training_format(self) -> dict[str, Any]:
        """Export for RL training (Slime/AlphaSeed compatible).

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

        return {
            "response_token_ids": all_response_tokens,
            "logprobs": all_logprobs,
            "reward": self.final_reward,
            "num_steps": len(self.steps),
            "metadata": self.metadata,
        }
