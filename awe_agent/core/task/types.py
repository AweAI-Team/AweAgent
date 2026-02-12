"""Core types for the Task & Evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from awe_agent.core.agent.loop import AgentResult


@dataclass
class Instance:
    """A single task instance (e.g., one SWE-Bench problem)."""

    id: str
    dataset_id: str
    repo: str = ""
    base_commit: str = ""
    workdir: str = "/testbed"
    image: str = ""
    language: str = "python"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Task-specific fields
    problem_statement: str = ""
    gold_patch: str = ""
    test_patch: str = ""
    setup_commands: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating an agent's submission."""

    accepted: bool = False
    score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


@dataclass
class TaskResult:
    """Complete result of running a task instance."""

    instance_id: str
    agent_result: AgentResult | None = None
    eval_result: EvalResult | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.eval_result is not None and self.eval_result.accepted
