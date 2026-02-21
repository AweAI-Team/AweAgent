"""RunStats — statistics tracking for agent runs."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunStats:
    """Track timing, step count, tool usage, and token counts for an agent run.

    Usage::

        stats = RunStats()
        stats.start()
        # ... agent loop ...
        stats.record_llm_call(elapsed=1.2, prompt_tokens=100, completion_tokens=50)
        stats.record_tool_call("execute_bash", elapsed=0.5)
        stats.end_step()
        stats.finish()
        print(stats.to_dict())
    """

    total_time: float = 0.0
    llm_time: float = 0.0
    tool_time: float = 0.0
    steps: int = 0
    llm_calls: int = 0
    tool_usage: Counter = field(default_factory=Counter)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    _start_time: float = field(default=0.0, repr=False)

    def start(self) -> None:
        """Mark the start of the agent run."""
        self._start_time = time.monotonic()

    def record_llm_call(
        self,
        elapsed: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record an LLM call with timing and token counts."""
        self.llm_time += elapsed
        self.llm_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    def record_tool_call(self, tool_name: str, elapsed: float) -> None:
        """Record a tool execution with timing."""
        self.tool_time += elapsed
        self.tool_usage[tool_name] += 1

    def end_step(self) -> None:
        """Mark the end of a step."""
        self.steps += 1

    def finish(self) -> None:
        """Mark the end of the agent run and compute total time."""
        self.total_time = time.monotonic() - self._start_time

    def to_dict(self) -> dict[str, Any]:
        """Export stats as a plain dict."""
        return {
            "total_time": round(self.total_time, 3),
            "llm_time": round(self.llm_time, 3),
            "tool_time": round(self.tool_time, 3),
            "steps": self.steps,
            "llm_calls": self.llm_calls,
            "tool_usage": dict(self.tool_usage),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
