"""AgentContext — runtime context for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from awe_agent.core.agent.trajectory import Trajectory
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.types import Message
from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool

# Type for step callbacks: (step, action, observation) -> optional result
StepCallback = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class BashConstraints:
    """Security constraints for bash execution."""

    blocklist: list[str] = field(default_factory=list)
    blocked_urls: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    """All runtime state and dependencies for agent execution.

    This is the single interface between the Agent and the outside world.
    Agent.step() receives this context and uses it to make decisions.
    """

    # Core dependencies (injected)
    llm: LLMClient
    session: RuntimeSession
    tools: list[Tool] = field(default_factory=list)

    # Conversation state
    messages: list[Message] = field(default_factory=list)
    trajectory: Trajectory = field(default_factory=Trajectory)

    # Execution config
    max_steps: int = 100
    max_context_length: int | None = None  # None = no limit
    current_step: int = 0

    # Task info
    task_info: dict[str, Any] = field(default_factory=dict)

    # Callbacks
    step_callbacks: list[StepCallback] = field(default_factory=list)

    # Security
    bash_constraints: BashConstraints | None = None

    # Context condensing (None = no condensing)
    condenser: Any = None

    # Tool call format (None = default OpenAI function calling)
    tool_call_format: Any = None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI function calling schemas for all tools."""
        return [tool.schema for tool in self.tools]

    def get_tool(self, name: str) -> Tool | None:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
