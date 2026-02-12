"""Agent Protocol — the interface all agents must satisfy.

Design principle: Agent is a stateless "policy" — it maps context to action.
State is managed in AgentContext, execution loop in AgentLoop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.trajectory import Action
from awe_agent.core.tool.protocol import Tool


class Agent(ABC):
    """Base class for all agents.

    An agent is a "policy function": given the current context (conversation
    history, tools, etc.), it decides what action to take next.

    The execution loop (AgentLoop) handles the step-by-step cycling.
    This separation makes it trivial to support RL training.
    """

    @abstractmethod
    async def step(self, context: AgentContext) -> Action:
        """Single-step decision: observe context, return an action.

        This is the core method. For RL training, this maps to a single
        policy forward pass + environment step.
        """
        ...

    @abstractmethod
    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        """Generate the system prompt for this agent."""
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return the tools this agent uses."""
        ...
