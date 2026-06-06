"""Agent Protocol — the interface all agents must satisfy.

Design principle: Agent is a stateless "policy" — it maps context to action.
State is managed in AgentContext, execution loop in AgentLoop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.trajectory import Action
from aweagent.core.tool.protocol import Tool

if TYPE_CHECKING:
    from aweagent.core.agent.loop import AgentLoop
    from aweagent.core.agent.policy import LoopPolicy
    from aweagent.core.config.schema import AweAgentConfig
    from aweagent.core.llm.format.protocol import ToolCallFormat


class Agent(ABC):
    """Base class for all agents.

    An agent is a "policy function": given the current context (conversation
    history, tools, etc.), it decides what action to take next.

    The execution loop (AgentLoop) handles the step-by-step cycling.
    This separation makes it trivial to support RL training.
    """

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> Agent:
        """Create an agent instance from the global config.

        Subclasses should override this to extract their own parameters
        from ``config``.  The default implementation calls ``cls()``
        with no arguments, which works for agents that need no config.
        """
        return cls()

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

    def get_loop_policy(self, context: AgentContext) -> LoopPolicy | None:
        """Return the lifecycle policy for this agent's loop.

        A policy shapes the loop's retry / finalize behavior (see
        :class:`~aweagent.core.agent.policy.LoopPolicy`). Return ``None`` for
        a single rollout returned unchanged.
        """
        return None

    def create_loop(self, context: AgentContext) -> AgentLoop:
        """Build the :class:`AgentLoop` that runs this agent.

        The loop carries the agent's :meth:`get_loop_policy`, so an agent
        customizes its lifecycle through the policy rather than the loop.
        """
        from aweagent.core.agent.loop import AgentLoop

        return AgentLoop(self, context, policy=self.get_loop_policy(context))

    def get_tool_call_format(self) -> ToolCallFormat | None:
        """Return the tool call format used by this agent.

        ``AgentLoop`` propagates this to ``AgentContext.tool_call_format``
        so that ``_execute_tools()`` can format observations correctly
        (e.g. native tool role vs. user-message observations).

        Return ``None`` (default) to use native OpenAI function calling.
        Override in subclasses that use text-based formats (XML, JSON).
        """
        return None

    def get_no_tool_call_prompt(self) -> str | None:
        """Return a reminder prompt for when the LLM returns no tool calls.

        When the LLM responds with text but does not invoke any tool, the
        :class:`AgentLoop` will check this value:

        * **Non-None** — the prompt is appended as a ``user`` message and
          the loop continues (gives the LLM another chance to call tools
          or explicitly finish).
        * **None** (default) — no-tool-call is treated as a terminal state
          and the loop breaks immediately.

        Override in subclasses that require explicit ``finish`` tool usage.
        """
        return None
