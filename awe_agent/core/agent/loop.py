"""AgentLoop — the execution engine that runs agents step by step.

Design:
- AgentLoop owns the loop; Agent owns the policy (step function).
- This separation allows RL frameworks to control the loop externally.
- Supports step callbacks for intermediate evaluation, data collection, etc.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.trajectory import Action, Trajectory
from awe_agent.core.llm.types import Message

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of an agent run."""

    trajectory: Trajectory
    patch: str = ""
    messages: list[Message] = field(default_factory=list)
    finish_reason: str = ""  # "finish" | "max_steps" | "error"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Executes an agent step-by-step.

    Usage:
        agent = SearchSWEAgent(enable_search=True)
        ctx = AgentContext(llm=llm, session=session, tools=agent.get_tools())
        loop = AgentLoop(agent, ctx)
        result = await loop.run("Fix the bug described in the issue")
    """

    def __init__(
        self,
        agent: Any,  # Agent protocol
        context: AgentContext,
    ) -> None:
        self.agent = agent
        self.ctx = context

    async def run(self, task_prompt: str) -> AgentResult:
        """Run the full agent loop until completion or max_steps."""
        # Initialize conversation
        system_prompt = self.agent.get_system_prompt(self.ctx.task_info)
        self.ctx.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task_prompt),
        ]
        self.ctx.trajectory = Trajectory()

        finish_reason = "max_steps"

        for step in range(self.ctx.max_steps):
            self.ctx.current_step = step
            logger.debug("Step %d/%d", step + 1, self.ctx.max_steps)

            try:
                # Agent decides action
                action = await self.agent.step(self.ctx)

                # Record in trajectory
                self.ctx.trajectory.add_step(step=step, action=action)

                # Handle finish
                if action.type == "finish":
                    finish_reason = "finish"
                    if action.content:
                        self.ctx.messages.append(
                            Message(role="assistant", content=action.content)
                        )
                    break

                # Handle message-only (no tool calls)
                if action.type == "message":
                    self.ctx.messages.append(
                        Message(role="assistant", content=action.content)
                    )
                    # If no tool calls, this is usually a terminal state
                    if not action.tool_calls:
                        finish_reason = "finish"
                        break

                # Execute tool calls
                observations = await self._execute_tools(action)

                # Update trajectory with observations
                self.ctx.trajectory.steps[-1].observations = observations

                # Step callbacks
                for callback in self.ctx.step_callbacks:
                    await callback(step, action, observations)

            except Exception as e:
                logger.error("Agent step %d failed: %s", step, e, exc_info=True)
                return AgentResult(
                    trajectory=self.ctx.trajectory,
                    messages=list(self.ctx.messages),
                    finish_reason="error",
                    error=str(e),
                )

        # Extract patch if in a code environment
        patch = ""
        try:
            workdir = self.ctx.task_info.get("workdir", "/testbed")
            base_commit = self.ctx.task_info.get("base_commit")
            patch = await self.ctx.session.get_patch(workdir, base_commit)
        except Exception as e:
            logger.warning("Failed to extract patch: %s", e)

        return AgentResult(
            trajectory=self.ctx.trajectory,
            patch=patch,
            messages=list(self.ctx.messages),
            finish_reason=finish_reason,
        )

    async def run_single_step(
        self, messages: list[Message]
    ) -> tuple[Action, list[str]]:
        """Execute a single step (for RL training integration).

        The RL framework controls the loop, calling this method each iteration.
        """
        self.ctx.messages = messages
        action = await self.agent.step(self.ctx)
        observations: list[str] = []
        if action.type == "tool_call" and action.tool_calls:
            observations = await self._execute_tools(action)
        return action, observations

    async def _execute_tools(self, action: Action) -> list[str]:
        """Execute tool calls and collect observations."""
        observations: list[str] = []

        # Add assistant message with tool calls
        assistant_msg = Message(
            role="assistant",
            content=action.content,
            tool_calls=[
                _make_tool_call_obj(tc) for tc in action.tool_calls
            ] if action.tool_calls else None,
        )
        self.ctx.messages.append(assistant_msg)

        for tc in action.tool_calls:
            tool_name = tc.get("name", tc.get("function", {}).get("name", ""))
            tool_call_id = tc.get("id", "")
            arguments_str = tc.get("arguments", tc.get("function", {}).get("arguments", "{}"))

            tool = self.ctx.get_tool(tool_name)
            if tool is None:
                obs = f"Error: tool '{tool_name}' not found."
            else:
                try:
                    params = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                    obs = await tool.execute(params, session=self.ctx.session)
                except json.JSONDecodeError:
                    obs = f"Error: invalid JSON arguments: {arguments_str}"
                except Exception as e:
                    obs = f"Error executing {tool_name}: {e}"

            observations.append(obs)
            self.ctx.messages.append(Message(
                role="tool",
                content=obs,
                tool_call_id=tool_call_id,
                name=tool_name,
            ))

        return observations


def _make_tool_call_obj(tc: dict[str, Any]) -> Any:
    """Convert dict tool call to ToolCall object for Message."""
    from awe_agent.core.llm.types import ToolCall
    name = tc.get("name", tc.get("function", {}).get("name", ""))
    arguments = tc.get("arguments", tc.get("function", {}).get("arguments", "{}"))
    return ToolCall(id=tc.get("id", ""), name=name, arguments=arguments)
