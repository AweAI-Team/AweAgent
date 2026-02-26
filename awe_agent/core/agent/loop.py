"""AgentLoop — the execution engine that runs agents step by step.

Design:
- AgentLoop owns the loop; Agent owns the policy (step function).
- This separation allows RL frameworks to control the loop externally.
- Supports step callbacks for intermediate evaluation, data collection, etc.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.stats import RunStats
from awe_agent.core.agent.trajectory import Action, Trajectory
from awe_agent.core.llm.types import Message

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result of an agent run."""

    trajectory: Trajectory
    patch: str = ""
    messages: list[Message] = field(default_factory=list)
    finish_reason: str = ""  # "finish" | "max_steps" | "context_length" | "error"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Executes an agent step-by-step.

    Usage:
        agent = SearchSWEAgent(enable_search=True)
        ctx = AgentContext(llm=llm, session=session, tools=agent.get_tools())
        loop = AgentLoop(agent, ctx)
        result = await loop.run("Fix the bug described in the issue")

    Args:
        agent: An object satisfying the Agent protocol (step / get_tools / ...).
            If the agent implements ``get_no_tool_call_prompt()`` and returns a
            non-None string, the loop will send that prompt as a ``user``
            message whenever the LLM responds without calling any tool, instead
            of treating it as a terminal state.
        context: Mutable runtime state shared with the agent.
    """

    def __init__(
        self,
        agent: Any,  # Agent protocol
        context: AgentContext,
    ) -> None:
        self.agent = agent
        self.ctx = context

    async def run(self, task_prompt: str) -> AgentResult:
        """Run the full agent loop until completion or max_steps.

        Termination conditions (in priority order):

        1. **Context length exceeded** — the previous LLM call's
           ``prompt_tokens`` exceeded ``max_context_length``.  Checked
           *before* the next LLM call to avoid wasting an API request.
        2. **Explicit finish** — ``action.type == "finish"`` (agent called the
           *finish* tool).  Any associated tool calls are executed first so that
           their responses appear in the conversation history.
        3. **No tool call with reminder** — ``action.type == "message"`` (LLM
           returned text without tool calls) *and* the agent provides a
           ``get_no_tool_call_prompt()``.  The reminder is appended as a
           ``user`` message and the loop **continues**.
        4. **No tool call without reminder** — same as (3) but agent returns
           ``None`` → treated as implicit finish.
        5. **Max steps** — loop counter exhausted.
        6. **Error** — any exception during a step.
        """
        # Read no-tool-call prompt from agent (may be None).
        no_tool_call_prompt: str | None = None
        if hasattr(self.agent, "get_no_tool_call_prompt"):
            no_tool_call_prompt = self.agent.get_no_tool_call_prompt()

        # Propagate tool call format from agent to context (for XML mode support)
        if self.ctx.tool_call_format is None and hasattr(self.agent, "_format"):
            self.ctx.tool_call_format = self.agent._format

        # Initialize conversation
        system_prompt = self.agent.get_system_prompt(self.ctx.task_info)
        self.ctx.messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task_prompt),
        ]
        self.ctx.trajectory = Trajectory()

        stats = RunStats()
        stats.start()

        finish_reason = "max_steps"
        # Estimated token count for the next LLM call's input.
        # Updated after each step as: prompt_tokens + completion_tokens,
        # which is a lower bound (tool observations add more).
        estimated_next_context = 0

        for step in range(self.ctx.max_steps):
            self.ctx.current_step = step

            # ── 0. Context length guard ───────────────────────────────
            if (self.ctx.max_context_length is not None
                    and estimated_next_context > self.ctx.max_context_length):
                logger.info(
                    "Estimated context %d exceeds limit %d at step %d, stopping",
                    estimated_next_context, self.ctx.max_context_length, step,
                )
                finish_reason = "context_length"
                break

            logger.debug("Step %d/%d", step + 1, self.ctx.max_steps)

            try:
                # Agent decides action
                llm_start = time.monotonic()
                action = await self.agent.step(self.ctx)
                llm_elapsed = time.monotonic() - llm_start

                # Record LLM call stats
                prompt_tokens = 0
                completion_tokens = 0
                if action.usage is not None:
                    prompt_tokens = getattr(action.usage, "prompt_tokens", 0)
                    completion_tokens = getattr(action.usage, "completion_tokens", 0)
                estimated_next_context = prompt_tokens + completion_tokens
                stats.record_llm_call(llm_elapsed, prompt_tokens, completion_tokens)

                # Record in trajectory
                self.ctx.trajectory.add_step(step=step, action=action)

                # ── 1. Explicit finish (agent called the finish tool) ────
                if action.type == "finish":
                    finish_reason = "finish"
                    if action.tool_calls:
                        # Execute finish (and any companion) tool calls so
                        # their responses are recorded in the conversation.
                        tool_start = time.monotonic()
                        observations = await self._execute_tools(action)
                        tool_elapsed = time.monotonic() - tool_start
                        for tc in action.tool_calls:
                            name = tc.get("name", tc.get("function", {}).get("name", ""))
                            stats.record_tool_call(name, tool_elapsed / max(len(action.tool_calls), 1))
                        self.ctx.trajectory.steps[-1].observations = observations
                    elif action.content:
                        self.ctx.messages.append(
                            Message(role="assistant", content=action.content)
                        )
                    stats.end_step()
                    break

                # ── 2. Message-only (LLM returned no tool calls) ─────────
                if action.type == "message":
                    self.ctx.messages.append(
                        Message(role="assistant", content=action.content)
                    )
                    if not action.tool_calls:
                        if no_tool_call_prompt:
                            # Send a reminder and let the loop continue.
                            logger.info(
                                "Step %d: no tool call — sending reminder",
                                step,
                            )
                            self.ctx.messages.append(
                                Message(
                                    role="user",
                                    content=no_tool_call_prompt,
                                )
                            )
                            stats.end_step()
                            continue
                        # No reminder configured → treat as finish.
                        finish_reason = "finish"
                        stats.end_step()
                        break

                # ── 3. Regular tool calls ─────────────────────────────────
                tool_start = time.monotonic()
                observations = await self._execute_tools(action)
                tool_elapsed = time.monotonic() - tool_start
                for tc in action.tool_calls:
                    name = tc.get("name", tc.get("function", {}).get("name", ""))
                    stats.record_tool_call(name, tool_elapsed / max(len(action.tool_calls), 1))

                # Update trajectory with observations
                self.ctx.trajectory.steps[-1].observations = observations

                # Step callbacks
                for callback in self.ctx.step_callbacks:
                    await callback(step, action, observations)

                stats.end_step()

            except Exception as e:
                logger.error("Agent step %d failed: %s", step, e, exc_info=True)
                return AgentResult(
                    trajectory=self.ctx.trajectory,
                    messages=list(self.ctx.messages),
                    finish_reason="error",
                    error=str(e),
                )

        stats.finish()

        # Extract patch if in a code environment
        patch = ""
        try:
            workdir = self.ctx.task_info.get("workdir", "/testbed")
            commit = (
                self.ctx.task_info.get("pre_agent_commit_id")
                or self.ctx.task_info.get("base_commit")
            )
            patch = await self.ctx.session.get_patch(workdir, commit)
        except Exception as e:
            logger.warning("Failed to extract patch: %s", e)

        return AgentResult(
            trajectory=self.ctx.trajectory,
            patch=patch,
            messages=list(self.ctx.messages),
            finish_reason=finish_reason,
            metadata={"stats": stats.to_dict()},
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
        if action.tool_calls:
            observations = await self._execute_tools(action)
        return action, observations

    async def _execute_tools(self, action: Action) -> list[str]:
        """Execute tool calls and collect observations."""
        observations: list[str] = []

        # Determine if we're in XML mode (text-based tool calls)
        xml_mode = (
            self.ctx.tool_call_format is not None
            and not self.ctx.tool_call_format.needs_native_tools()
        )

        if xml_mode:
            # XML mode: assistant message is plain text, observations as user messages
            self.ctx.messages.append(Message(
                role="assistant",
                content=action.content,
            ))
        else:
            # OpenAI mode: assistant message with tool_calls structure
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

            if xml_mode:
                # XML mode: tool responses as user messages
                self.ctx.messages.append(Message(
                    role="user",
                    content=f"OBSERVATION:\n[{tool_name}]\n{obs}",
                ))
            else:
                # OpenAI mode: standard tool role messages
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
