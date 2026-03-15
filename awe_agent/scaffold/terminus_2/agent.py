"""Terminus 2 Agent — tmux + JSON keystrokes for Terminal Bench 2.0.

Uses a custom loop: LLM outputs JSON with keystrokes -> send to tmux -> get output.
No tool-calling; compatible with Harbor/swalm Terminus 2 format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentResult
from awe_agent.core.agent.trajectory import Action, Trajectory
from awe_agent.core.llm.types import Message
from awe_agent.scaffold.terminus_2.parser import TerminusJSONParser
from awe_agent.scaffold.terminus_2.tmux_session import TmuxSessionAdapter

if TYPE_CHECKING:
    from awe_agent.core.config.schema import AweAgentConfig

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tasks"
    / "terminal_bench_v2"
    / "prompt"
    / "json_plain.txt"
)


class Terminus2Agent:
    """Terminus 2 agent: JSON keystrokes + tmux, no tool-calling.

    uses_custom_loop: TaskRunner / recipe calls run_async() instead of AgentLoop.
    """

    uses_custom_loop = True

    def __init__(
        self,
        parser_name: str = "json",
        session_name: str = "terminus-session",
        max_output_bytes: int = 10000,
    ) -> None:
        self._parser_name = parser_name
        self._session_name = session_name
        self._max_output_bytes = max_output_bytes
        self._parser = TerminusJSONParser()
        self._prompt_template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
        self._pending_completion = False

    @classmethod
    def from_config(cls, config: AweAgentConfig) -> Terminus2Agent:
        """Create from global config."""
        return cls()

    @classmethod
    def from_config_with_constraints(
        cls, config: AweAgentConfig, instance_constraints: Any
    ) -> Terminus2Agent:
        """Terminus 2 ignores search constraints."""
        return cls.from_config(config)

    def get_tools(self) -> list:
        """Terminus 2 uses no tools (JSON keystrokes only)."""
        return []

    def _limit_output(self, output: str) -> str:
        """Truncate output to max bytes, keep head and tail."""
        if len(output.encode("utf-8")) <= self._max_output_bytes:
            return output
        half = self._max_output_bytes // 2
        b = output.encode("utf-8")
        head = b[:half].decode("utf-8", errors="ignore")
        tail = b[-half:].decode("utf-8", errors="ignore")
        omitted = len(b) - half - half
        return f"{head}\n[... {omitted} bytes omitted ...]\n{tail}"

    def _get_confirmation_prompt(self, terminal_output: str) -> str:
        """Ask for double confirmation when task_complete. Align with Harbor."""
        return (
            f"Current terminal state:\n{terminal_output}\n\n"
            "Are you sure you want to mark the task as complete? "
            "This will trigger your solution to be graded and you won't be able to "
            'make any further corrections. If so, include "task_complete": true '
            "in your JSON response again."
        )

    async def run_async(self, prompt: str, context: AgentContext) -> AgentResult:
        """Run the Terminus 2 loop: LLM -> parse JSON -> tmux -> output."""
        workdir = context.task_info.get("workdir", "/workspace")
        max_steps = context.max_steps
        llm = context.llm

        tmux = TmuxSessionAdapter(
            session=context.session,
            session_name=self._session_name,
            workdir=workdir,
        )
        await tmux.start()

        initial_terminal_state = await tmux.get_incremental_output()
        instruction = context.task_info.get("instruction", "")
        initial_prompt = self._prompt_template.format(
            instruction=instruction,
            terminal_state=self._limit_output(initial_terminal_state),
        )
        history: list[dict[str, str]] = [{"role": "user", "content": initial_prompt}]
        trajectory = Trajectory()
        self._partial_trajectory = trajectory
        self._pending_completion = False
        finish_reason = "max_steps"

        for step in range(max_steps):
            logger.info("Terminus 2 step %d/%d", step + 1, max_steps)

            messages = [Message(role=m["role"], content=m["content"]) for m in history]
            response = await llm.chat(messages, tools=None)
            content = response.content or ""

            parse_result = self._parser.parse_response(content)

            trajectory.add_step(
                step=step,
                action=Action(type="message", content=content),
                observations=[],
            )

            if parse_result.error:
                err_msg = f"ERROR: {parse_result.error}"
                if parse_result.warning:
                    err_msg += f"\nWARNINGS: {parse_result.warning}"
                history.append({"role": "assistant", "content": content})
                history.append({
                    "role": "user",
                    "content": (
                        f"Previous response had parsing errors:\n{err_msg}\n\n"
                        "Please fix and provide valid JSON."
                    ),
                })
                continue

            commands = [
                (c.keystrokes, min(c.duration, 60.0)) for c in parse_result.commands
            ]

            for keystrokes, duration in commands:
                try:
                    await tmux.send_keys(
                        keystrokes,
                        block=False,
                        min_timeout_sec=max(duration, 0.1),
                    )
                except Exception as e:
                    logger.error("Send keys failed: %s", e)

            terminal_output = await tmux.get_incremental_output()
            terminal_output = self._limit_output(terminal_output)

            if parse_result.is_task_complete:
                if self._pending_completion:
                    logger.info("Task complete (confirmed)")
                    finish_reason = "finish"
                    break
                self._pending_completion = True
                history.append({"role": "assistant", "content": content})
                history.append({
                    "role": "user",
                    "content": self._get_confirmation_prompt(terminal_output),
                })
                continue

            self._pending_completion = False

            if parse_result.warning:
                next_prompt = (
                    f"Previous response had warnings:\n{parse_result.warning}"
                    f"\n\n{terminal_output}"
                )
            else:
                next_prompt = terminal_output

            history.append({"role": "assistant", "content": content})
            history.append({"role": "user", "content": next_prompt})

        return AgentResult(
            trajectory=trajectory,
            patch="",
            messages=[Message(role=m["role"], content=m["content"]) for m in history],
            finish_reason=finish_reason,
            metadata={},
        )
