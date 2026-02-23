"""ThinkTool — allows agent to reason without taking external action."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool


class ThinkTool(Tool):
    """A tool for the agent to think/reason without executing any action.

    Records all thoughts in a history list for later retrieval.
    Useful for structured reasoning, brainstorming, and planning.
    """

    def __init__(self) -> None:
        self.think_history: list[str] = []

    @property
    def name(self) -> str:
        return "think"

    @property
    def description(self) -> str:
        return dedent("""\
            Use the tool to think about something. It will not obtain new \
information or make any changes to the repository, but just log the thought. \
Use it when complex reasoning or brainstorming is needed.

            Common use cases:
            1. When exploring a repository and discovering the source of a bug, \
call this tool to brainstorm several unique ways of fixing the bug, and assess \
which change(s) are likely to be simplest and most effective.
            2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
            3. When planning a complex refactoring, use this tool to outline different \
approaches and their tradeoffs.
            4. When designing a new feature, use this tool to think through architecture \
decisions and implementation details.
            5. When debugging a complex issue, use this tool to organize your thoughts \
and hypotheses.

            The tool simply logs your thought process for better transparency and does \
not execute any code or make changes.""")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of your thought.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        self.think_history.append(params.get("content", ""))
        return "Your thought has been recorded. Please continue your work."
