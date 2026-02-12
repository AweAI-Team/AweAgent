"""Think tool — allows agent to reason without taking external action."""

from __future__ import annotations

from typing import Any

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool


class ThinkTool(Tool):
    """A tool for the agent to think/reason without executing any action.

    The agent can use this to organize thoughts, plan next steps, or
    reason about a problem before taking action.
    """

    @property
    def name(self) -> str:
        return "think"

    @property
    def description(self) -> str:
        return (
            "Use this tool to think about the problem, plan your approach, "
            "or reason through complex decisions. No external action is taken."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning or thought process.",
                },
            },
            "required": ["thought"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        # Think tool doesn't do anything — just returns acknowledgment
        return "Thought recorded. Continue with your next action."
