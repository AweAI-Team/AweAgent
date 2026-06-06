"""Tool-result omission condenser for search-heavy agents."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aweagent.core.condenser.protocol import Condenser
from aweagent.core.llm.types import Message

OMITTED_TOOL_RESULT = "Tool result is omitted to save tokens."


class ToolResultOmissionCondenser(Condenser):
    """Keep all turns, but omit older tool-result contents.

    The condenser preserves assistant messages and tool-call structure so native
    tool-calling histories remain valid, while replacing old observation bodies
    with a short placeholder.
    """

    def __init__(self, keep_recent_tool_results: int = 5) -> None:
        self._keep_recent_tool_results = keep_recent_tool_results

    async def condense(self, messages: list[Message]) -> list[Message]:
        if self._keep_recent_tool_results == -1:
            return list(messages)

        omitted_indices = self._indices_to_omit(messages)
        if not omitted_indices:
            return list(messages)

        return [
            self._omit_message(message) if i in omitted_indices else message
            for i, message in enumerate(messages)
        ]

    def _indices_to_omit(self, messages: list[Message]) -> set[int]:
        turns = self._tool_result_turns(messages)
        keep = max(self._keep_recent_tool_results, 0)
        kept_indices = {
            index
            for turn in (turns[-keep:] if keep else [])
            for index in turn
        }
        return {index for turn in turns for index in turn if index not in kept_indices}

    def _tool_result_turns(self, messages: list[Message]) -> list[list[int]]:
        turns: list[list[int]] = []
        current_turn: list[int] | None = None
        seen_initial_user = False

        for index, message in enumerate(messages):
            if message.role == "assistant":
                current_turn = []
                continue

            if message.role == "tool":
                if current_turn is None:
                    current_turn = []
                current_turn.append(index)
                if current_turn not in turns:
                    turns.append(current_turn)
                continue

            if message.role == "user":
                if not seen_initial_user:
                    seen_initial_user = True
                    continue
                if _is_text_observation(message.content):
                    if current_turn is None:
                        current_turn = []
                    current_turn.append(index)
                    if current_turn not in turns:
                        turns.append(current_turn)

        return [turn for turn in turns if turn]

    @staticmethod
    def _omit_message(message: Message) -> Message:
        return replace(message, content=_omitted_content(message.content))


def _is_text_observation(content: Any) -> bool:
    if isinstance(content, str):
        return content.startswith("OBSERVATION:")
    return False


def _omitted_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, list):
        return [{"type": "text", "text": OMITTED_TOOL_RESULT}]
    return OMITTED_TOOL_RESULT
