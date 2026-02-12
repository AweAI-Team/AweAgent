"""Truncation-based condenser — keeps first N and most recent messages."""

from __future__ import annotations

from awe_agent.core.condenser.protocol import Condenser
from awe_agent.core.llm.types import Message


class TruncationCondenser(Condenser):
    """Keep system prompt + first few messages + most recent messages.

    Simple but effective for most use cases.
    """

    def __init__(
        self,
        max_messages: int = 50,
        keep_first: int = 2,
    ) -> None:
        self._max_messages = max_messages
        self._keep_first = keep_first

    async def condense(self, messages: list[Message]) -> list[Message]:
        if len(messages) <= self._max_messages:
            return messages

        # Keep first N messages (system prompt + initial task)
        head = messages[:self._keep_first]
        # Keep most recent messages to fill remaining budget
        tail_budget = self._max_messages - self._keep_first
        tail = messages[-tail_budget:]

        return head + tail
