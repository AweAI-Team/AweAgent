"""Condenser Protocol — interface for conversation compression strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from awe_agent.core.llm.types import Message


class Condenser(ABC):
    """Abstract base for context compression.

    When conversations grow too long, condensers reduce them while
    preserving important information.
    """

    @abstractmethod
    async def condense(self, messages: list[Message]) -> list[Message]:
        """Compress a conversation. Returns a potentially shorter message list."""
        ...
