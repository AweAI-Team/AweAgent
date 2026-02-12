"""Tool Protocol — the interface all tools must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession


class Tool(ABC):
    """Base class for all tools.

    Tools provide actions agents can take (bash, file editing, search, etc.)
    Each tool exposes an OpenAI-compatible function calling schema.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @property
    def schema(self) -> dict[str, Any]:
        """Full OpenAI function calling schema. Override for custom schemas."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        """Execute the tool and return observation string.

        Args:
            params: Parsed parameters matching self.parameters schema.
            session: Optional runtime session for tools that need container access.

        Returns:
            Observation string to be fed back to the agent.
        """
        ...
