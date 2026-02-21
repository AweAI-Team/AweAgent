"""ToolCallFormat protocol — interface for different tool call encoding formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from awe_agent.core.llm.types import LLMResponse, ToolCall


class ToolCallFormat(ABC):
    """Abstract base for tool call format encoding/decoding.

    Supports both native OpenAI function calling and text-based
    formats like CodeAct XML.
    """

    @abstractmethod
    def prepare_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        """Prepare tool schemas for the API call.

        Returns tool schemas for native function calling, or None if tools
        are described in the system prompt instead.
        """
        ...

    @abstractmethod
    def get_system_prompt_suffix(self, tools: list[dict[str, Any]]) -> str:
        """Get additional system prompt content describing available tools.

        Returns empty string for native function calling, or tool descriptions
        for text-based formats.
        """
        ...

    @abstractmethod
    def parse_response(self, response: LLMResponse) -> list[ToolCall]:
        """Extract tool calls from the LLM response.

        For native function calling, returns response.tool_calls directly.
        For text-based formats, parses tool calls from response.content.
        """
        ...

    @abstractmethod
    def needs_native_tools(self) -> bool:
        """Whether this format uses native API tool/function calling."""
        ...
