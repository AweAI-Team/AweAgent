"""OpenAI function calling format — passthrough for native tool calling."""

from __future__ import annotations

from typing import Any

from awe_agent.core.llm.format.protocol import ToolCallFormat
from awe_agent.core.llm.types import LLMResponse, ToolCall


class OpenAIFunctionFormat(ToolCallFormat):
    """Passthrough format for native OpenAI function calling.

    Tool schemas are sent via the ``tools`` API parameter.
    Tool calls are extracted from ``response.tool_calls``.
    """

    def prepare_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        return tools if tools else None

    def get_system_prompt_suffix(self, tools: list[dict[str, Any]]) -> str:
        return ""

    def parse_response(self, response: LLMResponse) -> list[ToolCall]:
        return response.tool_calls

    def needs_native_tools(self) -> bool:
        return True
