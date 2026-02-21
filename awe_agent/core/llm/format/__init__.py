"""Tool call format registry and factory."""

from __future__ import annotations

from awe_agent.core.llm.format.openai import OpenAIFunctionFormat
from awe_agent.core.llm.format.protocol import ToolCallFormat
from awe_agent.core.llm.format.xml import CodeActXMLFormat

FORMATS: dict[str, type[ToolCallFormat]] = {
    "openai_function": OpenAIFunctionFormat,
    "codeact_xml": CodeActXMLFormat,
}

__all__ = [
    "CodeActXMLFormat",
    "FORMATS",
    "OpenAIFunctionFormat",
    "ToolCallFormat",
    "get_format",
]


def get_format(name: str) -> ToolCallFormat:
    """Instantiate a tool call format by name.

    Args:
        name: Format name. One of ``"openai_function"`` or ``"codeact_xml"``.

    Raises:
        ValueError: If the format name is unknown.
    """
    cls = FORMATS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown tool_call_format: {name!r}. "
            f"Available: {', '.join(FORMATS)}"
        )
    return cls()
