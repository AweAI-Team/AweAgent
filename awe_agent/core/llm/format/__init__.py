"""Tool call format registry and factory."""

from __future__ import annotations

from awe_agent.core.llm.format.openai import OpenAIFunctionFormat
from awe_agent.core.llm.format.protocol import ToolCallFormat
from awe_agent.core.llm.format.terminus_json import TerminusJSONFormat
from awe_agent.core.llm.format.xml import CodeActXMLFormat

FORMATS: dict[str, type[ToolCallFormat]] = {
    "openai_function": OpenAIFunctionFormat,
    "codeact_xml": CodeActXMLFormat,
    "terminus_json": TerminusJSONFormat,
}

__all__ = [
    "CodeActXMLFormat",
    "FORMATS",
    "OpenAIFunctionFormat",
    "TerminusJSONFormat",
    "ToolCallFormat",
    "get_tool_format",
]


def get_tool_format(name: str) -> ToolCallFormat:
    """Instantiate a tool call format by name.

    Args:
        name: One of ``"openai_function"``, ``"codeact_xml"``,
              or ``"terminus_json"``.

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
