"""CodeAct XML format — text-based tool calling via XML tags in LLM output.

Format::

    <function=tool_name>
    <parameter=param_name>value</parameter>
    <parameter=other_param>value</parameter>
    </function>

This format is used for models that do not support native function calling.
Tools are described in the system prompt and tool calls are parsed from
the response text using regex.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from awe_agent.core.llm.format.protocol import ToolCallFormat
from awe_agent.core.llm.types import LLMResponse, ToolCall

# Regex to match <function=name>...</function> blocks
_FUNCTION_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)

# Regex to match <parameter=name>value</parameter> within a function block
_PARAMETER_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)


class CodeActXMLFormat(ToolCallFormat):
    """XML-based tool call format for models without native function calling.

    Tools are described in the system prompt suffix. Tool calls are parsed
    from the LLM's text output using regex patterns.
    """

    def prepare_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        # Tools are described in the system prompt, not via API parameter
        return None

    def get_system_prompt_suffix(self, tools: list[dict[str, Any]]) -> str:
        """Generate XML tool descriptions for the system prompt."""
        if not tools:
            return ""

        lines = [
            "",
            "# Available Tools",
            "",
            "You have access to the following tools. To call a tool, use the XML format:",
            "",
            "<function=tool_name>",
            "<parameter=param_name>value</parameter>",
            "</function>",
            "",
            "Here are the available tools:",
            "",
        ]

        for tool_schema in tools:
            func = tool_schema.get("function", tool_schema)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            lines.append(f"## {name}")
            if desc:
                lines.append(f"Description: {desc}")

            properties = params.get("properties", {})
            required = set(params.get("required", []))
            if properties:
                lines.append("Parameters:")
                for pname, pinfo in properties.items():
                    ptype = pinfo.get("type", "string")
                    pdesc = pinfo.get("description", "")
                    req_marker = " (required)" if pname in required else " (optional)"
                    lines.append(f"  - {pname} ({ptype}{req_marker}): {pdesc}")

            lines.append("")

        return "\n".join(lines)

    def parse_response(self, response: LLMResponse) -> list[ToolCall]:
        """Parse XML tool calls from the response content."""
        if not response.content:
            return []

        tool_calls: list[ToolCall] = []

        for match in _FUNCTION_RE.finditer(response.content):
            func_name = match.group(1)
            func_body = match.group(2)

            # Extract parameters
            params: dict[str, str] = {}
            for param_match in _PARAMETER_RE.finditer(func_body):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                params[param_name] = param_value

            tool_calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=func_name,
                arguments=json.dumps(params),
            ))

        return tool_calls

    def needs_native_tools(self) -> bool:
        return False
