"""CodeAct XML format — text-based tool calling via XML tags in LLM output.

Format::

    <function=tool_name>
    <parameter=param_name>value</parameter>
    <parameter=other_param>
    This is the value for the second parameter
    that can span
    multiple lines
    </parameter>
    </function>

This format is used for models that do not support native function calling.
Tools are described in the system prompt and tool calls are parsed from
the response text using regex.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from awe_agent.core.llm.format.protocol import ToolCallFormat
from awe_agent.core.llm.types import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────

# Match <function=name>body</function>
_FUNCTION_RE = re.compile(
    r"<function=([^>]+)>\n?(.*?)</function>",
    re.DOTALL,
)

# Match <parameter=name>value</parameter> within a function block
_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)


def _convert_tools_to_description(tools: list[dict[str, Any]]) -> str:
    """Generate tool descriptions in ``---- BEGIN/END FUNCTION ----`` format."""
    parts: list[str] = []
    for i, tool_schema in enumerate(tools):
        func = tool_schema.get("function", tool_schema)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        if i > 0:
            parts.append("")
        parts.append(f"---- BEGIN FUNCTION #{i + 1}: {name} ----")
        parts.append(f"**Description**: {desc}")

        properties = params.get("properties", {})
        if properties:
            required_params = set(params.get("required", []))
            parts.append("**Parameters**:")
            for j, (pname, pinfo) in enumerate(properties.items()):
                ptype = pinfo.get("type", "string")
                pstatus = "required" if pname in required_params else "optional"
                pdesc = pinfo.get("description", "No description provided")

                if "enum" in pinfo:
                    enum_values = ", ".join(f"`{v}`" for v in pinfo["enum"])
                    pdesc += f"\nAllowed values: [{enum_values}]"

                parts.append(f"  ({j + 1}) {pname} ({ptype}, {pstatus}): {pdesc}")
        else:
            parts.append("No parameters are required for this function.")

        parts.append(f"---- END FUNCTION #{i + 1} ----")

    return "\n".join(parts)


# ── System prompt suffix ──────────────────────────────────────────────

_SYSTEM_PROMPT_SUFFIX = """\
You have access to the following functions:

{description}

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""


class CodeActXMLFormat(ToolCallFormat):
    """XML-based tool call format for models without native function calling.

    Prompt format uses the OpenHands CodeAct convention.  Parsing extracts
    one tool call per turn from the LLM's text output.
    """

    def prepare_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        # Tools are described in the system prompt, not via API parameter
        return None

    def get_system_prompt_suffix(self, tools: list[dict[str, Any]]) -> str:
        """Generate system prompt suffix with tool descriptions.

        Uses ``---- BEGIN/END FUNCTION ----`` format with ``<IMPORTANT>``
        reminder block.
        """
        if not tools:
            return ""

        description = _convert_tools_to_description(tools)
        return "\n" + _SYSTEM_PROMPT_SUFFIX.format(description=description)

    def parse_response(self, response: LLMResponse) -> list[ToolCall]:
        """Parse a single XML tool call from the response content.

        Only the **first** ``<function=...>...</function>`` block is parsed
        (one tool call per turn, matching the CodeAct convention).  Malformed parameter
        tags are detected and logged as warnings.
        """
        if not response.content:
            return []

        content = _fix_incomplete_tag(response.content)

        fn_match = _FUNCTION_RE.search(content)
        if not fn_match:
            return []

        func_name = fn_match.group(1).strip()
        func_body = fn_match.group(2)

        # Validate parameter tag pairing
        open_count = func_body.count("<parameter=")
        close_count = func_body.count("</parameter>")
        if open_count != close_count:
            logger.warning(
                "Mismatched parameter tags in function '%s': "
                "%d opening vs %d closing tags",
                func_name,
                open_count,
                close_count,
            )

        # Extract parameters
        params: dict[str, str] = {}
        for param_match in _PARAMETER_RE.finditer(func_body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            params[param_name] = param_value

        return [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=func_name,
                arguments=json.dumps(params),
            )
        ]

    def needs_native_tools(self) -> bool:
        return False


def _fix_incomplete_tag(content: str) -> str:
    """Fix incomplete ``</function>`` tag when LLM output is cut off.

    If there is exactly one ``<function=`` and no closing tag, append
    the missing portion so the regex can still match.
    """
    if "<function=" in content and content.count("<function=") == 1:
        if "</function>" not in content:
            stripped = content.rstrip()
            if stripped.endswith("</"):
                # Partially written closing tag — complete it
                content = stripped + "function>"
            else:
                content = stripped + "\n</function>"
    return content
