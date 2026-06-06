"""Tool framework for AweAgent.

Usage:
    from aweagent.core.tool import Tool, tool_registry
    from aweagent.core.tool.code import ExecuteBashTool, StrReplaceEditorTool, ThinkTool

    bash = ExecuteBashTool(timeout=180)
    result = await bash.execute({"command": "ls"}, session=session)
"""

from aweagent.core.tool.protocol import Tool
from aweagent.core.tool.registry import tool_registry

__all__ = ["Tool", "tool_registry"]
