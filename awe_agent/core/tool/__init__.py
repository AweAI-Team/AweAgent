"""Tool framework for AweAgent.

Usage:
    from awe_agent.core.tool import Tool, tool_registry
    from awe_agent.core.tool.builtin.bash import BashTool

    bash = BashTool(timeout=120)
    result = await bash.execute({"command": "ls"}, session=session)
"""

from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.registry import tool_registry

__all__ = ["Tool", "tool_registry"]
