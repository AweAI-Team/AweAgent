"""Tool registry — global registry for tool discovery."""

from aweagent.plugins.registry import Registry

# Global tool registry. Tools register here and are discovered via entry_points.
tool_registry: Registry[type] = Registry("aweagent.tool")
