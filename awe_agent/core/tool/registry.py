"""Tool registry — global registry for tool discovery."""

from awe_agent.plugins.registry import Registry

# Global tool registry. Tools register here and are discovered via entry_points.
tool_registry: Registry[type] = Registry("awe_agent.tool")
