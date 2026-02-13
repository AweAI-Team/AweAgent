"""Agent registry — global registry for agent scaffold discovery."""

from awe_agent.plugins.registry import Registry

# Global agent registry. Agents register here and are discovered via entry_points.
agent_registry: Registry[type] = Registry("awe_agent.agent")
