"""Agent registry — global registry for agent scaffold discovery."""

from awe_agent.plugins.registry import Registry
from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

# Global agent registry. Agents register here and are discovered via entry_points.
agent_registry: Registry[type] = Registry("awe_agent.agent")

# Built-in agents (always available, even without pip install -e .)
agent_registry.register("search_swe", SearchSWEAgent)

# Lazy-register terminus_2
try:
    from awe_agent.scaffold.terminus_2.agent import Terminus2Agent
    agent_registry.register("terminus_2", Terminus2Agent)
except ImportError:
    pass
