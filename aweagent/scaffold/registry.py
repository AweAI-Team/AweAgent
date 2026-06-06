"""Agent registry — global registry for agent scaffold discovery."""

from aweagent.plugins.registry import Registry
from aweagent.scaffold.deepsearch.agent import DeepSearchAgent
from aweagent.scaffold.iter_research.agent import IterResearchAgent
from aweagent.scaffold.search_swe.agent import SearchSWEAgent

# Global agent registry. Agents register here and are discovered via entry_points.
agent_registry: Registry[type] = Registry("aweagent.agent")

# Built-in agents (always available, even without pip install -e .)
agent_registry.register("search_swe", SearchSWEAgent)
# DeepSearch is registered as a scaffold so configs can select agent.type=deepsearch.
agent_registry.register("deepsearch", DeepSearchAgent)
# IterResearch: long-horizon research agent with Markovian context reconstruction.
agent_registry.register("iter_research", IterResearchAgent)

# Lazy-register terminus_2
try:
    from aweagent.scaffold.terminus_2.agent import Terminus2Agent
    agent_registry.register("terminus_2", Terminus2Agent)
except ImportError:
    pass
