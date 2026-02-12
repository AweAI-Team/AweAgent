"""Configuration system for AweAgent."""

from awe_agent.core.config.loader import load_config
from awe_agent.core.config.schema import AweAgentConfig

__all__ = ["AweAgentConfig", "load_config"]
