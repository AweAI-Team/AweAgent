"""Configuration system for AweAgent."""

from aweagent.core.config.loader import load_config
from aweagent.core.config.schema import AweAgentConfig

__all__ = ["AweAgentConfig", "load_config"]
