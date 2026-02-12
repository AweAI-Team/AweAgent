"""Context condensation strategies for managing long conversations."""

from awe_agent.core.condenser.protocol import Condenser
from awe_agent.core.condenser.truncation import TruncationCondenser

__all__ = ["Condenser", "TruncationCondenser"]
