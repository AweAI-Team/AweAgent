"""Context condensation strategies for managing long conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from awe_agent.core.condenser.protocol import Condenser
from awe_agent.core.condenser.truncation import TruncationCondenser

if TYPE_CHECKING:
    from awe_agent.core.config.schema import CondenserConfig

__all__ = ["Condenser", "TruncationCondenser", "build_condenser"]


def build_condenser(config: CondenserConfig) -> Condenser | None:
    """Build a condenser from config. Returns None if type is 'none'."""
    if config.type == "none":
        return None
    if config.type == "truncation":
        return TruncationCondenser(
            max_messages=config.max_messages,
            keep_first=config.keep_first,
        )
    raise ValueError(f"Unknown condenser type: {config.type!r}. Use 'none' or 'truncation'.")
