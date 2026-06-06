"""Context condensation strategies for managing long conversations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aweagent.core.condenser.protocol import Condenser
from aweagent.core.condenser.tool_result_omission import ToolResultOmissionCondenser
from aweagent.core.condenser.truncation import TruncationCondenser

if TYPE_CHECKING:
    from aweagent.core.config.schema import CondenserConfig

__all__ = [
    "Condenser",
    "ToolResultOmissionCondenser",
    "TruncationCondenser",
    "build_condenser",
]


def build_condenser(config: CondenserConfig) -> Condenser | None:
    """Build a condenser from config. Returns None if type is 'none'."""
    if config.type == "none":
        return None
    if config.type == "truncation":
        return TruncationCondenser(
            max_messages=config.max_messages,
            keep_first=config.keep_first,
        )
    if config.type == "tool_result_omission":
        return ToolResultOmissionCondenser(
            keep_recent_tool_results=config.keep_recent_tool_results,
        )
    raise ValueError(
        f"Unknown condenser type: {config.type!r}. "
        "Use 'none', 'truncation', or 'tool_result_omission'."
    )
