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
    if config.type == "terminus_2":
        from aweagent.core.condenser.terminus_2 import Terminus2Condenser

        return Terminus2Condenser(
            enable_summarize=config.enable_summarize,
            proactive_threshold=config.proactive_threshold,
            recovery_target_free_tokens=config.recovery_target_free_tokens,
            tokenizer_path=config.tokenizer_path,
        )
    raise ValueError(
        f"Unknown condenser type: {config.type!r}. "
        "Use 'none', 'truncation', 'tool_result_omission', or 'terminus_2'."
    )
