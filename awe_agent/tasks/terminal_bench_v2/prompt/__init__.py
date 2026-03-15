"""Terminal Bench 2.0 prompt — exactly matches Harbor/swalm.

Single message: template.format(instruction=..., terminal_state=...)
No system/user split. Only json_plain.txt.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_CACHE: str | None = None


def _get_template() -> str:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = (
            Path(__file__).parent / "json_plain.txt"
        ).read_text(encoding="utf-8")
    return _TEMPLATE_CACHE


def format_prompt(instruction: str, terminal_state: str = "") -> str:
    """Format the full prompt. Exactly matches swalm's template.format()."""
    return _get_template().format(
        instruction=instruction,
        terminal_state=terminal_state,
    )
