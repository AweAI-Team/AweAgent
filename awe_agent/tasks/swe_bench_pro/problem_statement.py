"""Problem statement formatting helpers for SWE-bench-Pro."""

from __future__ import annotations

from typing import Any

_REQUIREMENTS_HEADER = "Requirements:"
_INTERFACE_HEADER = "New interfaces introduced:"


def build_problem_statement(raw: dict[str, Any]) -> str:
    """Match the official SWE-bench-Pro issue formatting."""
    problem_statement = _coerce_text(raw.get("problem_statement", ""))
    requirements = _coerce_text(raw.get("requirements", ""))
    interface = _coerce_text(raw.get("interface", ""))

    if _looks_preformatted(problem_statement):
        return problem_statement

    return format_problem_statement(problem_statement, requirements, interface)


def format_problem_statement(
    problem_statement: Any,
    requirements: Any = "",
    interface: Any = "",
) -> str:
    """Render the official issue + requirements + interface block."""
    return (
        f"{_coerce_text(problem_statement)}\n\n"
        f"{_REQUIREMENTS_HEADER}\n"
        f"{_coerce_text(requirements)}\n\n"
        f"{_INTERFACE_HEADER}\n"
        f"{_coerce_text(interface)}"
    )


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_preformatted(problem_statement: str) -> bool:
    return (
        _REQUIREMENTS_HEADER in problem_statement
        and _INTERFACE_HEADER in problem_statement
    )
