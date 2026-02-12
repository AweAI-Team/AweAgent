"""Shared evaluation utilities — test ID parsing, result checking, script helpers.

Provides reusable building blocks for all evaluators:

- **Test ID parsing**: Handle the various formats datasets use to store test
  identifiers (JSON strings, lists, single IDs).
- **Pytest output parsing**: Extract structured counts from pytest summary lines.
- **Test result checking**: Verify FAIL_TO_PASS / PASS_TO_PASS outcomes.
- **Script generation**: Build pytest shell commands from test ID lists.
- **Test file protection**: Restore test files to HEAD after agent modifications.
- **Shared F2P/P2P evaluation**: Common test-running flow used by multiple
  evaluators.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession
    from awe_agent.core.task.types import EvalResult, Instance

logger = logging.getLogger(__name__)


# ── Test ID handling ────────────────────────────────────────────────────────


def parse_test_ids(raw: str | list[str] | None) -> list[str]:
    """Parse test identifiers from various dataset formats.

    Accepts:
        - ``'["test_a.py::test_one", "test_b.py::test_two"]'``  (JSON array)
        - ``["test_a", "test_b"]``  (Python list)
        - ``"test_a.py::test_one"``  (single test ID string)
        - ``""`` / ``None``  (empty — returns ``[]``)
    """
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
        if isinstance(parsed, str) and parsed:
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [raw]


def normalize_test_id(test_id: str) -> str:
    """Normalize a pytest node ID for fuzzy matching.

    ``test_foo.py::TestClass::test_method`` becomes
    ``test_foo.TestClass.test_method``.
    """
    return test_id.replace("::", ".").replace("/", ".").strip(".")


# ── Pytest output parsing ───────────────────────────────────────────────────


@dataclass
class PytestSummary:
    """Structured counts from a pytest summary line."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    warnings: int = 0
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0

    @property
    def total_run(self) -> int:
        """Tests that actually executed (excludes skipped / deselected)."""
        return self.passed + self.failed + self.errors

    @property
    def all_passed(self) -> bool:
        """True when at least one test ran and none failed or errored."""
        return self.failed == 0 and self.errors == 0 and self.passed > 0


_SUMMARY_RE = re.compile(r"=+\s*(.*?)\s+in\s+[\d.]+[sm]\s*=+")
_COUNT_RE = re.compile(r"(\d+)\s+(\w+)")
_LABEL_MAP: dict[str, str] = {
    "passed": "passed",
    "pass": "passed",
    "failed": "failed",
    "fail": "failed",
    "failure": "failed",
    "failures": "failed",
    "error": "errors",
    "errors": "errors",
    "skipped": "skipped",
    "skip": "skipped",
    "warning": "warnings",
    "warnings": "warnings",
    "xfailed": "xfailed",
    "xfail": "xfailed",
    "xpassed": "xpassed",
    "xpass": "xpassed",
    "deselected": "deselected",
}


def parse_pytest_summary(output: str) -> PytestSummary:
    """Parse the final pytest summary line into structured counts.

    Recognises lines like::

        ===== 5 passed, 2 failed in 3.45s =====
        ===== 1 passed in 0.01s =====
        ===== 3 passed, 1 warning in 0.50s =====
    """
    summary = PytestSummary()
    match = _SUMMARY_RE.search(output)
    if not match:
        return summary
    for m in _COUNT_RE.finditer(match.group(1)):
        label = m.group(2).lower()
        field_name = _LABEL_MAP.get(label)
        if field_name:
            setattr(summary, field_name, int(m.group(1)))
    return summary


# ── Test result verification ────────────────────────────────────────────────


def check_f2p_p2p(
    f2p_summary: PytestSummary,
    p2p_summary: PytestSummary,
    f2p_count: int,
    p2p_count: int,
) -> tuple[bool, dict[str, object]]:
    """Check FAIL_TO_PASS resolution and PASS_TO_PASS maintenance.

    Returns ``(accepted, details)`` where *accepted* is ``True`` iff every
    FAIL_TO_PASS test now passes and no PASS_TO_PASS test regressed.
    """
    f2p_resolved = f2p_summary.all_passed if f2p_count > 0 else True
    p2p_held = p2p_summary.all_passed if p2p_count > 0 else True
    accepted = f2p_resolved and p2p_held

    details: dict[str, object] = {
        "f2p_resolved": f2p_resolved,
        "p2p_held": p2p_held,
        "f2p": {
            "expected": f2p_count,
            "passed": f2p_summary.passed,
            "failed": f2p_summary.failed,
            "errors": f2p_summary.errors,
        },
        "p2p": {
            "expected": p2p_count,
            "passed": p2p_summary.passed,
            "failed": p2p_summary.failed,
            "errors": p2p_summary.errors,
        },
    }
    return accepted, details


# ── Script generation ───────────────────────────────────────────────────────


def build_pytest_command(
    test_ids: list[str],
    workdir: str,
    extra_args: str = "",
) -> str:
    """Build a shell command that runs specific pytest tests.

    Args:
        test_ids: Pytest node IDs to run.
        workdir: Project root inside the container.
        extra_args: Additional flags appended to the ``pytest`` invocation.
    """
    if not test_ids:
        return f"cd {workdir} && echo 'No tests specified'"
    tests = " ".join(shlex.quote(t) for t in test_ids)
    cmd = f"cd {workdir} && python -m pytest {tests} --tb=short --no-header -q"
    if extra_args:
        cmd += f" {extra_args}"
    return cmd


# ── Test file restoration ──────────────────────────────────────────────────


async def restore_test_files(session: RuntimeSession, workdir: str) -> None:
    """Restore test files to HEAD, preventing agent test tampering.

    Silently ignores errors when directories or patterns do not exist in
    the repository (common for projects with non-standard layouts).
    """
    await session.execute(
        f"cd {workdir} && "
        f"git checkout HEAD -- tests/ test/ Test/ Tests/ "
        f"2>/dev/null || true"
    )
    await session.execute(
        f"cd {workdir} && "
        f"git checkout HEAD -- "
        f"$(git ls-files 'test_*.py' '*_test.py' 'conftest.py' 2>/dev/null) "
        f"2>/dev/null || true"
    )


# ── Shared F2P / P2P evaluation flow ───────────────────────────────────────


async def run_f2p_p2p_eval(
    session: RuntimeSession,
    instance: Instance,
    timeout: int = 3600,
) -> EvalResult:
    """Run FAIL_TO_PASS and PASS_TO_PASS tests and return an ``EvalResult``.

    This is the standard evaluation flow shared by SWE-Bench, ScaleSWE,
    BeyondSWE (func-level tasks), and any dataset that uses the F2P / P2P
    test-ID convention.

    Steps:
        1. Parse F2P and P2P test IDs from ``instance.metadata``.
        2. Run F2P tests — all must now pass (bug fixed).
        3. Run P2P tests — none should regress (no new breakage).
        4. Return ``EvalResult(accepted=f2p_ok AND p2p_ok)``.
    """
    from awe_agent.core.task.types import EvalResult  # deferred to avoid cycles

    f2p_ids = parse_test_ids(instance.metadata.get("FAIL_TO_PASS"))
    p2p_ids = parse_test_ids(instance.metadata.get("PASS_TO_PASS"))

    if not f2p_ids and not p2p_ids:
        logger.warning("Instance %s has no F2P/P2P test IDs — skipping", instance.id)
        return EvalResult(
            accepted=False,
            score=0.0,
            details={"error": "no_test_ids"},
        )

    # ── Run F2P tests ──────────────────────────────────────────────────
    f2p_summary = PytestSummary()
    f2p_output = ""
    if f2p_ids:
        cmd = build_pytest_command(f2p_ids, instance.workdir)
        result = await session.execute(cmd, timeout=timeout)
        f2p_output = result.output
        f2p_summary = parse_pytest_summary(f2p_output)

    # ── Run P2P tests ──────────────────────────────────────────────────
    p2p_summary = PytestSummary()
    p2p_output = ""
    if p2p_ids:
        cmd = build_pytest_command(p2p_ids, instance.workdir)
        result = await session.execute(cmd, timeout=timeout)
        p2p_output = result.output
        p2p_summary = parse_pytest_summary(p2p_output)

    # ── Check results ──────────────────────────────────────────────────
    accepted, details = check_f2p_p2p(
        f2p_summary, p2p_summary, len(f2p_ids), len(p2p_ids),
    )
    details["f2p_output"] = f2p_output[-2000:]
    details["p2p_output"] = p2p_output[-2000:]

    return EvalResult(
        accepted=accepted,
        score=1.0 if accepted else 0.0,
        details=details,
    )
