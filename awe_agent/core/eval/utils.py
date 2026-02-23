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
import xml.etree.ElementTree as ET
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
        f"$(git ls-files '**/test_*.py' '**/*_test.py' '**/conftest.py' 2>/dev/null) "
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


# ── Pytest Runner Script (injected into container) ─────────────────────

PYTEST_RUNNER_SCRIPT = '''\
import json, sys, os
import pytest

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    test_ids = config["test_ids"]
    xml_path = config.get("xml_path", "/tmp/_awe_test_results.xml")
    sys.path.insert(0, os.getcwd())
    sys.argv = ["pytest"]
    args = ["-vv", f"--junitxml={xml_path}", "-o", "addopts=", "--rootdir=."] + test_ids
    ret = pytest.main(args)
    print("<pytest>true</pytest>" if ret == 0 else "<pytest>false</pytest>")
'''


# ── JUnit XML parsing ──────────────────────────────────────────────────


def _normalize_for_match(s: str) -> str:
    """Normalize a test ID for fuzzy matching: remove .py, replace / and :: with dots."""
    return s.replace(".py", "").replace("/", ".").replace("::", ".").strip(".")


def _fingerprint(s: str) -> str:
    """Remove all whitespace for fingerprint matching."""
    return re.sub(r"\s+", "", s)


def parse_junit_xml(
    xml_content: str,
    expected_tests: list[str],
) -> tuple[bool, dict[str, object]]:
    """Parse JUnit XML and match test results against expected test IDs.

    Uses 4 matching strategies:
    1. Exact match: ``file_attr::name`` vs known tests
    2. Normalized match: ``classname.name`` → dots, no ``.py``
    3. Fingerprint match: remove all whitespace
    4. Fallback: ``classname → file_path`` + ``::name``

    Skipped tests are ignored.  Returns ``(all_passed, details)``.
    """
    details: dict[str, object] = {
        "matched": {},
        "unmatched_expected": list(expected_tests),
        "xml_errors": [],
    }

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        details["xml_errors"] = [str(e)]
        return False, details

    # Build lookup sets for expected tests
    exact_set = set(expected_tests)
    norm_map = {_normalize_for_match(t): t for t in expected_tests}
    fp_map = {_fingerprint(_normalize_for_match(t)): t for t in expected_tests}

    matched: dict[str, str] = {}  # expected_id → status
    found_expected = set()

    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        classname = tc.get("classname", "")
        file_attr = tc.get("file", "")

        # Skip skipped tests
        if tc.find("skipped") is not None:
            continue

        # Determine status
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        else:
            status = "passed"

        # Strategy 1: exact match with file::name
        candidate1 = f"{file_attr}::{name}" if file_attr else ""
        if candidate1 in exact_set:
            matched[candidate1] = status
            found_expected.add(candidate1)
            continue

        # Strategy 2: normalized classname.name
        candidate2 = _normalize_for_match(f"{classname}.{name}")
        if candidate2 in norm_map:
            orig = norm_map[candidate2]
            matched[orig] = status
            found_expected.add(orig)
            continue

        # Strategy 3: fingerprint
        candidate3 = _fingerprint(candidate2)
        if candidate3 in fp_map:
            orig = fp_map[candidate3]
            matched[orig] = status
            found_expected.add(orig)
            continue

        # Strategy 4: classname → file path + ::name
        fallback_file = classname.replace(".", "/") + ".py"
        candidate4 = f"{fallback_file}::{name}"
        if candidate4 in exact_set:
            matched[candidate4] = status
            found_expected.add(candidate4)
            continue

    unmatched = [t for t in expected_tests if t not in found_expected]
    all_passed = (
        len(found_expected) > 0
        and all(v == "passed" for v in matched.values())
        and len(unmatched) == 0
    )

    details["matched"] = matched
    details["unmatched_expected"] = unmatched
    details["total_matched"] = len(matched)
    details["total_expected"] = len(expected_tests)

    return all_passed, details


# ── Test runner with injected script ───────────────────────────────────


async def run_tests_with_runner(
    session: RuntimeSession,
    workdir: str,
    test_ids: list[str],
    timeout: int = 3600,
) -> tuple[bool, str, dict[str, object]]:
    """Run tests using the injected pytest runner script.

    Steps:
    1. Upload ``PYTEST_RUNNER_SCRIPT`` → ``/tmp/_awe_pytest_runner.py``
    2. Upload test config JSON → ``/tmp/_awe_test_config.json``
    3. Execute the runner
    4. Check ``<pytest>true</pytest>`` marker (fast path)
    5. Download JUnit XML → ``parse_junit_xml()`` (detailed path)
    6. If XML unavailable → fallback to ``parse_pytest_summary()``

    Returns ``(all_passed, raw_output, details)``.
    """
    if not test_ids:
        return False, "", {"error": "no_test_ids"}

    # 1. Upload runner script
    await session.upload_file(
        "/tmp/_awe_pytest_runner.py", PYTEST_RUNNER_SCRIPT.encode(),
    )

    # 2. Upload test config
    config_data = json.dumps({"test_ids": test_ids, "xml_path": "/tmp/_awe_test_results.xml"})
    await session.upload_file(
        "/tmp/_awe_test_config.json", config_data.encode(),
    )

    # 3. Execute runner
    result = await session.execute(
        f"cd {workdir} && python /tmp/_awe_pytest_runner.py /tmp/_awe_test_config.json",
        timeout=timeout,
    )
    raw_output = result.output

    # 4. Fast path: check <pytest>true</pytest> marker
    if "<pytest>true</pytest>" in raw_output:
        return True, raw_output, {"marker": "pytest_true", "exit_code": result.exit_code}

    # 5. Try JUnit XML for detailed results
    try:
        xml_bytes = await session.download_file("/tmp/_awe_test_results.xml")
        xml_content = xml_bytes.decode("utf-8", errors="replace")
        all_passed, xml_details = parse_junit_xml(xml_content, test_ids)
        xml_details["exit_code"] = result.exit_code
        xml_details["source"] = "junit_xml"
        return all_passed, raw_output, xml_details
    except (FileNotFoundError, Exception) as exc:
        logger.warning("JUnit XML not available, falling back to summary: %s", exc)

    # 6. Fallback to pytest summary parsing
    summary = parse_pytest_summary(raw_output)
    all_passed = summary.all_passed
    details: dict[str, object] = {
        "source": "pytest_summary",
        "passed": summary.passed,
        "failed": summary.failed,
        "errors": summary.errors,
        "exit_code": result.exit_code,
    }
    return all_passed, raw_output, details


# ── Repo-level test output parsing ─────────────────────────────────────


def parse_pytest_output(output: str, pytest_num: int) -> bool:
    """Check repo-level test output: passed >= pytest_num and no failures.

    Used by ``_eval_repo_level`` for doc2repo tasks where the expected
    number of passing tests is known (``test_suite_num``).
    """
    summary = parse_pytest_summary(output)
    if summary.passed >= pytest_num and summary.failed == 0 and summary.errors == 0:
        return True
    return False
