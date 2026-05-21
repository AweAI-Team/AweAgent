#!/usr/bin/env python3
"""Patch swebench==4.1.0 log parsing for narrow log-format mismatches.

This is an evaluation-side compatibility patch. It does not change predictions.
It does not change Docker, images, test specs, eval scripts, or model patches.
It handles three narrow parser cases:

1. pytest produced only dot output plus a summary such as "1 passed", so the
   stock SWE-bench parser returned an empty status map even though the one
   expected test passed.
2. pytest reported every parametrized case as passed, but the SWE-bench
   expected test name used an empty parametrization suffix, e.g. "test_x[]",
   while pytest emitted concrete ids such as "test_x[unit0]".
3. Sympy's custom runner printed an expected test name before the harness'
   "End Test Output" marker and the trailing "ok" after it. The fallback only
   treats this as passed when the full raw log contains a Sympy all-passed
   summary and the test is in FAIL_TO_PASS/PASS_TO_PASS.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


MARKER_SINGLE = "# AWEAGENT_SINGLE_TEST_PYTEST_DOT_FALLBACK"
MARKER_EMPTY_PARAM = "# AWEAGENT_PYTEST_EMPTY_PARAM_ID_FALLBACK"
MARKER_SYMPY_INTERRUPTED_OK = "# AWEAGENT_SYMPY_INTERRUPTED_OK_FALLBACK"
MARKER_FULL_LOG_SUPPLEMENT = "# AWEAGENT_EXPECTED_TEST_FULL_LOG_SUPPLEMENT"


def single_test_fallback() -> str:
    return f"""\
    if not test_status_map:
        # {MARKER_SINGLE}
        expected_tests = []
        expected_tests.extend(getattr(test_spec, "FAIL_TO_PASS", []) or [])
        expected_tests.extend(getattr(test_spec, "PASS_TO_PASS", []) or [])
        clean_log = re.sub(r"\\[(\\d+)m", "", log).translate(
            str.maketrans("", "", escapes)
        )
        summary_match = re.search(r"(?m)(?:^|=+\\s+)(\\d+) passed(?:[,\\s=]|$)", clean_log)
        if summary_match and len(expected_tests) == int(summary_match.group(1)) == 1:
            test_status_map[expected_tests[0]] = TestStatus.PASSED.value

"""


def empty_param_fallback() -> str:
    return f"""\
    # {MARKER_EMPTY_PARAM}
    expected_tests = []
    expected_tests.extend(getattr(test_spec, "FAIL_TO_PASS", []) or [])
    expected_tests.extend(getattr(test_spec, "PASS_TO_PASS", []) or [])
    clean_log = re.sub(r"\\[(\\d+)m", "", log).translate(
        str.maketrans("", "", escapes)
    )
    all_passed_summary = re.search(
        r"(?m)(?:^|=+\\s+)\\d+ passed(?:[,=\\s]|$)", clean_log
    )
    no_failure_summary = not re.search(
        r"(?m)^=+.*\\b(?:failed|error|errors)\\b.*=+$",
        clean_log,
        re.IGNORECASE,
    )
    if all_passed_summary and no_failure_summary:
        for expected_test in expected_tests:
            if expected_test in test_status_map or not expected_test.endswith("[]"):
                continue
            prefix = expected_test[:-2] + "["
            matching_statuses = [
                status
                for test_name, status in test_status_map.items()
                if test_name.startswith(prefix) and test_name.endswith("]")
            ]
            if matching_statuses and all(
                status == TestStatus.PASSED.value for status in matching_statuses
            ):
                test_status_map[expected_test] = TestStatus.PASSED.value

"""


def patch_pytest_parser(path: Path) -> bool:
    text = path.read_text()
    if MARKER_SINGLE in text and "clean_log = re.sub" not in text:
        old = '        summary_match = re.search(r"(?m)(?:^|=+\\\\s+)(\\\\d+) passed(?:[,\\\\s=]|$)", log)\n'
        new = """\
        clean_log = re.sub(r"\\[(\\d+)m", "", log).translate(
            str.maketrans("", "", escapes)
        )
        summary_match = re.search(r"(?m)(?:^|=+\\s+)(\\d+) passed(?:[,\\s=]|$)", clean_log)
"""
        if old not in text:
            raise SystemExit(f"Found single-test marker but could not update existing patch in {path}")
        text = text.replace(old, new, 1)
        path.write_text(text)
        print(f"Updated existing single-test patch: {path}")

    text = path.read_text()
    old_failure_summary = """\
    no_failure_summary = not re.search(
        r"(?m)(?:^|=+\\s+).*(?:failed|error|errors)(?:[,=\\s]|$)",
        clean_log,
        re.IGNORECASE,
    )
"""
    new_failure_summary = """\
    no_failure_summary = not re.search(
        r"(?m)^=+.*\\b(?:failed|error|errors)\\b.*=+$",
        clean_log,
        re.IGNORECASE,
    )
"""
    if old_failure_summary in text:
        text = text.replace(old_failure_summary, new_failure_summary, 1)
        path.write_text(text)
        print(f"Updated empty-param failure-summary patch: {path}")

    text = path.read_text()
    changed = False

    if MARKER_SINGLE not in text and MARKER_EMPTY_PARAM not in text:
        target = "    return test_status_map\n\n\ndef parse_log_seaborn"
        fallback = empty_param_fallback() + single_test_fallback() + target
        if target not in text:
            raise SystemExit(f"Could not find patch target in {path}")

        text = text.replace(target, fallback, 1)
        changed = True
    elif MARKER_SINGLE in text and MARKER_EMPTY_PARAM not in text:
        anchor = f"    if not test_status_map:\n        # {MARKER_SINGLE}\n"
        if anchor not in text:
            raise SystemExit(f"Could not find single-test patch anchor in {path}")
        text = text.replace(anchor, empty_param_fallback() + anchor, 1)
        changed = True
    elif MARKER_EMPTY_PARAM in text and MARKER_SINGLE not in text:
        target = "    return test_status_map\n\n\ndef parse_log_seaborn"
        fallback = single_test_fallback() + target
        if target not in text:
            raise SystemExit(f"Could not find patch target in {path}")
        text = text.replace(target, fallback, 1)
        changed = True

    return changed


def patch_sympy_parser(path: Path) -> bool:
    text = path.read_text()
    if MARKER_SYMPY_INTERRUPTED_OK in text:
        return False

    old = """\
def parse_log_sympy(log: str, test_spec: TestSpec) -> dict[str, str]:
    \"\"\"
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    \"\"\"
    test_status_map = {}
    pattern = r\"(_*) (.*)\\.py:(.*) (_*)\"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f\"{match[1]}.py:{match[2]}\"
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split(\"\\n\"):
        line = line.strip()
        if line.startswith(\"test_\"):
            if line.endswith(\" E\"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(\" F\"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(\" ok\"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map
"""
    new = f"""\
def parse_log_sympy(log: str, test_spec: TestSpec) -> dict[str, str]:
    \"\"\"
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    \"\"\"
    test_status_map = {{}}
    pattern = r\"(_*) (.*)\\.py:(.*) (_*)\"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f\"{{match[1]}}.py:{{match[2]}}\"
        test_status_map[test_case] = TestStatus.FAILED.value

    # {MARKER_SYMPY_INTERRUPTED_OK}
    expected_tests = set(getattr(test_spec, \"FAIL_TO_PASS\", []) or [])
    expected_tests.update(getattr(test_spec, \"PASS_TO_PASS\", []) or [])
    clean_log = re.sub(r\"\\[(\\d+)m\", \"\", log)
    sympy_all_passed_summary = re.search(
        r\"(?m)^=+ tests finished: \\d+ passed(?:, \\d+ expected to fail)?, in .*=+$\",
        clean_log,
    )
    pending_expected_test = None

    for line in log.split(\"\\n\"):
        line = line.strip()
        if (
            line == \"ok\"
            and pending_expected_test is not None
            and sympy_all_passed_summary
        ):
            test_status_map[pending_expected_test] = TestStatus.PASSED.value
            pending_expected_test = None
            continue
        if line.startswith(\"test_\"):
            test = line.split()[0]
            if line.endswith(\" E\"):
                test_status_map[test] = TestStatus.ERROR.value
                pending_expected_test = None
            elif line.endswith(\" F\"):
                test_status_map[test] = TestStatus.FAILED.value
                pending_expected_test = None
            elif line.endswith(\" ok\"):
                test_status_map[test] = TestStatus.PASSED.value
                pending_expected_test = None
            elif test in expected_tests and sympy_all_passed_summary:
                pending_expected_test = test
    return test_status_map
"""
    if old not in text:
        raise SystemExit(f"Could not find Sympy parser target in {path}")
    text = text.replace(old, new, 1)
    path.write_text(text)
    return True


def patch_grading_full_log_supplement() -> bool:
    spec = importlib.util.find_spec("swebench.harness.grading")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate swebench.harness.grading")

    path = Path(spec.origin)
    text = path.read_text()
    if MARKER_FULL_LOG_SUPPLEMENT in text:
        return False

    old = """\
        # Try parsing the content between markers first
        status_map = log_parser(test_content, test_spec)

        # If no test results found between markers (common in Modal environment),
        # try parsing the entire log content as fallback
        if not status_map:
            # Look for pytest output patterns in the entire log content
            # This handles cases where pytest output goes to stderr and isn't captured between markers
            status_map = log_parser(content, test_spec)

        return status_map, True
"""
    new = f"""\
        # Try parsing the content between markers first
        status_map = log_parser(test_content, test_spec)

        # {MARKER_FULL_LOG_SUPPLEMENT}
        # If marker placement truncated explicit expected-test statuses, parse
        # the full raw log and supplement only missing FAIL_TO_PASS/PASS_TO_PASS
        # tests. This keeps scoring tied to tests selected by the benchmark.
        expected_tests = []
        expected_tests.extend(getattr(test_spec, "FAIL_TO_PASS", []) or [])
        expected_tests.extend(getattr(test_spec, "PASS_TO_PASS", []) or [])
        missing_expected_tests = [
            test for test in expected_tests if test not in status_map
        ]
        if status_map and missing_expected_tests:
            full_status_map = log_parser(content, test_spec)
            for test in missing_expected_tests:
                if test in full_status_map:
                    status_map[test] = full_status_map[test]

        # If no test results found between markers (common in Modal environment),
        # try parsing the entire log content as fallback
        if not status_map:
            # Look for pytest output patterns in the entire log content
            # This handles cases where pytest output goes to stderr and isn't captured between markers
            status_map = log_parser(content, test_spec)

        return status_map, True
"""
    if old not in text:
        raise SystemExit(f"Could not find grading supplement target in {path}")
    text = text.replace(old, new, 1)
    path.write_text(text)
    print(f"Patched grading full-log supplement: {path}")
    return True


def main() -> None:
    spec = importlib.util.find_spec("swebench.harness.log_parsers.python")
    if spec is None or spec.origin is None:
        raise SystemExit("Could not locate swebench.harness.log_parsers.python")

    path = Path(spec.origin)
    changed = patch_pytest_parser(path)
    changed = patch_sympy_parser(path) or changed
    changed = patch_grading_full_log_supplement() or changed
    if changed:
        print(f"Patched: {path}")
    else:
        print(f"Already patched: {path}")


if __name__ == "__main__":
    main()
