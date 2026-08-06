"""Tests for infer_error_kind (PR2) — the centralized error classifier."""

from __future__ import annotations

import pytest

from aweagent.core.task.error_kind import infer_error_kind
from aweagent.core.task.types import ErrorKind, EvalResult


def _ev(accepted=False, score=0.0, **details):
    return EvalResult(accepted=accepted, score=score, details=details)


# (finish_reason, eval_result, task_error, expected)
CASES = [
    # 1. Hard infra: no eval happened, runner-level error string present.
    ("error", None, "[RuntimeError] boom", ErrorKind.INFRA_ERROR.value),
    (None, None, "[Timeout] x", ErrorKind.INFRA_ERROR.value),
    # 2. finish_reason lifecycle signals.
    ("timeout", _ev(), None, ErrorKind.TIMEOUT.value),
    ("context_length", _ev(), None, ErrorKind.CONTEXT_LENGTH.value),
    ("error", _ev(), None, ErrorKind.INFRA_ERROR.value),
    # 3a. eval-proxy internal exception.
    ("finish", _ev(raw_status=2), None, ErrorKind.INFRA_ERROR.value),
    ("finish", _ev(exception="proxy blew up"), None, ErrorKind.INFRA_ERROR.value),
    # 3b. swe_bench_pro harness-did-not-run.
    # 3b. swe_bench_pro eval-harness faults are carried as details["error"] VALUES.
    ("finish", _ev(error="entryscript_failed"), None, ErrorKind.INFRA_ERROR.value),
    ("finish", _ev(error="invalid_output_json"), None, ErrorKind.INFRA_ERROR.value),
    # 3c. TB2 verifier died.
    ("finish", _ev(verifier_timed_out=True), None, ErrorKind.INFRA_ERROR.value),
    # 3d. Agent faults stay TASK_FAILURE (kept in the pass-rate denominator).
    ("finish", _ev(error="patch_apply_failed"), None, ErrorKind.TASK_FAILURE.value),
    ("finish", _ev(error="empty_patch"), None, ErrorKind.TASK_FAILURE.value),
    ("finish", _ev(error="missing_agent_final_answer"), None, ErrorKind.TASK_FAILURE.value),
    # 3e. A bare exception string (unknown marker) is infra.
    ("finish", _ev(error="KeyError: 'x'"), None, ErrorKind.INFRA_ERROR.value),
    # 4. Ran to completion.
    ("finish", _ev(accepted=True, score=1.0), None, ErrorKind.OK.value),
    ("max_steps", _ev(accepted=False, score=0.0), None, ErrorKind.TASK_FAILURE.value),
    # A genuine miss with a non-error detail is still task_failure.
    ("finish", _ev(accepted=False, score=0.0, reason="tests_failed"), None,
     ErrorKind.TASK_FAILURE.value),
]


@pytest.mark.parametrize("finish_reason,eval_result,task_error,expected", CASES)
def test_infer_error_kind(finish_reason, eval_result, task_error, expected):
    assert infer_error_kind(
        finish_reason=finish_reason,
        eval_result=eval_result,
        task_error=task_error,
    ) == expected


def test_accepted_overrides_error_detail_absence():
    # score>0, accepted → ok even if reward_source signals nothing special.
    r = _ev(accepted=True, score=0.7, reward_source="reward.json")
    assert infer_error_kind(finish_reason="finish", eval_result=r, task_error=None) == "ok"


def test_error_detail_but_accepted_is_not_infra():
    # If somehow accepted with an 'error' key, don't misclassify as infra.
    r = _ev(accepted=True, score=1.0, error="noise")
    assert infer_error_kind(finish_reason="finish", eval_result=r, task_error=None) == "ok"
