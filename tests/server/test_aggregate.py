"""Tests for the eval-server aggregation (PR4)."""

from __future__ import annotations

from aweagent.core.task.types import ErrorKind, EvalResult, TaskResult
from aweagent.server.normalize import aggregate


def _result(instance_id, accepted, score, error_kind, eval_present=True, error=None):
    ev = None
    if eval_present:
        ev = EvalResult(accepted=accepted, score=score, error_kind=error_kind)
    return TaskResult(instance_id=instance_id, eval_result=ev, error=error)


def test_aggregate_excludes_infra_from_denominator():
    results = [
        _result("a", True, 1.0, ErrorKind.OK.value),
        _result("b", False, 0.0, ErrorKind.TASK_FAILURE.value),
        _result("c", False, 0.0, ErrorKind.INFRA_ERROR.value),   # excluded
        _result("d", False, 0.0, ErrorKind.TIMEOUT.value),       # excluded
        _result("e", False, 0.0, ErrorKind.CONTEXT_LENGTH.value),  # excluded
    ]
    s = aggregate("bench", results, "/tmp/run")
    assert s.n_total == 5
    assert s.n_excluded == 3
    assert s.n_scored == 2          # a + b
    assert s.n_accepted == 1        # a
    assert s.pass_rate == 0.5       # 1 / 2
    assert s.bench_id == "bench"
    assert s.run_dir == "/tmp/run"


def test_aggregate_missing_eval_is_infra():
    # A result with no eval_result (retries exhausted) counts as infra.
    results = [
        _result("a", True, 1.0, ErrorKind.OK.value),
        _result("b", False, 0.0, "", eval_present=False, error="[Err] boom"),
    ]
    s = aggregate("bench", results, "/tmp/run")
    assert s.n_excluded == 1
    assert s.n_scored == 1
    assert s.pass_rate == 1.0


def test_aggregate_clamps_scores():
    results = [
        _result("a", True, 2.5, ErrorKind.OK.value),   # out-of-range reward
        _result("b", False, -1.0, ErrorKind.TASK_FAILURE.value),
    ]
    s = aggregate("bench", results, "/tmp/run")
    # scores clamped to [0,1]: 1.0 and 0.0 → mean 0.5
    assert s.mean_score == 0.5


def test_aggregate_empty():
    s = aggregate("bench", [], "/tmp/run")
    assert s.pass_rate == 0.0
    assert s.mean_score == 0.0
    assert s.n_total == 0
