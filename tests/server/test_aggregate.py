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


# ── multi-rollout (num_rollouts > 1) ──────────────────────────────────────────


def test_aggregate_n1_byte_equivalent():
    """At num_rollouts=1, grouped aggregation equals the old flat behavior."""
    results = [
        _result("a", True, 1.0, ErrorKind.OK.value),
        _result("b", False, 0.0, ErrorKind.TASK_FAILURE.value),
    ]
    s = aggregate("bench", results, "/tmp/run", num_rollouts=1)
    assert s.pass_rate == s.avg_pass_rate == s.pass_at_k == 0.5
    assert s.num_rollouts == 1
    assert s.n_instances == 2 and s.n_scored_instances == 2
    assert s.missing_rollouts == []


def test_aggregate_pass_any_multi_rollout():
    # instance "a" run 3x: [pass, fail, fail]
    results = [
        _result("a", True, 1.0, ErrorKind.OK.value),
        _result("a", False, 0.0, ErrorKind.TASK_FAILURE.value),
        _result("a", False, 0.0, ErrorKind.TASK_FAILURE.value),
    ]
    s = aggregate("bench", results, "/tmp/run", num_rollouts=3)
    assert s.n_instances == 1 and s.n_scored_instances == 1
    assert s.pass_at_k == 1.0            # >=1 success
    assert abs(s.avg_pass_rate - 1 / 3) < 1e-9
    assert s.missing_rollouts == []      # 3 valid, none missing


def test_aggregate_partial_infra_denominator():
    # "a" = [infra, pass, fail]: denominator is the 2 non-infra rollouts
    results = [
        _result("a", False, 0.0, ErrorKind.INFRA_ERROR.value),
        _result("a", True, 1.0, ErrorKind.OK.value),
        _result("a", False, 0.0, ErrorKind.TASK_FAILURE.value),
    ]
    s = aggregate("bench", results, "/tmp/run", num_rollouts=3)
    assert s.avg_pass_rate == 0.5        # 1 success / 2 valid
    assert s.pass_at_k == 1.0
    assert s.min_rollouts_per_instance == 2
    assert len(s.missing_rollouts) == 1
    m = s.missing_rollouts[0]
    assert m.instance_id == "a" and m.expected == 3 and m.valid == 2 and m.missing == 1
    assert m.missing_reasons == [ErrorKind.INFRA_ERROR.value]


def test_aggregate_all_infra_instance_dropped():
    # "a" all infra → dropped from scored; "b" real → scored.
    results = [
        _result("a", False, 0.0, ErrorKind.INFRA_ERROR.value),
        _result("a", False, 0.0, ErrorKind.TIMEOUT.value),
        _result("b", True, 1.0, ErrorKind.OK.value),
        _result("b", False, 0.0, ErrorKind.TASK_FAILURE.value),
    ]
    s = aggregate("bench", results, "/tmp/run", num_rollouts=2)
    assert s.n_instances == 2 and s.n_scored_instances == 1   # "a" dropped
    assert s.pass_at_k == 1.0 and s.avg_pass_rate == 0.5      # only "b"
    # "a" fully missing (2 of 2), "b" complete
    miss = {m.instance_id: m for m in s.missing_rollouts}
    assert miss["a"].missing == 2 and miss["a"].valid == 0
    assert "b" not in miss


def test_aggregate_mean_score_grouping():
    # two rollouts of one instance: scores 0.4, 0.6 → instance mean 0.5
    results = [
        _result("a", False, 0.4, ErrorKind.TASK_FAILURE.value),
        _result("a", False, 0.6, ErrorKind.TASK_FAILURE.value),
    ]
    s = aggregate("bench", results, "/tmp/run", num_rollouts=2)
    assert abs(s.mean_score - 0.5) < 1e-9
