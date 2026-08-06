"""Score normalization + aggregation for the eval server.

Reconciles the two historically-divergent per-recipe aggregation conventions
into one: read the structured ``error_kind`` to exclude infrastructure failures
from the pass-rate denominator (a dead sandbox or eval crash says nothing about
the checkpoint), clamp scores defensively to [0, 1], and report both pass rate
and mean score.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from aweagent.core.task.error_kind import infer_error_kind
from aweagent.core.task.types import ErrorKind, TaskResult

logger = logging.getLogger(__name__)

# error_kind values that denote infrastructure failures — excluded from the
# scored denominator rather than counted as task failures.
_INFRA_KINDS = frozenset(
    {
        ErrorKind.INFRA_ERROR.value,
        ErrorKind.TIMEOUT.value,
        ErrorKind.CONTEXT_LENGTH.value,
    }
)


@dataclass
class InstanceRollups:
    """Per-instance summary of missing (infra-failed) rollouts, for re-running.

    When an instance is run ``num_rollouts`` times, some rollouts may fail on
    infrastructure (dead sandbox, timeout, eval crash). Those are dropped from
    the pass-rate denominator (they say nothing about the model), but recorded
    here so the caller knows exactly which instances are short and by how many.
    """

    instance_id: str
    expected: int             # num_rollouts requested
    valid: int                # non-infra rollouts actually scored
    missing: int              # expected - valid (infra failures)
    missing_reasons: list[str] = field(default_factory=list)  # error_kind per missing rollout


@dataclass
class BenchScore:
    """Aggregated score for one benchmark run.

    Scores are computed PER INSTANCE first (grouping the N rollouts of each
    instance), then averaged across instances — so multiple rollouts of the
    same instance do not micro-average as if they were independent instances.
    """

    bench_id: str
    pass_rate: float          # == avg_pass_rate (legacy field name kept)
    mean_score: float         # mean over instances of their mean clamped score
    n_total: int              # all rollout results (n_instances * expected, minus dead)
    n_scored: int             # non-infra rollouts
    n_excluded: int           # infra rollouts (infra_error / timeout / context_length)
    n_accepted: int           # successful non-infra rollouts
    run_dir: str
    num_rollouts: int = 1
    n_instances: int = 0            # distinct instance_ids seen
    n_scored_instances: int = 0     # instances with >=1 non-infra rollout
    min_rollouts_per_instance: int = 0   # smallest non-infra count among scored instances
    pass_at_k: float = 0.0          # frac of scored instances with >=1 non-infra success
    avg_pass_rate: float = 0.0      # mean over scored instances of successes / non-infra count
    missing_rollouts: list[InstanceRollups] = field(default_factory=list)


def _error_kind(result: TaskResult) -> str:
    """The instance's error_kind, consistent with what the runner persisted.

    Prefer the value the runner already stamped on eval_result. When there is
    no eval_result, re-derive it the same way the runner does (infer_error_kind)
    rather than assuming infra — an agent that finished normally with an empty
    submission is a genuine TASK_FAILURE and must stay in the denominator.
    """
    if result.eval_result is not None:
        return result.eval_result.error_kind
    return infer_error_kind(
        finish_reason=(result.agent_result.finish_reason if result.agent_result else None),
        eval_result=None,
        task_error=result.error,
    )


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def aggregate(
    bench_id: str,
    results: list[TaskResult],
    run_dir: str,
    *,
    num_rollouts: int = 1,
) -> BenchScore:
    """Aggregate per-instance results into a benchmark-level score.

    Groups the ``num_rollouts`` rollouts of each instance by ``instance_id``.
    Per instance: infra rollouts are dropped, pass rate = successes / non-infra
    count, and an instance whose rollouts are ALL infra is dropped entirely
    (mirrors single-rollout infra exclusion at instance granularity). The bench
    score averages over the surviving (scored) instances.

    NOTE (variable-k): ``pass_at_k`` is a pass-any estimator at each instance's
    effective k (its non-infra count, which can be < ``num_rollouts`` when some
    rollouts hit infra), not a textbook fixed-k pass@k. ``min_rollouts_per_instance``
    and ``missing_rollouts`` expose the shortfall; re-run the missing ones for a
    clean fixed-k estimate.

    Equivalent to the old flat aggregate at ``num_rollouts == 1`` (each group
    has one result), provided instance_ids are unique — true for all benches.
    """
    groups: dict[str, list[TaskResult]] = defaultdict(list)
    for r in results:
        groups[r.instance_id].append(r)

    inst_pass_any: list[float] = []
    inst_avg_rate: list[float] = []
    inst_mean_score: list[float] = []
    non_infra_counts: list[int] = []
    missing: list[InstanceRollups] = []
    n_scored_roll = 0
    n_accepted_roll = 0

    for instance_id, rs in groups.items():
        scored = [r for r in rs if _error_kind(r) not in _INFRA_KINDS]
        infra = [r for r in rs if _error_kind(r) in _INFRA_KINDS]
        n_scored_roll += len(scored)

        # Record shortfall for re-running (expected uses num_rollouts, not
        # len(rs): a fully-dead instance may have fewer rows than expected).
        n_missing = num_rollouts - len(scored)
        if n_missing > 0:
            missing.append(InstanceRollups(
                instance_id=instance_id,
                expected=num_rollouts,
                valid=len(scored),
                missing=n_missing,
                missing_reasons=[_error_kind(r) for r in infra],
            ))

        if not scored:  # all rollouts infra → drop instance from scoring
            continue

        non_infra_counts.append(len(scored))
        n_succ = sum(1 for r in scored if r.success)
        n_accepted_roll += n_succ
        inst_pass_any.append(1.0 if n_succ > 0 else 0.0)
        inst_avg_rate.append(n_succ / len(scored))
        sc = [_clamp(r.eval_result.score) for r in scored if r.eval_result is not None]
        if sc:
            inst_mean_score.append(sum(sc) / len(sc))

    n_si = len(inst_pass_any)
    if n_si and min(non_infra_counts) < num_rollouts:
        logger.warning(
            "bench %s: some instances have < %d non-infra rollouts (min=%d); "
            "pass_at_k is a variable-k pass-any estimator — see missing_rollouts",
            bench_id, num_rollouts, min(non_infra_counts),
        )

    pass_any = sum(inst_pass_any) / n_si if n_si else 0.0
    avg_pass_rate = sum(inst_avg_rate) / n_si if n_si else 0.0
    mean_score = sum(inst_mean_score) / len(inst_mean_score) if inst_mean_score else 0.0

    return BenchScore(
        bench_id=bench_id,
        pass_rate=avg_pass_rate,
        mean_score=mean_score,
        n_total=len(results),
        n_scored=n_scored_roll,
        n_excluded=len(results) - n_scored_roll,
        n_accepted=n_accepted_roll,
        run_dir=str(run_dir),
        num_rollouts=num_rollouts,
        n_instances=len(groups),
        n_scored_instances=n_si,
        min_rollouts_per_instance=(min(non_infra_counts) if non_infra_counts else 0),
        pass_at_k=pass_any,
        avg_pass_rate=avg_pass_rate,
        missing_rollouts=missing,
    )
