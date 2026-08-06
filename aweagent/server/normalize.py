"""Score normalization + aggregation for the eval server.

Reconciles the two historically-divergent per-recipe aggregation conventions
into one: read the structured ``error_kind`` to exclude infrastructure failures
from the pass-rate denominator (a dead sandbox or eval crash says nothing about
the checkpoint), clamp scores defensively to [0, 1], and report both pass rate
and mean score.
"""

from __future__ import annotations

from dataclasses import dataclass

from aweagent.core.task.types import ErrorKind, TaskResult

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
class BenchScore:
    """Aggregated score for one benchmark run."""

    bench_id: str
    pass_rate: float          # accepted / (total - excluded)
    mean_score: float         # mean of clamped scores over scored instances
    n_total: int              # all instances attempted
    n_scored: int             # instances that produced a genuine verdict
    n_excluded: int           # infra_error / timeout / context_length
    n_accepted: int
    run_dir: str


def _error_kind(result: TaskResult) -> str:
    if result.eval_result is not None:
        return result.eval_result.error_kind
    # No eval → a runner-level infra failure (retries exhausted).
    return ErrorKind.INFRA_ERROR.value


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def aggregate(bench_id: str, results: list[TaskResult], run_dir: str) -> BenchScore:
    """Aggregate per-instance results into a benchmark-level score.

    Infra failures are excluded from the denominator; genuine task failures
    (score 0, ran fine) stay in it.
    """
    scored = [r for r in results if _error_kind(r) not in _INFRA_KINDS]
    n_excluded = len(results) - len(scored)
    n_accepted = sum(1 for r in scored if r.success)
    scores = [
        _clamp(r.eval_result.score) for r in scored if r.eval_result is not None
    ]

    pass_rate = (n_accepted / len(scored)) if scored else 0.0
    mean_score = (sum(scores) / len(scores)) if scores else 0.0

    return BenchScore(
        bench_id=bench_id,
        pass_rate=pass_rate,
        mean_score=mean_score,
        n_total=len(results),
        n_scored=len(scored),
        n_excluded=n_excluded,
        n_accepted=n_accepted,
        run_dir=str(run_dir),
    )
