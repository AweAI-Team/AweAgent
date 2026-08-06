"""AweAgent server — programmatic multi-benchmark evaluation + rollout collection.

Public API:
    evaluate     — async entry point: score a served checkpoint across benches.
    SuiteResult  — per-bench scores for one checkpoint.
    BenchScore   — aggregated score for a single benchmark.
    BENCHES      — the benchmark registry (bench_id → BenchSpec).
    rollout      — async entry point: collect teacher trajectories from a config.
    RolloutResult — trajectory location + usability metadata for a rollout run.
"""

from aweagent.server.benches import BENCHES, BenchSpec, get_bench
from aweagent.server.normalize import BenchScore, aggregate
from aweagent.server.rollout import RolloutResult, rollout
from aweagent.server.suite import SuiteResult, evaluate

__all__ = [
    "BENCHES",
    "BenchScore",
    "BenchSpec",
    "RolloutResult",
    "SuiteResult",
    "aggregate",
    "evaluate",
    "get_bench",
    "rollout",
]


