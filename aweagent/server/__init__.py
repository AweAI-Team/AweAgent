"""AweAgent eval server — programmatic multi-benchmark evaluation of a checkpoint.

Public API:
    evaluate     — async entry point: score a served checkpoint across benches.
    SuiteResult  — per-bench scores for one checkpoint.
    BenchScore   — aggregated score for a single benchmark.
    BENCHES      — the benchmark registry (bench_id → BenchSpec).
"""

from aweagent.server.benches import BENCHES, BenchSpec, get_bench
from aweagent.server.normalize import BenchScore, aggregate
from aweagent.server.suite import SuiteResult, evaluate

__all__ = [
    "BENCHES",
    "BenchScore",
    "BenchSpec",
    "SuiteResult",
    "aggregate",
    "evaluate",
    "get_bench",
]
