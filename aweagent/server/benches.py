"""Benchmark registry for the eval server.

Maps a stable ``bench_id`` to the fixed-harness config that defines it. The
eval server loads this config, overrides only the serving + execution knobs
(never the harness/prompt), and runs it through the shared pipeline.

Initial focus: code (search_swe) + terminal (terminus_2). Extend ``BENCHES``
to add more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Config files live under AweAgent/configs/tasks/. Resolve relative to the repo
# root (three parents up from this file: aweagent/server/benches.py).
_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "tasks"


@dataclass(frozen=True)
class BenchSpec:
    """Definition of one evaluation benchmark."""

    bench_id: str
    config_path: str
    # Fixed config overrides that are part of the bench's identity (rarely
    # needed; serving/concurrency overrides are applied by the eval server).
    overrides: dict = field(default_factory=dict)


BENCHES: dict[str, BenchSpec] = {
    "swe_bench_pro": BenchSpec(
        bench_id="swe_bench_pro",
        config_path=str(_CONFIGS / "swe_bench_pro.yaml"),
    ),
    "terminal_bench_v2": BenchSpec(
        bench_id="terminal_bench_v2",
        config_path=str(_CONFIGS / "terminal_bench_v2.yaml"),
    ),
}


def get_bench(bench_id: str) -> BenchSpec:
    if bench_id not in BENCHES:
        available = ", ".join(sorted(BENCHES)) or "(none)"
        raise KeyError(f"Unknown bench_id {bench_id!r}. Available: {available}")
    return BENCHES[bench_id]
