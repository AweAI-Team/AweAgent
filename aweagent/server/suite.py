"""Eval server — programmatic entry point for scoring a checkpoint on benches.

``evaluate()`` is the surface the AutoTrain outer loop calls: given a served
checkpoint (an OpenAI-compatible sglang base_url) and a set of benchmarks, it
runs each bench through the shared pipeline and returns per-bench scores.

Serving is out of scope by design: this layer never launches sglang. The caller
(or an external system) is responsible for serving the checkpoint and passing
its ``base_url``. AweAgent's sglang backend is a plain HTTP client.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from aweagent.core.config.loader import _deep_merge, load_config
from aweagent.core.task.pipeline import run_pipeline
from aweagent.server.benches import get_bench
from aweagent.server.normalize import BenchScore, aggregate

logger = logging.getLogger(__name__)


@dataclass
class SuiteResult:
    """Result of evaluating one checkpoint across a set of benchmarks."""

    per_bench: dict[str, BenchScore] = field(default_factory=dict)
    run_root: str = ""


def _serving_overrides(
    base_url: str,
    model_name: str,
    concurrency: int,
    timeout: int,
    output_path: str,
) -> dict:
    """Config overrides that point the harness at a served checkpoint.

    Only serving + execution knobs — never the harness, prompt, or tools.
    """
    return {
        "llm": {
            "backend": "sglang",
            "base_url": base_url,
            "model": model_name,
        },
        "execution": {
            "max_concurrent": concurrency,
            "output_path": output_path,
        },
        "eval": {"timeout": timeout},
    }


async def evaluate(
    ckpt_base_url: str,
    bench_ids: list[str],
    *,
    concurrency: int = 50,
    timeout: int = 3600,
    model_name: str = "ckpt",
    instance_ids: dict[str, list[str]] | None = None,
    output_root: str | Path = "./results/eval_server",
) -> SuiteResult:
    """Evaluate one checkpoint across ``bench_ids``.

    Args:
        ckpt_base_url: OpenAI-compatible base URL of the served checkpoint.
        bench_ids: benchmark ids registered in ``server.benches.BENCHES``.
        concurrency: max concurrent instances per bench (TaskRunner semaphore).
        timeout: per-instance eval timeout (seconds).
        model_name: model id passed to the sglang backend.
        instance_ids: optional per-bench instance-id subset filter.
        output_root: directory root for per-bench run outputs.

    Returns:
        SuiteResult with a BenchScore per bench (infra failures excluded from
        pass-rate denominators).

    Benches run sequentially to avoid oversubscribing the shared sglang
    endpoint and Docker/portal pool; instances within a bench run concurrently.
    """
    output_root = Path(output_root)
    per_bench: dict[str, BenchScore] = {}

    for bench_id in bench_ids:
        spec = get_bench(bench_id)
        overrides = _serving_overrides(
            base_url=ckpt_base_url,
            model_name=model_name,
            concurrency=concurrency,
            timeout=timeout,
            output_path=str(output_root / bench_id),
        )
        # Apply the bench's own identity overrides first, then serving knobs.
        # Deep-merge (not a shallow spread) so a bench setting e.g. llm.params
        # does not get clobbered by the serving llm.* overrides.
        if spec.overrides:
            overrides = _deep_merge(spec.overrides, overrides)

        config = load_config(spec.config_path, overrides=overrides)
        ids = (instance_ids or {}).get(bench_id)

        logger.info("Evaluating bench %s (concurrency=%d)", bench_id, concurrency)
        runner, results = await run_pipeline(config, instance_ids=ids)
        per_bench[bench_id] = aggregate(bench_id, results, str(runner.run_dir))

    return SuiteResult(per_bench=per_bench, run_root=str(output_root))
