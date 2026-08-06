"""Eval server — programmatic entry point for scoring a checkpoint on benches.

``evaluate()`` is the surface the AutoTrain outer loop calls: given a served
checkpoint (an OpenAI-compatible sglang base_url) and a set of benchmarks, it
runs each bench through the shared pipeline and returns per-bench scores.

Serving is out of scope by design: this layer never launches sglang. The caller
(or an external system) is responsible for serving the checkpoint and passing
its ``base_url``. AweAgent's sglang backend is a plain HTTP client.
"""

from __future__ import annotations

import json
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
    num_rollouts: int,
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
            "num_rollouts": num_rollouts,
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
    num_rollouts: int = 1,
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
        num_rollouts: independent full rollouts per instance (default 1). N>1
            runs each instance N times continuously (semaphore-throttled, not
            batched), writes rollout_k/ subdirs, and scores pass@k / avg pass
            rate per instance. Infra-failed rollouts are dropped from each
            instance's denominator and recorded in ``BenchScore.missing_rollouts``
            (also written to ``<bench>/missing_rollouts.json``) for re-running.
        instance_ids: optional per-bench instance-id subset filter.
        output_root: directory root for per-bench run outputs.

    Returns:
        SuiteResult with a BenchScore per bench.

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
            num_rollouts=num_rollouts,
        )
        # Apply the bench's own identity overrides first, then serving knobs.
        # Deep-merge (not a shallow spread) so a bench setting e.g. llm.params
        # does not get clobbered by the serving llm.* overrides.
        if spec.overrides:
            overrides = _deep_merge(spec.overrides, overrides)

        config = load_config(spec.config_path, overrides=overrides)
        ids = (instance_ids or {}).get(bench_id)

        logger.info(
            "Evaluating bench %s (concurrency=%d, num_rollouts=%d)",
            bench_id, concurrency, num_rollouts,
        )
        runner, results = await run_pipeline(config, instance_ids=ids)
        score = aggregate(bench_id, results, str(runner.run_dir), num_rollouts=num_rollouts)
        per_bench[bench_id] = score
        _write_missing_rollouts(runner.run_dir, score)

    return SuiteResult(per_bench=per_bench, run_root=str(output_root))


def _write_missing_rollouts(run_dir: Path | None, score: BenchScore) -> None:
    """Write the per-instance missing-rollout report so infra-failed rollouts
    can be re-run. Skipped when nothing is missing or the run dir is unknown."""
    if run_dir is None or not score.missing_rollouts:
        return
    from dataclasses import asdict

    payload = {
        "bench_id": score.bench_id,
        "num_rollouts": score.num_rollouts,
        "instances": [asdict(m) for m in score.missing_rollouts],
    }
    try:
        (Path(run_dir) / "missing_rollouts.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )
    except Exception as e:  # best-effort; never fail the eval over a report
        logger.warning("Failed to write missing_rollouts.json: %s", e)
