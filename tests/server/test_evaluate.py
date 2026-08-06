"""Tests for the eval-server evaluate() entry point (PR4).

run_pipeline is monkeypatched so no Docker / sglang / real bench data is needed;
we assert that evaluate wires serving overrides into the config and produces a
per-bench score.
"""

from __future__ import annotations

import asyncio

from aweagent.core.task.types import ErrorKind, EvalResult, TaskResult
from aweagent.server import evaluate


class _FakeRunner:
    def __init__(self, run_dir):
        self.run_dir = run_dir


def test_evaluate_applies_serving_overrides_and_aggregates(monkeypatch):
    seen_configs = {}

    async def fake_run_pipeline(config, *, instance_ids=None, **kwargs):
        # Record how the config was overridden per bench.
        seen_configs[config.task.type] = config
        results = [
            TaskResult(
                instance_id="i1",
                eval_result=EvalResult(accepted=True, score=1.0,
                                       error_kind=ErrorKind.OK.value),
            ),
            TaskResult(
                instance_id="i2",
                eval_result=EvalResult(accepted=False, score=0.0,
                                       error_kind=ErrorKind.INFRA_ERROR.value),
            ),
        ]
        return _FakeRunner(f"/tmp/{config.task.type}"), results

    # Patch the symbol imported into suite.py.
    monkeypatch.setattr("aweagent.server.suite.run_pipeline", fake_run_pipeline)

    result = asyncio.run(
        evaluate(
            "http://sglang:30000/v1",
            ["swe_bench_pro", "terminal_bench_v2"],
            concurrency=8,
            timeout=1234,
            model_name="my-ckpt",
        )
    )

    # Both benches scored; infra instance excluded from denominator.
    assert set(result.per_bench) == {"swe_bench_pro", "terminal_bench_v2"}
    for bench_id, score in result.per_bench.items():
        assert score.n_total == 2
        assert score.n_excluded == 1
        assert score.pass_rate == 1.0        # 1 accepted / 1 scored
        assert score.run_dir == f"/tmp/{bench_id}"

    # Serving overrides landed on the config.
    for cfg in seen_configs.values():
        assert cfg.llm.backend == "sglang"
        assert cfg.llm.base_url == "http://sglang:30000/v1"
        assert cfg.llm.model == "my-ckpt"
        assert cfg.execution.max_concurrent == 8
        assert cfg.eval.timeout == 1234


def test_evaluate_unknown_bench_raises(monkeypatch):
    async def fake_run_pipeline(config, **kwargs):
        raise AssertionError("should not be called for unknown bench")

    monkeypatch.setattr("aweagent.server.suite.run_pipeline", fake_run_pipeline)

    try:
        asyncio.run(evaluate("http://x/v1", ["does_not_exist"]))
        raise AssertionError("expected KeyError")
    except KeyError as e:
        assert "does_not_exist" in str(e)


def test_evaluate_num_rollouts_end_to_end(tmp_path, monkeypatch):
    """evaluate(num_rollouts=3) threads N into config, aggregates pass@k,
    and writes missing_rollouts.json for infra-failed rollouts."""
    seen_n = {}
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    async def fake_run_pipeline(config, *, instance_ids=None, **kwargs):
        seen_n[config.task.type] = config.execution.num_rollouts
        # one instance "a" run 3x: [infra, pass, fail]
        results = [
            TaskResult(instance_id="a",
                       eval_result=EvalResult(accepted=False, score=0.0,
                                              error_kind=ErrorKind.INFRA_ERROR.value)),
            TaskResult(instance_id="a",
                       eval_result=EvalResult(accepted=True, score=1.0,
                                              error_kind=ErrorKind.OK.value)),
            TaskResult(instance_id="a",
                       eval_result=EvalResult(accepted=False, score=0.0,
                                              error_kind=ErrorKind.TASK_FAILURE.value)),
        ]
        return _FakeRunner(str(run_dir)), results

    monkeypatch.setattr("aweagent.server.suite.run_pipeline", fake_run_pipeline)
    result = asyncio.run(evaluate("http://x/v1", ["swe_bench_pro"], num_rollouts=3))

    assert seen_n["swe_bench_pro"] == 3          # N threaded into config
    score = result.per_bench["swe_bench_pro"]
    assert score.num_rollouts == 3
    assert score.avg_pass_rate == 0.5            # 1 success / 2 non-infra
    assert score.pass_at_k == 1.0
    assert score.min_rollouts_per_instance == 2
    # missing-rollout report written to disk for re-running
    import json
    report = json.loads((run_dir / "missing_rollouts.json").read_text())
    assert report["num_rollouts"] == 3
    assert report["instances"][0]["instance_id"] == "a"
    assert report["instances"][0]["missing"] == 1
