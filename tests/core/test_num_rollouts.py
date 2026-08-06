"""Tests for num_rollouts fan-out in TaskRunner (multi-rollout eval).

These avoid Docker/agents by monkeypatching _run_instance; they exercise the
fan-out shape, per-rollout subdir routing, N==1 back-compat, and the
per-rollout Instance isolation (the load-bearing deepcopy fix).
"""

from __future__ import annotations

import asyncio
import json

from aweagent.core.task.runner import TaskRunner
from aweagent.core.task.types import EvalResult, Instance, TaskResult


class _FakeTask:
    """Minimal Task stand-in: yields a fixed set of instances."""

    def __init__(self, instances):
        self._instances = instances

    def get_instances(self, instance_ids=None):
        return list(self._instances)

    def default_evaluator(self, timeout=None):
        return None


def _make_runner(tmp_path, instances, num_rollouts):
    return TaskRunner(
        task=_FakeTask(instances),
        agent_factory=lambda **_: None,  # never called (we patch _run_instance)
        llm_config=type("L", (), {"model": "fake-model"})(),
        runtime_config=None,
        evaluator=None,
        max_concurrent=8,
        output_path=str(tmp_path),
        save_trajectories=False,
        num_rollouts=num_rollouts,
    )


def _instances(n):
    return [Instance(id=f"t{i}", dataset_id="fake") for i in range(n)]


def test_fanout_count_is_n_times_instances(tmp_path, monkeypatch):
    """N rollouts x M instances => N*M scheduled runs, one continuous gather."""
    seen = []

    async def fake_run_instance(self, instance):
        seen.append(instance.id)
        return TaskResult(instance_id=instance.id,
                          eval_result=EvalResult(accepted=True, score=1.0))

    monkeypatch.setattr(TaskRunner, "_run_instance", fake_run_instance)
    runner = _make_runner(tmp_path, _instances(3), num_rollouts=4)
    results = asyncio.run(runner.run_all())
    assert len(results) == 12          # 3 instances x 4 rollouts
    assert len(seen) == 12
    assert sorted(set(seen)) == ["t0", "t1", "t2"]


def test_n1_flat_layout_no_subdir(tmp_path, monkeypatch):
    """num_rollouts=1 keeps the flat run_dir/results.jsonl, no rollout_0/, no rollout key."""
    async def fake_run_instance(self, instance):
        return TaskResult(instance_id=instance.id,
                          eval_result=EvalResult(accepted=True, score=1.0))

    monkeypatch.setattr(TaskRunner, "_run_instance", fake_run_instance)
    runner = _make_runner(tmp_path, _instances(2), num_rollouts=1)
    asyncio.run(runner.run_all())
    run_dir = runner.run_dir
    assert (run_dir / "results.jsonl").exists()
    assert not (run_dir / "rollout_0").exists()
    rows = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l]
    assert all("rollout" not in r for r in rows)   # byte-identical: no rollout key


def test_multi_rollout_subdirs_and_rollout_key(tmp_path, monkeypatch):
    """num_rollouts>1 writes rollout_k/ subdirs, each row carries its rollout idx."""
    async def fake_run_instance(self, instance):
        return TaskResult(instance_id=instance.id,
                          eval_result=EvalResult(accepted=True, score=1.0))

    monkeypatch.setattr(TaskRunner, "_run_instance", fake_run_instance)
    runner = _make_runner(tmp_path, _instances(2), num_rollouts=3)
    asyncio.run(runner.run_all())
    run_dir = runner.run_dir
    for k in range(3):
        sub = run_dir / f"rollout_{k}"
        assert (sub / "results.jsonl").exists()
        rows = [json.loads(l) for l in (sub / "results.jsonl").read_text().splitlines() if l]
        assert len(rows) == 2
        assert all(r["rollout"] == k for r in rows)


def test_instance_isolation_no_metadata_crosstalk(tmp_path, monkeypatch):
    """Concurrent rollouts of the same instance must not share metadata.

    Regression test for the deepcopy fix: _run_instance writes into
    instance.metadata; without per-rollout isolation the N concurrent copies
    would race on one dict.
    """
    object_ids = set()

    async def fake_run_instance(self, instance):
        # each rollout should receive a distinct Instance object at N>1
        object_ids.add(id(instance))
        instance.metadata["scratch"] = instance.id  # mutate; must not leak
        return TaskResult(instance_id=instance.id,
                          eval_result=EvalResult(accepted=True, score=1.0))

    monkeypatch.setattr(TaskRunner, "_run_instance", fake_run_instance)
    insts = _instances(2)
    runner = _make_runner(tmp_path, insts, num_rollouts=3)
    asyncio.run(runner.run_all())
    # 2 instances x 3 rollouts = 6 distinct Instance objects (deepcopy per run)
    assert len(object_ids) == 6
    # original instances untouched (deepcopy isolated the mutation)
    assert all("scratch" not in i.metadata for i in insts)
