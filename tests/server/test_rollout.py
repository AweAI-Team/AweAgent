"""Tests for the rollout-server rollout() entry point (config-path driven).

run_pipeline is monkeypatched so no Docker / teacher endpoint / real data is
needed; we assert that rollout loads the config, threads data_file/output_root
overrides, enumerates trajectory files, and summarizes usability. Pure helpers
(enumeration, summary) are unit-tested directly.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from aweagent.core.task.types import ErrorKind, EvalResult, TaskResult
from aweagent.server.rollout import (
    RolloutResult,
    _enumerate_trajectories,
    _summarize,
    rollout,
)

# The package binds ``rollout`` to the *function*, shadowing the submodule
# attribute — fetch the real module via sys.modules for monkeypatching.
rollout_module = sys.modules["aweagent.server.rollout"]


class _FakeRunner:
    def __init__(self, run_dir):
        self.run_dir = run_dir


def _results():
    return [
        TaskResult(
            instance_id="i1",
            eval_result=EvalResult(accepted=True, score=1.0, error_kind=ErrorKind.OK.value),
        ),
        TaskResult(
            instance_id="i2",
            eval_result=EvalResult(accepted=False, score=0.0,
                                   error_kind=ErrorKind.TASK_FAILURE.value),
        ),
        TaskResult(
            instance_id="i3",
            eval_result=EvalResult(accepted=False, score=0.0,
                                   error_kind=ErrorKind.INFRA_ERROR.value),
        ),
    ]


# ── _enumerate_trajectories ─────────────────────────────────────────────────────

def test_enumerate_single_rollout(tmp_path):
    (tmp_path / "trajectories.jsonl").write_text("{}\n")
    (tmp_path / "results.jsonl").write_text("{}\n")
    paths = _enumerate_trajectories(tmp_path, 1)
    assert len(paths) == 1
    assert paths[0]["rollout"] == 0
    assert paths[0]["trajectories"].endswith("trajectories.jsonl")


def test_enumerate_multi_rollout(tmp_path):
    for k in range(3):
        d = tmp_path / f"rollout_{k}"
        d.mkdir()
        (d / "trajectories.jsonl").write_text("{}\n")
        (d / "results.jsonl").write_text("{}\n")
    paths = _enumerate_trajectories(tmp_path, 3)
    assert [p["rollout"] for p in paths] == [0, 1, 2]
    assert all("rollout_" in p["trajectories"] for p in paths)


def test_enumerate_tolerates_missing_rollout(tmp_path):
    d0 = tmp_path / "rollout_0"
    d0.mkdir()
    (d0 / "results.jsonl").write_text("{}\n")
    paths = _enumerate_trajectories(tmp_path, 2)
    assert [p["rollout"] for p in paths] == [0]
    assert paths[0]["trajectories"] is None      # only results.jsonl existed


# ── _summarize ──────────────────────────────────────────────────────────────────

def test_summarize_counts():
    n_acc, n_infra, counts = _summarize(_results())
    assert n_acc == 1
    assert n_infra == 1
    assert counts[ErrorKind.OK.value] == 1
    assert counts[ErrorKind.TASK_FAILURE.value] == 1
    assert counts[ErrorKind.INFRA_ERROR.value] == 1


# ── rollout() end-to-end (run_pipeline mocked) ──────────────────────────────────

def _write_config(tmp_path, *, eval_enabled=True, num_rollouts=1):
    cfg = tmp_path / "rollout_exp.yaml"
    cfg.write_text(
        "llm:\n"
        "  backend: openai\n"
        "  base_url: http://teacher/v1\n"
        "  model: dpsk\n"
        "  params: {temperature: 1.0}\n"
        "agent:\n"
        "  type: search_swe\n"
        "task:\n"
        "  type: scale_swe\n"
        "  dataset_id: scale_swe\n"
        "  data_file: /orig.jsonl\n"
        f"eval:\n  enabled: {'true' if eval_enabled else 'false'}\n"
        f"execution:\n  num_rollouts: {num_rollouts}\n"
        "  output_path: /orig_out\n"
    )
    return cfg


def test_rollout_loads_config_and_summarizes(monkeypatch, tmp_path):
    seen = {}

    async def fake_run_pipeline(config, *, instance_ids=None, save_trajectories=True, **kw):
        seen["config"] = config
        seen["save_trajectories"] = save_trajectories
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "trajectories.jsonl").write_text("{}\n")
        (run_dir / "results.jsonl").write_text("{}\n")
        return _FakeRunner(run_dir), _results()

    monkeypatch.setattr(rollout_module, "run_pipeline", fake_run_pipeline)
    cfg = _write_config(tmp_path)

    result = asyncio.run(rollout(str(cfg)))

    assert isinstance(result, RolloutResult)
    # Config's teacher wiring is used as-is.
    assert seen["config"].llm.backend == "openai"
    assert seen["config"].llm.base_url == "http://teacher/v1"
    assert seen["save_trajectories"] is True
    assert result.n_instances == 3
    assert result.n_trajectories == 3
    assert result.n_accepted == 1
    assert result.n_infra == 1
    assert result.eval_skipped is False
    assert len(result.rollout_paths) == 1


def test_rollout_overrides_data_file_and_output_root(monkeypatch, tmp_path):
    seen = {}

    async def fake_run_pipeline(config, *, instance_ids=None, save_trajectories=True, **kw):
        seen["config"] = config
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunner(run_dir), []

    monkeypatch.setattr(rollout_module, "run_pipeline", fake_run_pipeline)
    cfg = _write_config(tmp_path)

    asyncio.run(rollout(str(cfg), data_file="/new.jsonl", output_root="/new_out"))

    assert seen["config"].task.data_file == "/new.jsonl"      # overridden
    assert seen["config"].execution.output_path == "/new_out"  # overridden


def test_rollout_forces_save_trajectories(monkeypatch, tmp_path):
    seen = {}

    async def fake_run_pipeline(config, *, instance_ids=None, save_trajectories=True, **kw):
        seen["config"] = config
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunner(run_dir), []

    monkeypatch.setattr(rollout_module, "run_pipeline", fake_run_pipeline)
    # Config tries to disable trajectories; rollout must force it back on.
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "task: {type: scale_swe, dataset_id: scale_swe, data_file: /d.jsonl}\n"
        "execution: {save_trajectories: false, num_rollouts: 1}\n"
    )
    asyncio.run(rollout(str(cfg)))
    assert seen["config"].execution.save_trajectories is True


def test_rollout_eval_disabled_marks_skipped(monkeypatch, tmp_path):
    async def fake_run_pipeline(config, *, instance_ids=None, save_trajectories=True, **kw):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return _FakeRunner(run_dir), []

    monkeypatch.setattr(rollout_module, "run_pipeline", fake_run_pipeline)
    cfg = _write_config(tmp_path, eval_enabled=False)
    result = asyncio.run(rollout(str(cfg)))
    assert result.eval_skipped is True


def test_rollout_missing_config_raises():
    with pytest.raises(FileNotFoundError, match="rollout config not found"):
        asyncio.run(rollout("/no/such/config.yaml"))
