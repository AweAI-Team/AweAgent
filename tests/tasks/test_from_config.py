"""Tests for Task.from_config and the task registry (PR1)."""

from __future__ import annotations

from aweagent.core.config.schema import AweAgentConfig, TaskConfig
from aweagent.core.task.registry import task_registry


def _config(**task_kwargs) -> AweAgentConfig:
    """Build a config with the given task.* overrides."""
    return AweAgentConfig(task=TaskConfig(**task_kwargs))


def test_registry_lists_builtin_tasks():
    names = task_registry.list_available()
    for expected in ("beyond_swe", "scale_swe", "swe_bench_pro"):
        assert expected in names


def test_registry_get_returns_class():
    from aweagent.tasks.scale_swe.task import ScaleSWETask

    assert task_registry.get("scale_swe") is ScaleSWETask


def test_registry_entry_point_discovery():
    """A fresh registry on the aweagent.task group discovers built-ins."""
    from aweagent.plugins.registry import Registry

    reg: Registry[type] = Registry("aweagent.task")
    cls = reg.get("beyond_swe")
    assert cls.__name__ == "BeyondSWETask"


def test_scale_swe_from_config_default():
    """ScaleSWE uses the default from_config (dataset_id + data_file only)."""
    cfg = _config(type="scale_swe", dataset_id="scale_swe", data_file="/tmp/x.jsonl")
    task = task_registry.get("scale_swe").from_config(cfg)
    assert task.dataset_id == "scale_swe"
    assert task.data_file == "/tmp/x.jsonl"


def test_beyond_swe_from_config_reads_search_mode():
    cfg = _config(type="beyond_swe", dataset_id="beyond_swe", test_suite_dir="/suite")
    cfg.agent.enable_search = True
    task = task_registry.get("beyond_swe").from_config(cfg)
    assert task._search_mode is True
    assert task._test_suite_dir == "/suite"


def test_swe_bench_pro_from_config_reads_split_args():
    """The formerly CLI-only split args now flow through config.task.*."""
    cfg = _config(
        type="swe_bench_pro",
        dataset_id="swe_bench_pro",
        all_languages=True,
        split_num=4,
        split_id=2,
    )
    task = task_registry.get("swe_bench_pro").from_config(cfg)
    assert task._all_languages is True
    assert task._split_num == 4
    assert task._split_id == 2


def test_taskconfig_split_defaults():
    """New schema fields default to unset (behavior unchanged when omitted)."""
    tc = TaskConfig()
    assert tc.all_languages is False
    assert tc.split_num is None
    assert tc.split_id is None


def test_denovo_swe_from_config_reads_own_params():
    """DeNovoSWE reads its recipe-only params from config.task.* now."""
    cfg = _config(
        type="denovo_swe",
        dataset_id="denovo_swe",
        data_file="/tmp/x.jsonl",
        validate_run=True,
        eval_iters=3,
        prompt_version="v1",
    )
    task = task_registry.get("denovo_swe").from_config(cfg)
    assert task._validate_run is True
    assert task._eval_iters == 3
    assert task._prompt_version == "v1"


def test_registry_lists_denovo():
    assert "denovo_swe" in task_registry.list_available()
