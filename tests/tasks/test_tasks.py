"""Tests for task implementations (BeyondSWE)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from awe_agent.core.task.types import Instance
from awe_agent.tasks.beyond_swe.prompts import get_beyond_swe_prompt
from awe_agent.tasks.beyond_swe.task import BeyondSWETask


def _write_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ── BeyondSWE ────────────────────────────────────────────────────────────────

_BEYOND_SWE_INSTANCES = [
    {
        "instance_id": "mylib_doc2repo_001",
        "task": "Doc2Repo",
        "workdir": "/workspace",
        "image_url": "ubuntu:22.04",
        "REPO_DOCUMENT_CONTENT": "# MyLib\n\nA library for data processing...",
        "base_commit": "aaa111",
        "language": "python",
    },
    {
        "instance_id": "django_crossrepo_002",
        "task": "CrossRepo",
        "workdir": "/workspace",
        "image_url": "python:3.11",
        "problem_statement": "Import fails across modules after rename",
        "base_commit": "bbb222",
        "parent_commit": "bbb221",
        "FAIL_TO_PASS": '["test_import"]',
        "language": "python",
    },
    {
        "instance_id": "flask_depmigrate_003",
        "task": "DepMigrate",
        "workdir": "/workspace",
        "image_url": "python:3.11",
        "problem_statement": "Refactor request handling to use async",
        "base_commit": "ccc333",
        "f2p_patch": "diff --git ...",
        "language": "python",
    },
    {
        "instance_id": "scipy_domainfix_004",
        "task": "DomainFix",
        "workdir": "/workspace",
        "image_url": "python:3.11",
        "problem_statement": "Numerical instability in SVD for near-singular matrices",
        "base_commit": "ddd444",
        "language": "python",
    },
]


def test_beyond_swe_task_from_instances():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances()
    assert len(instances) == 4


def test_beyond_swe_task_types():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances()
    types = [i.metadata["task_type"] for i in instances]
    assert "doc2repo" in types
    assert "crossrepo" in types
    assert "depmigrate" in types
    assert "domainfix" in types


def test_beyond_swe_doc2repo_prompt():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["mylib_doc2repo_001"])
    assert len(instances) == 1
    prompt = task.get_prompt(instances[0])
    assert "MyLib" in prompt
    assert "implement" in prompt.lower()
    assert "specification" in prompt.lower()


def test_beyond_swe_crossrepo_prompt():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["django_crossrepo_002"])
    prompt = task.get_prompt(instances[0])
    assert "Import fails" in prompt
    assert "bbb222" in prompt


def test_beyond_swe_depmigrate_prompt():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["flask_depmigrate_003"])
    prompt = task.get_prompt(instances[0])
    assert "Refactor" in prompt
    assert "async" in prompt.lower()


def test_beyond_swe_domainfix_prompt():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["scipy_domainfix_004"])
    prompt = task.get_prompt(instances[0])
    assert "Numerical instability" in prompt


def test_beyond_swe_setup_commands_crossrepo():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["django_crossrepo_002"])
    commands = task.get_setup_commands(instances[0])
    assert any("git checkout bbb222" in cmd for cmd in commands)


def test_beyond_swe_prompt_unknown_type():
    with pytest.raises(ValueError, match="Unknown BeyondSWE task type"):
        get_beyond_swe_prompt(task_type="unknown_type")


def test_beyond_swe_from_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in _BEYOND_SWE_INSTANCES:
            f.write(json.dumps(item) + "\n")
        f.flush()
        task = BeyondSWETask(data_file=f.name)
        instances = task.get_instances()
        assert len(instances) == 4


# ── Prompt routing ────────────────────────────────────────────────────────────

def test_prompt_routing_beyond_swe():
    """BeyondSWE routes resolve correctly for all task types."""
    from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys

    sys_key, usr_key = resolve_prompt_keys("beyond_swe", "doc2repo", False)
    assert sys_key == "beyondswe"
    assert usr_key == "doc2repo"

    sys_key, usr_key = resolve_prompt_keys("beyond_swe", "domainfix", True)
    assert sys_key == "search_domainfix"
    assert usr_key == "search_domainfix"


def test_prompt_routing_fallback():
    """Unknown dataset falls back to default route."""
    from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys

    sys_key, usr_key = resolve_prompt_keys("unknown_dataset", None, False)
    assert sys_key == "beyondswe"
    assert usr_key == "domainfix"


def test_search_mode_beyond_swe_task():
    """BeyondSWETask with search_mode uses search prompt keys."""
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES, search_mode=True)
    instances = task.get_instances(instance_ids=["django_crossrepo_002"])
    prompt = task.get_prompt(instances[0])
    # Search variant includes search-specific phases
    assert "Search Tool" in prompt or "search" in prompt.lower()
    assert "Import fails" in prompt


# ── test_suite_dir support ────────────────────────────────────────────────────

def test_test_suite_dir_from_constructor():
    """test_suite_dir passed to constructor populates metadata."""
    task = BeyondSWETask(
        instances=_BEYOND_SWE_INSTANCES,
        test_suite_dir="/data/test_suites",
    )
    inst = task.get_instances(instance_ids=["mylib_doc2repo_001"])[0]
    assert inst.metadata["test_suite_path"] == "/data/test_suites"


def test_test_suite_dir_from_env(monkeypatch):
    """BEYONDSWE_TEST_SUITE_DIR env var is used as fallback."""
    monkeypatch.setenv("BEYONDSWE_TEST_SUITE_DIR", "/env/test_suites")
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    inst = task.get_instances(instance_ids=["mylib_doc2repo_001"])[0]
    assert inst.metadata["test_suite_path"] == "/env/test_suites"


def test_test_suite_dir_constructor_overrides_env(monkeypatch):
    """Constructor argument takes priority over env var."""
    monkeypatch.setenv("BEYONDSWE_TEST_SUITE_DIR", "/env/path")
    task = BeyondSWETask(
        instances=_BEYOND_SWE_INSTANCES,
        test_suite_dir="/constructor/path",
    )
    inst = task.get_instances(instance_ids=["mylib_doc2repo_001"])[0]
    assert inst.metadata["test_suite_path"] == "/constructor/path"
