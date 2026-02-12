"""Tests for task implementations (SWE-Bench and BeyondSWE)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from awe_agent.core.task.types import Instance
from awe_agent.tasks.beyond_swe.prompts import get_beyond_swe_prompt
from awe_agent.tasks.beyond_swe.task import BeyondSWETask
from awe_agent.tasks.swe_bench.prompts import get_swe_bench_prompt
from awe_agent.tasks.swe_bench.task import SWEBenchTask


# ── SWE-Bench ────────────────────────────────────────────────────────────────

_SWE_INSTANCES = [
    {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "ValueError when calling QuerySet.filter() with None",
        "patch": "diff --git a/django/db/models/query.py ...",
        "FAIL_TO_PASS": '["test_filter_none"]',
        "PASS_TO_PASS": '["test_filter_basic"]',
        "version": "4.2",
        "language": "python",
    },
    {
        "instance_id": "requests__requests-6789",
        "repo": "psf/requests",
        "base_commit": "def456",
        "problem_statement": "Timeout not respected for streaming responses",
        "patch": "diff --git a/requests/adapters.py ...",
        "language": "python",
    },
]


def _write_jsonl(data: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def test_swe_bench_task_from_instances():
    task = SWEBenchTask(instances=_SWE_INSTANCES)
    instances = task.get_instances()
    assert len(instances) == 2
    assert instances[0].id == "django__django-12345"
    assert instances[0].repo == "django/django"
    assert instances[0].base_commit == "abc123"


def test_swe_bench_task_from_jsonl():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in _SWE_INSTANCES:
            f.write(json.dumps(item) + "\n")
        f.flush()
        task = SWEBenchTask(data_file=f.name)
        instances = task.get_instances()
        assert len(instances) == 2


def test_swe_bench_task_filter_by_id():
    task = SWEBenchTask(instances=_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["django__django-12345"])
    assert len(instances) == 1
    assert instances[0].id == "django__django-12345"


def test_swe_bench_task_prompt():
    task = SWEBenchTask(instances=_SWE_INSTANCES)
    instances = task.get_instances()
    prompt = task.get_prompt(instances[0])
    assert "ValueError" in prompt
    assert "QuerySet.filter()" in prompt
    assert "/testbed" in prompt


def test_swe_bench_task_setup_commands():
    task = SWEBenchTask(instances=_SWE_INSTANCES)
    instances = task.get_instances()
    commands = task.get_setup_commands(instances[0])
    assert any("git checkout abc123" in cmd for cmd in commands)


def test_swe_bench_task_info():
    task = SWEBenchTask(instances=_SWE_INSTANCES, task_type="issue_resolving")
    instances = task.get_instances()
    info = task.get_task_info(instances[0])
    assert info["instance_id"] == "django__django-12345"
    assert info["task_type"] == "issue_resolving"


def test_swe_bench_prompt():
    """SWE-bench prompt includes problem statement and workspace dir."""
    prompt = get_swe_bench_prompt(
        problem_statement="Bug in parser",
        workspace_dir="/testbed",
        language="python",
    )
    assert "Bug in parser" in prompt
    assert "/testbed" in prompt
    assert "READING" in prompt  # Workflow phase


def test_swe_bench_no_data_source():
    task = SWEBenchTask()
    with pytest.raises(ValueError, match="No data source"):
        task.get_instances()


def test_swe_bench_missing_jsonl():
    task = SWEBenchTask(data_file="/nonexistent/file.jsonl")
    with pytest.raises(FileNotFoundError):
        task.get_instances()


# ── BeyondSWE ────────────────────────────────────────────────────────────────

_BEYOND_SWE_INSTANCES = [
    {
        "instance_id": "mylib_doc2repo_001",
        "task": "doc2repo",
        "workdir": "/workspace",
        "image_url": "ubuntu:22.04",
        "REPO_DOCUMENT_CONTENT": "# MyLib\n\nA library for data processing...",
        "base_commit": "aaa111",
        "language": "python",
    },
    {
        "instance_id": "django_crossrepo_002",
        "task": "cross-repo",
        "workdir": "/workspace",
        "image_url": "python:3.11",
        "problem_statement": "Import fails across modules after rename",
        "base_commit": "bbb222",
        "parent_commit": "bbb221",
        "FAIL_TO_PASS": '["test_import"]',
        "language": "python",
    },
    {
        "instance_id": "flask_refactor_003",
        "task": "refactor",
        "workdir": "/workspace",
        "image_url": "python:3.11",
        "problem_statement": "Refactor request handling to use async",
        "base_commit": "ccc333",
        "f2p_patch": "diff --git ...",
        "language": "python",
    },
    {
        "instance_id": "scipy_domain_004",
        "task": "domain",
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
    assert "cross-repo" in types
    assert "refactor" in types
    assert "domain" in types


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


def test_beyond_swe_refactor_prompt():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["flask_refactor_003"])
    prompt = task.get_prompt(instances[0])
    assert "Refactor" in prompt
    assert "async" in prompt.lower()


def test_beyond_swe_domain_prompt():
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES)
    instances = task.get_instances(instance_ids=["scipy_domain_004"])
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

def test_prompt_routing_swe_bench():
    """SWE-bench routes resolve correctly for both search modes."""
    from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys

    sys_key, usr_key = resolve_prompt_keys("swe_bench", None, False)
    assert sys_key == "base"
    assert usr_key == "swe_bench"

    sys_key, usr_key = resolve_prompt_keys("swe_bench", None, True)
    assert sys_key == "search"
    assert usr_key == "swe_bench"


def test_prompt_routing_beyond_swe():
    """BeyondSWE routes resolve correctly for all task types."""
    from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys

    sys_key, usr_key = resolve_prompt_keys("beyond_swe", "doc2repo", False)
    assert sys_key == "base"
    assert usr_key == "doc2repo"

    sys_key, usr_key = resolve_prompt_keys("beyond_swe", "domain", True)
    assert sys_key == "search_domain"
    assert usr_key == "search_domain"


def test_prompt_routing_fallback():
    """Unknown dataset falls back to default route."""
    from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys

    sys_key, usr_key = resolve_prompt_keys("unknown_dataset", None, False)
    assert sys_key == "base"
    assert usr_key == "swe_bench"


def test_search_mode_beyond_swe_task():
    """BeyondSWETask with search_mode uses search prompt keys."""
    task = BeyondSWETask(instances=_BEYOND_SWE_INSTANCES, search_mode=True)
    instances = task.get_instances(instance_ids=["django_crossrepo_002"])
    prompt = task.get_prompt(instances[0])
    # Search variant includes search-specific phases
    assert "Search Tool" in prompt or "search" in prompt.lower()
    assert "Import fails" in prompt


def test_search_mode_swe_bench_task():
    """SWEBenchTask with search_mode still resolves correctly."""
    task = SWEBenchTask(instances=_SWE_INSTANCES, search_mode=True)
    instances = task.get_instances()
    prompt = task.get_prompt(instances[0])
    assert "ValueError" in prompt
