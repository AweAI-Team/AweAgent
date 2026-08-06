"""Tests for configuration loading and schema."""

from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from aweagent.core.config.loader import load_config
from aweagent.core.config.schema import (
    AgentConfig,
    AweAgentConfig,
    ExecutionConfig,
    SecurityConfig,
    TaskConfig,
)
from aweagent.core.llm.config import LLMConfig
from aweagent.core.runtime.config import RuntimeConfig


def test_default_config():
    """Default config should be valid."""
    config = AweAgentConfig()
    assert config.llm.backend == "openai"
    assert config.runtime.backend == "docker"
    assert config.agent.type == "search_swe"
    assert config.agent.max_steps == 100
    assert config.task.type == "beyond_swe"


def test_llm_config_fields():
    cfg = LLMConfig(
        backend="ark",
        model="deepseek-r1",
        thinking=True,
        thinking_budget=10000,
        stop=["<END>"],
    )
    assert cfg.thinking is True
    assert cfg.thinking_budget == 10000
    assert cfg.stop == ["<END>"]


def test_runtime_config_defaults():
    cfg = RuntimeConfig()
    assert cfg.backend == "docker"
    assert cfg.timeout == 14400
    assert cfg.workdir == "/testbed"
    assert cfg.resource_limits.cpu == "4"
    assert cfg.resource_limits.memory == "8Gi"


def test_security_config_blocklist():
    cfg = SecurityConfig()
    # Default bash_blocklist is empty — core patterns live in agent.py
    assert cfg.bash_blocklist == []


def test_load_config_from_yaml():
    """Test YAML config loading."""
    yaml_content = {
        "llm": {"backend": "azure", "model": "gpt-4"},
        "runtime": {"backend": "docker", "timeout": 3600},
        "agent": {"max_steps": 50},
        "execution": {"start_index": 2, "end_index": 5, "max_instances": 10},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        f.flush()
        try:
            config = load_config(f.name)
            assert config.llm.backend == "azure"
            assert config.llm.model == "gpt-4"
            assert config.runtime.timeout == 3600
            assert config.agent.max_steps == 50
            assert config.execution.start_index == 2
            assert config.execution.end_index == 5
            assert config.execution.max_instances == 10
        finally:
            os.unlink(f.name)


def test_execution_max_instances_must_be_positive():
    """A configured instance cap must be positive when provided."""
    with pytest.raises(ValueError):
        ExecutionConfig(max_instances=0)


def test_execution_num_rollouts_default_and_validation():
    """num_rollouts defaults to 1 and must be a positive int."""
    assert ExecutionConfig().num_rollouts == 1
    assert ExecutionConfig(num_rollouts=3).num_rollouts == 3
    with pytest.raises(ValueError):
        ExecutionConfig(num_rollouts=0)


def test_execution_instance_range_must_be_non_negative():
    """Configured instance range indices must be non-negative."""
    with pytest.raises(ValueError):
        ExecutionConfig(start_index=-1)
    with pytest.raises(ValueError):
        ExecutionConfig(end_index=-1)


def test_execution_end_index_must_not_precede_start_index():
    """The inclusive end index must not be before the start index."""
    with pytest.raises(ValueError):
        ExecutionConfig(start_index=3, end_index=2)


def test_load_config_env_override():
    """Test env var override of config values."""
    yaml_content = {"llm": {"backend": "openai", "model": "gpt-3.5"}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(yaml_content, f)
        f.flush()
        try:
            os.environ["AWE_AGENT__LLM__MODEL"] = "gpt-4o"
            config = load_config(f.name)
            assert config.llm.model == "gpt-4o"
        finally:
            os.environ.pop("AWE_AGENT__LLM__MODEL", None)
            os.unlink(f.name)


def test_agent_config_tools():
    cfg = AgentConfig(tools=["bash", "editor"])
    assert "bash" in cfg.tools
    assert "think" not in cfg.tools


def test_task_config():
    cfg = TaskConfig(
        type="beyond_swe",
        dataset_id="beyond_swe_bench",
        data_file="/path/to/data.jsonl",
    )
    assert cfg.type == "beyond_swe"
    assert cfg.data_file == "/path/to/data.jsonl"
