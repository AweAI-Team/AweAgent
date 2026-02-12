"""Tests for configuration loading and schema."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from awe_agent.core.config.loader import load_config
from awe_agent.core.config.schema import (
    AgentConfig,
    AweAgentConfig,
    EvalConfig,
    ExecutionConfig,
    SecurityConfig,
    TaskConfig,
)
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.runtime.config import RuntimeConfig


def test_default_config():
    """Default config should be valid."""
    config = AweAgentConfig()
    assert config.llm.backend == "openai"
    assert config.runtime.backend == "docker"
    assert config.agent.type == "search_swe"
    assert config.agent.max_steps == 100
    assert config.task.type == "swe_bench"


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
    assert cfg.timeout == 7200
    assert cfg.workdir == "/testbed"
    assert cfg.resource_limits.cpu == "4"
    assert cfg.resource_limits.memory == "8Gi"


def test_security_config_blacklist():
    cfg = SecurityConfig()
    assert len(cfg.bash_blacklist) > 0
    assert any("git clone" in p for p in cfg.bash_blacklist)


def test_load_config_from_yaml():
    """Test YAML config loading."""
    yaml_content = {
        "llm": {"backend": "azure", "model": "gpt-4"},
        "runtime": {"backend": "docker", "timeout": 3600},
        "agent": {"max_steps": 50},
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
        finally:
            os.unlink(f.name)


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
