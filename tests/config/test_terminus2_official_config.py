from __future__ import annotations

from pathlib import Path

import pytest

from aweagent.core.config.loader import load_config


def test_official_terminus_config_enables_explicit_context_management() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/tasks/terminal_bench_v2_official.yaml")

    assert config.agent.type == "terminus_2"
    assert config.agent.max_context_length == 131_072
    assert config.agent.condenser.type == "terminus_2"
    assert config.agent.condenser.proactive_threshold == 8_000
    assert config.agent.condenser.recovery_target_free_tokens == 4_000
    assert config.runtime.backend == "docker"
    assert config.task.task_data_dir == "datasets/terminal_bench_v2/tasks"

def test_official_config_contains_no_internal_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    config_text = (
        root / "configs/tasks/terminal_bench_v2_official.yaml"
    ).read_text()

    assert "/mnt/bn/" not in config_text
    assert "SANDBOX_PSM" not in config_text
    assert "portal" not in config_text


def test_official_config_accepts_local_tokenizer_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TERMINUS2_TOKENIZER_PATH", "/models/local-qwen")
    root = Path(__file__).resolve().parents[2]

    config = load_config(root / "configs/tasks/terminal_bench_v2_official.yaml")

    assert config.agent.condenser.tokenizer_path == "/models/local-qwen"
