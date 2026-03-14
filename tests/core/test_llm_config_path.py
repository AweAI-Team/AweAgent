"""Test --llm-config path normalization in recipe runners.

Verifies that --llm-config paths (relative to CWD) are correctly converted
to paths relative to the task config directory for !include resolution.
"""

from __future__ import annotations

import os
from pathlib import Path


def test_llm_config_relpath_from_cwd(tmp_path):
    """--llm-config configs/llm/anthropic.yaml resolves correctly."""
    # Set up directory structure mimicking the real repo
    task_dir = tmp_path / "configs" / "tasks"
    task_dir.mkdir(parents=True)
    llm_dir = tmp_path / "configs" / "llm"
    llm_dir.mkdir(parents=True)
    llm_file = llm_dir / "anthropic.yaml"
    llm_file.write_text("backend: anthropic\nmodel: claude-sonnet-4-6\n")

    # Simulate what run.py does: resolve CWD-relative path to task-dir-relative
    llm_config_arg = str(llm_file)  # absolute (as Path.resolve() would produce)
    llm_abs = Path(llm_config_arg).resolve()
    rel = os.path.relpath(llm_abs, task_dir)

    # Should be ../llm/anthropic.yaml, not configs/llm/anthropic.yaml
    assert rel == os.path.join("..", "llm", "anthropic.yaml")
    # And it should resolve back to the original file
    assert (task_dir / rel).resolve() == llm_abs


def test_llm_config_example_path(tmp_path):
    """--llm-config configs/llm/examples/kimi.yaml resolves correctly."""
    task_dir = tmp_path / "configs" / "tasks"
    task_dir.mkdir(parents=True)
    examples_dir = tmp_path / "configs" / "llm" / "examples"
    examples_dir.mkdir(parents=True)
    llm_file = examples_dir / "kimi.yaml"
    llm_file.write_text("backend: openai\nmodel: kimi-k2.5\n")

    llm_abs = llm_file.resolve()
    rel = os.path.relpath(llm_abs, task_dir)

    assert rel == os.path.join("..", "llm", "examples", "kimi.yaml")
    assert (task_dir / rel).resolve() == llm_abs


def test_llm_config_already_task_relative(tmp_path):
    """Path that's already inside configs/tasks/ stays simple."""
    task_dir = tmp_path / "configs" / "tasks"
    task_dir.mkdir(parents=True)
    llm_file = task_dir / "custom_llm.yaml"
    llm_file.write_text("backend: openai\n")

    llm_abs = llm_file.resolve()
    rel = os.path.relpath(llm_abs, task_dir)

    assert rel == "custom_llm.yaml"
    assert (task_dir / rel).resolve() == llm_abs
