"""Tests for SearchSWEAgent explicit system-prompt / skill overrides.

Covers the rollout-server prompt path: config points at a system-prompt file
and/or skill files, and the agent uses them verbatim instead of the route
table — while still appending the tool suffix for text-based formats.
"""

from __future__ import annotations

import pytest

from aweagent.core.config.schema import AweAgentConfig
from aweagent.scaffold.search_swe.agent import SearchSWEAgent


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ── Route-table default (no override) ──────────────────────────────────────────

def test_no_override_uses_route_table():
    """Without a system_prompt_file, the route table drives the system prompt."""
    agent = SearchSWEAgent(enable_search=False)
    task_info = {"dataset_id": "scale_swe", "task_type": None}
    prompt = agent.get_system_prompt(task_info)
    assert prompt  # resolved from route table (openhands)
    assert agent._system_prompt_override is None


def test_no_route_and_no_override_raises():
    """An unknown dataset with no override raises (fail-fast, no silent default)."""
    agent = SearchSWEAgent(enable_search=False)
    with pytest.raises(KeyError, match="No prompt route"):
        agent.get_system_prompt({"dataset_id": "mystery", "task_type": None})


# ── Explicit system prompt override ─────────────────────────────────────────────

def test_system_prompt_override_replaces_base():
    agent = SearchSWEAgent(system_prompt_override="CUSTOM SP")
    # Even an unroutable task_info works — override bypasses the route table.
    prompt = agent.get_system_prompt({"dataset_id": "mystery", "task_type": None})
    assert prompt.startswith("CUSTOM SP")


def test_from_config_loads_system_prompt_file(tmp_path):
    sp = _write(tmp_path, "sp.txt", "MY SYSTEM PROMPT")
    cfg = AweAgentConfig()
    cfg.agent.system_prompt_file = sp
    agent = SearchSWEAgent.from_config(cfg)
    prompt = agent.get_system_prompt({"dataset_id": "mystery", "task_type": None})
    assert prompt.startswith("MY SYSTEM PROMPT")


def test_from_config_missing_system_prompt_file_raises(tmp_path):
    cfg = AweAgentConfig()
    cfg.agent.system_prompt_file = str(tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError):
        SearchSWEAgent.from_config(cfg)


# ── Skills ──────────────────────────────────────────────────────────────────────

def test_skill_appended_after_system_prompt():
    agent = SearchSWEAgent(
        system_prompt_override="BASE",
        skill_text='<skill name="s1">\nbe careful\n</skill>',
    )
    prompt = agent.get_system_prompt({"dataset_id": "mystery", "task_type": None})
    assert prompt.index("BASE") < prompt.index('<skill name="s1">')
    assert "be careful" in prompt


def test_from_config_loads_skill_files(tmp_path):
    sp = _write(tmp_path, "sp.txt", "BASE SP")
    s1 = _write(tmp_path, "s1.md", "---\nname: debug\n---\nRead the failing test.")
    cfg = AweAgentConfig()
    cfg.agent.system_prompt_file = sp
    cfg.agent.skill_files = [s1]
    agent = SearchSWEAgent.from_config(cfg)
    prompt = agent.get_system_prompt({"dataset_id": "mystery", "task_type": None})
    assert "BASE SP" in prompt
    assert '<skill name="debug">' in prompt
    assert "Read the failing test." in prompt


def test_skills_work_over_route_table_base(tmp_path):
    """Skills can layer on top of a route-table base prompt (no SP override)."""
    s1 = _write(tmp_path, "s1.md", "---\nname: tip\n---\nA tip.")
    cfg = AweAgentConfig()
    cfg.agent.skill_files = [s1]
    agent = SearchSWEAgent.from_config(cfg)
    prompt = agent.get_system_prompt({"dataset_id": "scale_swe", "task_type": None})
    assert '<skill name="tip">' in prompt


# ── Tool suffix preserved (XML format) ─────────────────────────────────────────

def test_tool_suffix_appended_with_override_xml():
    """Override must NOT drop the tool suffix in text-based formats — it is the
    model's only channel to learn the tools."""
    agent = SearchSWEAgent(
        system_prompt_override="BASE",
        tool_call_format="codeact_xml",
    )
    prompt = agent.get_system_prompt({"dataset_id": "mystery", "task_type": None})
    assert prompt.startswith("BASE")
    # The XML format appends a non-empty tool description suffix.
    assert len(prompt) > len("BASE")
    # order: base → (no skill) → suffix; suffix mentions tools/functions
    assert "execute_bash" in prompt or "function" in prompt.lower()
