"""Tests for CalibForge explicit system-prompt / skill overrides (config-driven).

CalibForge is a native function-calling scaffold; unlike search_swe it has no
route table — the base system prompt is the built-in SYSTEM_PROMPT, optionally
replaced by ``agent.system_prompt_file`` and extended by ``agent.skill_files``.
"""

from __future__ import annotations

import pytest

from aweagent.core.config.schema import AweAgentConfig
from aweagent.scaffold.calibforge.agent import CalibForgeAgent
from aweagent.scaffold.calibforge.prompts import SYSTEM_PROMPT


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _config(**agent_kwargs) -> AweAgentConfig:
    cfg = AweAgentConfig()
    for k, v in agent_kwargs.items():
        setattr(cfg.agent, k, v)
    return cfg


# ── Default: built-in SYSTEM_PROMPT ─────────────────────────────────────────────

def test_default_uses_builtin_system_prompt():
    agent = CalibForgeAgent()
    assert agent.get_system_prompt({}) == SYSTEM_PROMPT


def test_from_config_default_is_builtin():
    agent = CalibForgeAgent.from_config(_config())
    assert agent.get_system_prompt({}) == SYSTEM_PROMPT


# ── Explicit system prompt override ─────────────────────────────────────────────

def test_system_prompt_override_replaces_builtin():
    agent = CalibForgeAgent(system_prompt_override="CUSTOM CF SP")
    assert agent.get_system_prompt({}) == "CUSTOM CF SP"


def test_from_config_loads_system_prompt_file(tmp_path):
    sp = _write(tmp_path, "sp.txt", "MY CALIBFORGE SP")
    agent = CalibForgeAgent.from_config(_config(system_prompt_file=sp))
    assert agent.get_system_prompt({}).startswith("MY CALIBFORGE SP")
    assert SYSTEM_PROMPT not in agent.get_system_prompt({})


def test_from_config_missing_system_prompt_file_raises(tmp_path):
    cfg = _config(system_prompt_file=str(tmp_path / "nope.txt"))
    with pytest.raises(FileNotFoundError):
        CalibForgeAgent.from_config(cfg)


# ── Skills ──────────────────────────────────────────────────────────────────────

def test_skill_appended_after_builtin():
    agent = CalibForgeAgent(skill_text='<skill name="s1">\ncare\n</skill>')
    prompt = agent.get_system_prompt({})
    assert prompt.startswith(SYSTEM_PROMPT)
    assert prompt.index(SYSTEM_PROMPT) < prompt.index('<skill name="s1">')


def test_from_config_loads_skill_files(tmp_path):
    s1 = _write(tmp_path, "s1.md", "---\nname: tb_debug\n---\nCheck the verifier logs.")
    agent = CalibForgeAgent.from_config(_config(skill_files=[s1]))
    prompt = agent.get_system_prompt({})
    assert prompt.startswith(SYSTEM_PROMPT)
    assert '<skill name="tb_debug">' in prompt
    assert "Check the verifier logs." in prompt


def test_override_plus_skill(tmp_path):
    sp = _write(tmp_path, "sp.txt", "BASE")
    s1 = _write(tmp_path, "s1.md", "---\nname: tip\n---\nA tip.")
    agent = CalibForgeAgent.from_config(_config(system_prompt_file=sp, skill_files=[s1]))
    prompt = agent.get_system_prompt({})
    assert prompt.startswith("BASE")
    assert '<skill name="tip">' in prompt
    assert SYSTEM_PROMPT not in prompt


# ── tool_call_format guard still enforced ───────────────────────────────────────

def test_from_config_rejects_non_openai_format():
    cfg = _config(tool_call_format="codeact_xml")
    with pytest.raises(ValueError, match="openai_function"):
        CalibForgeAgent.from_config(cfg)
