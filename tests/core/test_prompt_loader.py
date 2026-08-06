"""Tests for the file-based prompt loader (system prompt + skills)."""

from __future__ import annotations

import pytest

from aweagent.core.agent.prompt_loader import (
    Skill,
    load_skill_file,
    load_skills,
    load_system_prompt_file,
    parse_skill_frontmatter,
    wrap_skills,
)


# ── System prompt file ────────────────────────────────────────────────────────

def test_load_system_prompt_file(tmp_path):
    p = tmp_path / "sp.txt"
    p.write_text("You are a coding agent.\nBe careful.")
    assert load_system_prompt_file(p) == "You are a coding agent.\nBe careful."


def test_load_system_prompt_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="system_prompt_file not found"):
        load_system_prompt_file(tmp_path / "nope.txt")


def test_load_system_prompt_empty_raises(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n  ")
    with pytest.raises(ValueError, match="empty"):
        load_system_prompt_file(p)


# ── Skill frontmatter parsing ─────────────────────────────────────────────────

def test_parse_skill_frontmatter_ok():
    text = "---\nname: swe_debug\n---\nAlways read the failing test first."
    skill = parse_skill_frontmatter(text)
    assert skill == Skill(name="swe_debug", body="Always read the failing test first.")


def test_parse_skill_frontmatter_extra_keys_ignored():
    text = "---\nname: s1\ndescription: future field\nwhen_to_use: later\n---\nBody."
    skill = parse_skill_frontmatter(text)
    assert skill.name == "s1"
    assert skill.body == "Body."


def test_parse_skill_tolerates_bom_and_blank_lines():
    text = "﻿\n---\nname: s2\n---\n\nBody text.\n"
    skill = parse_skill_frontmatter(text)
    assert skill.name == "s2"
    assert skill.body == "Body text."


def test_parse_skill_no_frontmatter_raises():
    with pytest.raises(ValueError, match="must start with a YAML frontmatter"):
        parse_skill_frontmatter("name: s\nbody without fence")


def test_parse_skill_loose_opening_delimiter_rejected():
    """An opening line that merely starts with '---' (e.g. '----' or
    '--- yaml') is not a valid fence — reject it."""
    with pytest.raises(ValueError, match="must start with a YAML frontmatter"):
        parse_skill_frontmatter("----\nname: s\n----\nBody.")
    with pytest.raises(ValueError, match="must start with a YAML frontmatter"):
        parse_skill_frontmatter("--- yaml\nname: s\n---\nBody.")


def test_parse_skill_name_with_unsafe_chars_raises():
    """A name containing quotes / angle brackets / newlines would break the
    ``<skill name="...">`` tag — reject it at parse time."""
    for bad in ('a"b', "a<b", "a>b"):
        with pytest.raises(ValueError, match="invalid 'name'"):
            parse_skill_frontmatter(f"---\nname: '{bad}'\n---\nBody.")


def test_parse_skill_unterminated_frontmatter_raises():
    with pytest.raises(ValueError, match="unterminated frontmatter"):
        parse_skill_frontmatter("---\nname: s\nbody with no closing fence")


def test_parse_skill_missing_name_raises():
    with pytest.raises(ValueError, match="must declare a non-empty string 'name'"):
        parse_skill_frontmatter("---\ndescription: x\n---\nBody.")


def test_parse_skill_empty_body_raises():
    with pytest.raises(ValueError, match="no body"):
        parse_skill_frontmatter("---\nname: s\n---\n   ")


def test_parse_skill_non_mapping_frontmatter_raises():
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        parse_skill_frontmatter("---\n- just\n- a\n- list\n---\nBody.")


# ── Skill file loading ────────────────────────────────────────────────────────

def test_load_skill_file(tmp_path):
    p = tmp_path / "skill.md"
    p.write_text("---\nname: my_skill\n---\nDo the thing.")
    skill = load_skill_file(p)
    assert skill == Skill(name="my_skill", body="Do the thing.")


def test_load_skill_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="skill file not found"):
        load_skill_file(tmp_path / "nope.md")


def test_load_skills_preserves_order(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("---\nname: a\n---\nAA")
    b = tmp_path / "b.md"
    b.write_text("---\nname: b\n---\nBB")
    skills = load_skills([b, a])  # order = argument order, not sorted
    assert [s.name for s in skills] == ["b", "a"]


def test_load_skills_duplicate_name_raises(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("---\nname: dup\n---\nAA")
    b = tmp_path / "b.md"
    b.write_text("---\nname: dup\n---\nBB")
    with pytest.raises(ValueError, match="duplicate skill name"):
        load_skills([a, b])


# ── Wrapping ──────────────────────────────────────────────────────────────────

def test_wrap_skills():
    skills = [Skill(name="s1", body="body one"), Skill(name="s2", body="body two")]
    out = wrap_skills(skills)
    assert '<skill name="s1">\nbody one\n</skill>' in out
    assert '<skill name="s2">\nbody two\n</skill>' in out
    # s1 appears before s2 (order preserved)
    assert out.index("s1") < out.index("s2")


def test_wrap_skills_empty():
    assert wrap_skills([]) == ""
