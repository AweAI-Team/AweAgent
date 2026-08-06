"""Explicit system-prompt and skill loading from files.

The route table (``scaffold/search_swe/prompts/config.py``) maps a task's
identity to a built-in prompt key. That is right for fixed-harness eval, but
the rollout server needs to *vary* the prompt to search for better ones. This
module is the file-based alternative: point ``agent.system_prompt_file`` and
``agent.skill_files`` at plain-text docs and the scaffold uses them verbatim
instead of the route table.

Design (agreed with the framework owner):

* **System prompt** — a plain-text file, used as-is (no frontmatter, no
  formatting). It replaces only the *base* system text; the scaffold still
  appends tool descriptions for text-based tool-call formats (XML/CodeAct),
  because that suffix is the model's only channel to learn the tools.
* **Skill** — a static solving-technique doc appended after the system prompt.
  It is *not* loaded at runtime the way Claude Code loads skills (not every
  harness has such a mechanism); it is inlined into the system prompt as a
  reminder of what to watch for. Each skill file declares its own name via a
  YAML frontmatter block::

      ---
      name: swe_debug
      ---
      <skill body ...>

  and the framework wraps the body as ``<skill name="swe_debug">…</skill>``.
  Order is **system prompt → skills → tool suffix**.

Everything here is fail-fast: a missing file or a skill without a ``name``
raises rather than silently falling back. Silent fallbacks are exactly what
this refactor removes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from aweagent.core.config.schema import AweAgentConfig

__all__ = [
    "Skill",
    "load_system_prompt_file",
    "load_skill_file",
    "load_skills",
    "wrap_skills",
    "parse_skill_frontmatter",
    "load_prompt_overrides",
]

# A skill file must start with a YAML frontmatter block delimited by ``---``.
_FRONTMATTER_DELIM = "---"


@dataclass(frozen=True)
class Skill:
    """A parsed skill: its declared name and its body text."""

    name: str
    body: str


def load_system_prompt_file(path: str | Path) -> str:
    """Read a plain-text system prompt file and return its contents.

    Raises FileNotFoundError if the file is missing and ValueError if it is
    empty (an empty system prompt is almost always a mistake — and an empty
    string means "no system message" to the loop, silently changing behavior).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"system_prompt_file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"system_prompt_file is empty: {p}")
    return text


def parse_skill_frontmatter(text: str, *, source: str = "<string>") -> Skill:
    """Parse a skill doc: a required YAML frontmatter (declaring ``name``)
    followed by the body.

    The frontmatter is a leading block delimited by lines containing exactly
    ``---``. Only ``name`` is required today; extra keys (e.g. a future
    ``description`` / ``when_to_use``) are accepted and ignored, leaving room
    to grow without a format change.
    """
    stripped = text.lstrip("﻿")  # tolerate a UTF-8 BOM
    lines = stripped.splitlines()

    # The opening delimiter must be the first non-empty line and contain
    # exactly '---' — not merely start with it, so '----' or '--- yaml' are
    # rejected rather than silently accepted as a frontmatter fence.
    start = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if start is None or lines[start].strip() != _FRONTMATTER_DELIM:
        first = lines[start] if start is not None else ""
        raise ValueError(
            f"skill file {source} must start with a YAML frontmatter block "
            f"(a line containing only '---'); got: {first[:40]!r}"
        )

    # Find the closing delimiter (a line containing exactly '---').
    end = None
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == _FRONTMATTER_DELIM:
            end = i
            break
    if end is None:
        raise ValueError(
            f"skill file {source} has an unterminated frontmatter block "
            f"(missing the closing '---')"
        )

    fm_text = "\n".join(lines[start + 1 : end])
    body = "\n".join(lines[end + 1 :]).strip()

    try:
        meta = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"skill file {source} has invalid YAML frontmatter: {e}") from e
    if not isinstance(meta, dict):
        raise ValueError(
            f"skill file {source} frontmatter must be a YAML mapping, got {type(meta).__name__}"
        )

    name = meta.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"skill file {source} frontmatter must declare a non-empty string 'name'"
        )
    name = name.strip()
    # The name is interpolated into ``<skill name="...">`` verbatim (no
    # escaping), so reject characters that would break the tag rather than
    # silently emit malformed markup.
    if any(c in name for c in '"<>\n\r'):
        raise ValueError(
            f"skill file {source} has an invalid 'name' {name!r}: "
            f'must not contain quotes, angle brackets, or newlines'
        )
    if not body:
        raise ValueError(f"skill file {source} has no body after the frontmatter")

    return Skill(name=name, body=body)


def load_skill_file(path: str | Path) -> Skill:
    """Read and parse one skill file into a :class:`Skill`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"skill file not found: {p}")
    return parse_skill_frontmatter(p.read_text(encoding="utf-8"), source=str(p))


def load_skills(paths: list[str] | list[Path]) -> list[Skill]:
    """Load several skill files, preserving order and rejecting duplicate names."""
    skills: list[Skill] = []
    seen: dict[str, str] = {}
    for path in paths:
        skill = load_skill_file(path)
        if skill.name in seen:
            raise ValueError(
                f"duplicate skill name {skill.name!r}: {seen[skill.name]} and {path}"
            )
        seen[skill.name] = str(path)
        skills.append(skill)
    return skills


def wrap_skills(skills: list[Skill]) -> str:
    """Render skills as ``<skill name="...">body</skill>`` blocks joined by
    blank lines. Returns ``""`` for an empty list."""
    return "\n\n".join(
        f'<skill name="{s.name}">\n{s.body}\n</skill>' for s in skills
    )


def load_prompt_overrides(config: AweAgentConfig) -> tuple[str | None, str]:
    """Resolve explicit system-prompt / skill overrides from config.

    Returns ``(system_prompt_override, skill_text)``:

    * ``system_prompt_override`` — the contents of ``agent.system_prompt_file``
      if set, else ``None`` (meaning "use the scaffold's built-in prompt").
    * ``skill_text`` — the ``agent.skill_files`` wrapped as ``<skill>`` blocks,
      or ``""`` if none.

    Files are read here (at agent construction) so a bad path or a skill
    missing its ``name`` fails fast rather than mid-rollout. Shared by every
    scaffold that supports config-driven prompts (search_swe, calibforge, …)
    so the loading semantics stay identical across them.
    """
    system_prompt_override: str | None = None
    if config.agent.system_prompt_file:
        system_prompt_override = load_system_prompt_file(config.agent.system_prompt_file)
    skill_text = ""
    if config.agent.skill_files:
        skill_text = wrap_skills(load_skills(config.agent.skill_files))
    return system_prompt_override, skill_text

