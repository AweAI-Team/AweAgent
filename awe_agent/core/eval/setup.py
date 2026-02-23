"""PreAgentSetup — prepare the container environment before agent execution.

Encapsulates:

- Running instance setup commands (from ``pre_commands`` / ``setup_commands``)
- Removing future git commits (prevent data leakage)

Usage::

    setup = PreAgentSetup(session, instance.workdir)
    await setup.prepare(instance)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from awe_agent.core.task.types import Instance

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)

# Shell snippet that resets all git refs to HEAD and clears the stash.
# Prevents the agent from seeing future commits (data leakage).
_REMOVE_FUTURE_COMMITS = (
    'current_branch=$(git rev-parse --abbrev-ref HEAD) && '
    'git for-each-ref --format="%(refname)" | while read ref; do '
    'if [[ "$ref" == refs/heads/* ]]; then '
    'branch_name="${ref#refs/heads/}"; '
    'if [[ "$branch_name" != "$current_branch" ]]; then '
    'git branch -f "$branch_name" HEAD; '
    'fi; '
    'else git update-ref "$ref" HEAD 2>/dev/null || true; fi; done && '
    'git stash clear 2>/dev/null || true && '
    'git reflog expire --expire=now --all 2>/dev/null || true && '
    'git gc --prune=now 2>/dev/null || true'
)


class PreAgentSetup:
    """Prepare the container environment before agent execution or evaluation.

    Encapsulates:

    - Running instance setup commands (from ``pre_commands``)
    - Removing future git commits (prevent data leakage)
    """

    def __init__(self, session: RuntimeSession, workdir: str) -> None:
        self.session = session
        self.workdir = workdir

    async def run_setup_commands(self, commands: list[str]) -> None:
        """Execute setup commands (from ``instance.setup_commands``)."""
        for cmd in commands:
            result = await self.session.execute(cmd, timeout=300)
            if not result.success:
                logger.warning(
                    "Setup command failed (exit %d): %s -> %s",
                    result.exit_code,
                    cmd[:120],
                    result.stderr[:200],
                )

    async def remove_future_commits(self) -> None:
        """Reset all git refs to HEAD, clear stash.  Prevents data leakage."""
        await self.session.execute(
            f"cd {self.workdir} && {_REMOVE_FUTURE_COMMITS}",
        )

    async def prepare(self, instance: Instance) -> None:
        """Full pre-agent preparation: setup commands + remove future commits."""
        await self.run_setup_commands(instance.setup_commands)
        await self.remove_future_commits()
