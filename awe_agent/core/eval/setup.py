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

# Shell snippet that commits all current changes as a "pre-agent" snapshot.
_GIT_COMMIT_PRE_AGENT = (
    'git config user.email "pre-agent@awe-agent.local" && '
    'git config user.name "Pre-Agent" && '
    'git add -A && '
    'git commit -m "pre-agent commit"'
)

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

    async def commit_and_get_id(self) -> str | None:
        """Commit current state, return HEAD SHA."""
        await self.session.execute(
            f"cd {self.workdir} && {_GIT_COMMIT_PRE_AGENT}", timeout=120,
        )
        result = await self.session.execute(
            f"cd {self.workdir} && git rev-parse HEAD", timeout=120,
        )
        if result.success and result.stdout.strip():
            return result.stdout.strip()
        return None

    async def remove_future_commits(self) -> None:
        """Reset all git refs to HEAD, clear stash.  Prevents data leakage."""
        await self.session.execute(
            f"cd {self.workdir} && {_REMOVE_FUTURE_COMMITS}",
        )

    async def prepare(self, instance: Instance) -> str | None:
        """Full pre-agent preparation: setup → commit snapshot → remove future commits.

        Returns the pre-agent commit SHA (or ``None`` on failure).
        """
        await self.run_setup_commands(instance.setup_commands)
        commit_id = await self.commit_and_get_id()
        await self.remove_future_commits()
        return commit_id
