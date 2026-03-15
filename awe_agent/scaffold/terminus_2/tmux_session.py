"""TmuxSessionAdapter — tmux operations via AweAgent RuntimeSession.

Uses session.execute() to run tmux commands inside the container.
Aligned with Harbor's TmuxSessionTool interface (send_keys, get_incremental_output).
"""

from __future__ import annotations

import asyncio
import re
import shlex

from awe_agent.core.runtime.protocol import RuntimeSession

_ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
_ENDS_WITH_NEWLINE = re.compile(r"[\r\n]$")
_TMUX_COMPLETION = "; tmux wait -S done"
_SESSION_LOGS_PATH = "/tmp/terminus_sessions"


def _keystrokes_to_tmux_args(keystrokes: str) -> list[str]:
    """Convert keystrokes string to list of tmux send-keys arguments.

    "ls -la\\n" -> ["ls -la", "Enter"]
    "cd x\\n" -> ["cd x", "Enter"]
    """
    if not keystrokes:
        return []
    parts = keystrokes.split("\n")
    result: list[str] = []
    for i, p in enumerate(parts):
        if p or i < len(parts) - 1:
            result.append(p.replace("\r", ""))
        if i < len(parts) - 1:
            result.append("Enter")
    return result


class TmuxSessionAdapter:
    """Tmux session backed by RuntimeSession (Docker exec)."""

    def __init__(
        self,
        session: RuntimeSession,
        session_name: str = "terminus-session",
        workdir: str = "/workspace",
    ) -> None:
        self._session = session
        self._session_name = session_name
        self._workdir = workdir
        self._log_path = f"{_SESSION_LOGS_PATH}/{session_name}.log"
        self._previous_buffer: str | None = None
        self._started = False

    async def start(self) -> None:
        """Create tmux session and pipe pane to log file."""
        if self._started:
            return

        await self._session.execute(
            f"mkdir -p {_SESSION_LOGS_PATH}",
            cwd=self._workdir,
            timeout=30,
        )

        cmd = (
            f"tmux new-session -x 160 -y 40 -d -s {self._session_name} \\; "
            f"set-option -t {self._session_name} history-limit 10000000 \\; "
            f'pipe-pane -t {self._session_name} "cat > {self._log_path}"'
        )
        result = await self._session.execute(cmd, cwd=self._workdir, timeout=30)
        if not result.success:
            raise RuntimeError(
                f"Failed to start tmux session: {result.stderr or result.stdout}"
            )

        self._started = True

    def _build_send_keys_cmd(self, keys: list[str]) -> str:
        """Build tmux send-keys command string."""
        parts = []
        for k in keys:
            if k in _ENTER_KEYS:
                parts.append(k)
            else:
                parts.append(shlex.quote(k))
        return f"tmux send-keys -t {self._session_name} " + " ".join(parts)

    async def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        """Send keystrokes to tmux session."""
        if isinstance(keys, str):
            keys = _keystrokes_to_tmux_args(keys)
        elif keys and isinstance(keys[0], str):
            expanded = []
            for k in keys:
                expanded.extend(_keystrokes_to_tmux_args(k))
            keys = expanded

        if not keys:
            if min_timeout_sec > 0:
                await asyncio.sleep(min_timeout_sec)
            return

        def _is_executing(k: str) -> bool:
            return k in _ENTER_KEYS or bool(_ENDS_WITH_NEWLINE.search(k))

        if block and keys and _is_executing(keys[-1]):
            keys = keys.copy()
            while keys and _is_executing(keys[-1]):
                keys.pop()
            keys.append(_TMUX_COMPLETION)
            keys.append("Enter")

            cmd = self._build_send_keys_cmd(keys)
            await self._session.execute(cmd, cwd=self._workdir, timeout=10)

            wait_cmd = f"timeout {int(max_timeout_sec)}s tmux wait done"
            result = await self._session.execute(
                wait_cmd, cwd=self._workdir, timeout=int(max_timeout_sec) + 5
            )
            if result.exit_code != 0:
                raise TimeoutError(f"Command timed out after {max_timeout_sec}s")
        else:
            cmd = self._build_send_keys_cmd(keys)
            await self._session.execute(cmd, cwd=self._workdir, timeout=10)
            if min_timeout_sec > 0:
                await asyncio.sleep(min_timeout_sec)

    async def capture_pane(self, capture_entire: bool = False) -> str:
        """Capture tmux pane content."""
        extra = "-S -" if capture_entire else ""
        cmd = f"tmux capture-pane -p {extra} -t {self._session_name}".strip()
        result = await self._session.execute(cmd, cwd=self._workdir, timeout=30)
        return (result.stdout or "") + (result.stderr or "")

    async def get_incremental_output(self) -> str:
        """Get new terminal output since last call, or current screen."""
        current = await self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current
            visible = await self.capture_pane(capture_entire=False)
            return f"Current Terminal Screen:\n{visible}"

        pb = self._previous_buffer.strip()
        self._previous_buffer = current
        if pb and pb in current:
            idx = current.rfind(pb)
            if idx >= 0:
                new_part = current[idx + len(pb):].strip()
                if new_part:
                    return f"New Terminal Output:\n{new_part}"

        visible = await self.capture_pane(capture_entire=False)
        return f"Current Terminal Screen:\n{visible}"
