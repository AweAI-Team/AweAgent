"""Official-compatible tmux transport over an AweAgent RuntimeSession."""

from __future__ import annotations

import asyncio
import base64
import re
import shlex
import time
import uuid

from aweagent.core.runtime.protocol import RuntimeSession
from aweagent.core.runtime.types import ExecutionResult

_ENTER_KEYS = {"Enter", "C-m", "KPEnter", "C-j", "^M", "^J"}
_ENDS_WITH_NEWLINE = re.compile(r"[\r\n]$")
_NEWLINE_CHARS = "\r\n"
_TMUX_COMPLETION = "; tmux wait -S done"
_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH = 16_000
_TMUX_COMMAND_TOO_LONG_MARKER = "command too long"
_PASTE_BASE64_CHUNK_LEN = 65_536
_SESSION_LOGS_PATH = "/tmp/terminus_sessions"


class TmuxSessionAdapter:
    """Drive the task shell with the same tmux semantics as Terminus 2."""

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

    @staticmethod
    def _utf8_len(value: str) -> int:
        return len(value.encode("utf-8"))

    def _runtime_error(
        self,
        action: str,
        command: str,
        result: ExecutionResult,
    ) -> RuntimeError:
        return RuntimeError(
            f"Failed to {action}: command={command!r:.200}, "
            f"exit_code={result.exit_code}, stderr={result.stderr!r}, "
            f"stdout={result.stdout!r}"
        )

    async def _execute_checked(
        self,
        command: str,
        *,
        action: str,
        timeout: int | None = 30,
    ) -> ExecutionResult:
        result = await self._session.execute(
            command,
            cwd=self._workdir,
            timeout=timeout,
        )
        if not result.success:
            raise self._runtime_error(action, command, result)
        return result

    async def start(self) -> None:
        """Create a 160x40 login-shell pane and pipe it to a session log."""
        if self._started:
            return

        await self._execute_checked(
            f"mkdir -p {shlex.quote(_SESSION_LOGS_PATH)}",
            action="create tmux log directory",
        )

        session = shlex.quote(self._session_name)
        log_path = shlex.quote(self._log_path)
        command = (
            "export TERM=xterm-256color && "
            "export SHELL=/bin/bash && "
            f"tmux new-session -x 160 -y 40 -d -s {session} 'bash --login' \\; "
            f"set-option -t {session} history-limit 10000000 \\; "
            f"pipe-pane -t {session} 'cat > {log_path}'"
        )
        await self._execute_checked(
            command,
            action="start tmux session",
        )
        self._started = True

    @property
    def _send_keys_prefix(self) -> str:
        return f"tmux send-keys -t {shlex.quote(self._session_name)} --"

    @property
    def _max_escaped_key_length(self) -> int:
        return (
            _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
            - self._utf8_len(self._send_keys_prefix)
            - 1
        )

    def _send_keys_command(self, keys: list[str]) -> str:
        return self._send_keys_prefix + " " + " ".join(
            shlex.quote(key) for key in keys
        )

    def _key_requires_paste(self, key: str) -> bool:
        return self._utf8_len(shlex.quote(key)) > self._max_escaped_key_length

    def _batch_keys_for_send(self, keys: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        batch: list[str] = []
        batch_len = self._utf8_len(self._send_keys_prefix)

        for key in keys:
            escaped_len = self._utf8_len(shlex.quote(key))
            if escaped_len > self._max_escaped_key_length:
                raise ValueError("Oversized key must use the tmux paste buffer")
            addition = 1 + escaped_len
            if batch and batch_len + addition > _TMUX_SEND_KEYS_MAX_COMMAND_LENGTH:
                batches.append(batch)
                batch = []
                batch_len = self._utf8_len(self._send_keys_prefix)
            batch.append(key)
            batch_len += addition

        if batch:
            batches.append(batch)
        return batches

    @staticmethod
    def _is_executing_command(key: str) -> bool:
        return key in _ENTER_KEYS or bool(_ENDS_WITH_NEWLINE.search(key))

    def _prevent_execution(self, keys: list[str]) -> list[str]:
        prepared = keys.copy()
        while prepared and self._is_executing_command(prepared[-1]):
            if prepared[-1] in _ENTER_KEYS:
                prepared.pop()
                continue
            stripped = prepared[-1].rstrip(_NEWLINE_CHARS)
            if stripped:
                prepared[-1] = stripped
            else:
                prepared.pop()
        return prepared

    def _prepare_keys(
        self,
        keys: str | list[str],
        block: bool,
    ) -> tuple[list[str], bool]:
        prepared = [keys] if isinstance(keys, str) else list(keys)
        if (
            not block
            or not prepared
            or not self._is_executing_command(prepared[-1])
        ):
            return prepared, False

        prepared = self._prevent_execution(prepared)
        prepared.extend([_TMUX_COMPLETION, "Enter"])
        return prepared, True

    @staticmethod
    def _is_command_too_long_error(result: ExecutionResult) -> bool:
        return (
            not result.success
            and _TMUX_COMMAND_TOO_LONG_MARKER in (result.stderr or "")
        )

    async def _paste_key(self, key: str, action: str) -> None:
        """Stage a long literal outside tmux and paste it without truncation."""
        if not key:
            return

        token = uuid.uuid4().hex
        path = f"/tmp/.aweagent-tmux-paste-{token}"
        quoted_path = shlex.quote(path)
        buffer_name = f"aweagent-paste-{token}"
        payload = base64.b64encode(key.encode("utf-8")).decode("ascii")

        async def run(command: str) -> None:
            result = await self._session.execute(
                command,
                cwd=self._workdir,
                timeout=30,
            )
            if not result.success:
                raise self._runtime_error(f"send {action} keys", command, result)

        try:
            for offset in range(0, len(payload), _PASTE_BASE64_CHUNK_LEN):
                chunk = payload[offset : offset + _PASTE_BASE64_CHUNK_LEN]
                redirect = ">>" if offset else ">"
                await run(
                    f"printf %s {chunk} | base64 -d {redirect} {quoted_path}"
                )
            await run(
                f"tmux load-buffer -b {buffer_name} {quoted_path} && "
                f"tmux paste-buffer -d -b {buffer_name} "
                f"-t {shlex.quote(self._session_name)}"
            )
        finally:
            await self._session.execute(
                f"rm -f {quoted_path}",
                cwd=self._workdir,
                timeout=30,
            )

    async def _send_single_key(self, key: str, action: str) -> None:
        command = self._send_keys_command([key])
        result = await self._session.execute(
            command,
            cwd=self._workdir,
            timeout=30,
        )
        if result.success:
            return
        if self._is_command_too_long_error(result):
            await self._paste_key(key, action)
            return
        raise self._runtime_error(f"send {action} keys", command, result)

    async def _send_key_batches(self, keys: list[str], action: str) -> None:
        for batch in self._batch_keys_for_send(keys):
            command = self._send_keys_command(batch)
            result = await self._session.execute(
                command,
                cwd=self._workdir,
                timeout=30,
            )
            if result.success:
                continue
            if self._is_command_too_long_error(result):
                for key in batch:
                    await self._send_single_key(key, action)
                continue
            raise self._runtime_error(f"send {action} keys", command, result)

    async def _send_keys_to_session(self, keys: list[str], action: str) -> None:
        batch: list[str] = []

        async def flush() -> None:
            if batch:
                await self._send_key_batches(batch, action)
                batch.clear()

        for key in keys:
            if self._key_requires_paste(key):
                await flush()
                await self._paste_key(key, action)
            else:
                batch.append(key)
        await flush()

    async def send_keys(
        self,
        keys: str | list[str],
        block: bool = False,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 180.0,
    ) -> None:
        """Send literal/special keys, optionally waiting for command completion."""
        prepared, is_blocking = self._prepare_keys(keys, block)
        started = time.monotonic()

        if is_blocking:
            await self._send_keys_to_session(prepared, action="blocking")
            wait_command = f"timeout {max_timeout_sec}s tmux wait done"
            result = await self._session.execute(
                wait_command,
                cwd=self._workdir,
                timeout=int(max_timeout_sec) + 5,
            )
            if not result.success:
                raise TimeoutError(
                    f"Command timed out after {max_timeout_sec} seconds: "
                    f"{result.stderr or result.stdout}"
                )
            return

        await self._send_keys_to_session(prepared, action="non-blocking")
        remaining = min_timeout_sec - (time.monotonic() - started)
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def capture_pane(self, capture_entire: bool = False) -> str:
        """Capture the tmux pane and surface transport failures."""
        extra = "-S - " if capture_entire else ""
        command = (
            f"tmux capture-pane -p {extra}-t {shlex.quote(self._session_name)}"
        )
        result = await self._session.execute(
            command,
            cwd=self._workdir,
            timeout=30,
        )
        if not result.success:
            raise self._runtime_error("capture tmux pane", command, result)
        return result.stdout or ""

    async def is_session_alive(self) -> bool:
        result = await self._session.execute(
            f"tmux has-session -t {shlex.quote(self._session_name)}",
            cwd=self._workdir,
            timeout=30,
        )
        return result.success

    async def get_incremental_output(self) -> str:
        """Return newly appended pane output, falling back to the visible pane."""
        current_buffer = await self.capture_pane(capture_entire=True)

        if self._previous_buffer is None:
            self._previous_buffer = current_buffer
            return f"Current Terminal Screen:\n{await self._get_visible_screen()}"

        new_content = await self._find_new_content(current_buffer)
        self._previous_buffer = current_buffer
        if new_content is not None and new_content.strip():
            return f"New Terminal Output:\n{new_content}"
        return f"Current Terminal Screen:\n{await self._get_visible_screen()}"

    async def _get_visible_screen(self) -> str:
        return await self.capture_pane(capture_entire=False)

    async def _find_new_content(self, current_buffer: str) -> str | None:
        """Use Harbor's previous-buffer matching semantics verbatim."""
        previous = "" if self._previous_buffer is None else self._previous_buffer.strip()
        if previous in current_buffer:
            index = current_buffer.index(previous)
            if "\n" in previous:
                index = previous.rfind("\n")
            return current_buffer[index:]
        return None
