from __future__ import annotations

from collections import deque

import pytest

from aweagent.core.runtime.types import ExecutionResult
from aweagent.scaffold.terminus_2.tmux_session import TmuxSessionAdapter


class FakeRuntime:
    def __init__(self, results: list[ExecutionResult] | None = None) -> None:
        self.results = deque(results or [])
        self.commands: list[str] = []

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        self.commands.append(command)
        if self.results:
            return self.results.popleft()
        return ExecutionResult()


@pytest.mark.asyncio
async def test_start_matches_official_login_shell_environment() -> None:
    runtime = FakeRuntime()
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    await tmux.start()

    assert runtime.commands[0].startswith("mkdir -p ")
    start_command = runtime.commands[1]
    assert "export TERM=xterm-256color" in start_command
    assert "export SHELL=/bin/bash" in start_command
    assert "tmux new-session -x 160 -y 40" in start_command
    assert "'bash --login'" in start_command
    assert "history-limit 10000000" in start_command
    assert "pipe-pane" in start_command


@pytest.mark.asyncio
async def test_short_string_is_sent_as_one_key_with_option_boundary() -> None:
    runtime = FakeRuntime()
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    await tmux.send_keys("echo hello\n")

    assert len(runtime.commands) == 1
    assert runtime.commands[0].startswith("tmux send-keys -t terminus-session -- ")
    assert "echo hello" in runtime.commands[0]


@pytest.mark.asyncio
async def test_oversized_unicode_key_uses_base64_paste_buffer() -> None:
    runtime = FakeRuntime()
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    await tmux.send_keys("界" * 6_000 + "\n")

    assert any("base64 -d" in command for command in runtime.commands)
    assert any("tmux load-buffer" in command for command in runtime.commands)
    assert any("tmux paste-buffer" in command for command in runtime.commands)
    assert runtime.commands[-1].startswith("rm -f /tmp/.aweagent-tmux-paste-")


@pytest.mark.asyncio
async def test_send_keys_failure_is_not_silently_ignored() -> None:
    runtime = FakeRuntime([ExecutionResult(stderr="boom", exit_code=1)])
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="boom"):
        await tmux.send_keys("pwd\n")


@pytest.mark.asyncio
async def test_runtime_command_too_long_error_falls_back_to_paste() -> None:
    runtime = FakeRuntime(
        [
            ExecutionResult(stderr="command too long", exit_code=1),
            ExecutionResult(stderr="command too long", exit_code=1),
        ]
    )
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    await tmux.send_keys("small enough but rejected by this tmux\n")

    assert any("tmux paste-buffer" in command for command in runtime.commands)


@pytest.mark.asyncio
async def test_blocking_send_appends_wait_marker_and_waits() -> None:
    runtime = FakeRuntime()
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    await tmux.send_keys("make test\n", block=True, max_timeout_sec=12.5)

    assert "; tmux wait -S done" in runtime.commands[0]
    assert runtime.commands[1] == "timeout 12.5s tmux wait done"


@pytest.mark.asyncio
async def test_capture_failure_is_not_returned_as_terminal_output() -> None:
    runtime = FakeRuntime([ExecutionResult(stderr="no server", exit_code=1)])
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no server"):
        await tmux.capture_pane()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code, expected", [(0, True), (1, False)])
async def test_session_liveness_uses_tmux_has_session(
    exit_code: int,
    expected: bool,
) -> None:
    runtime = FakeRuntime([ExecutionResult(exit_code=exit_code)])
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]

    assert await tmux.is_session_alive() is expected
    assert runtime.commands == ["tmux has-session -t terminus-session"]


@pytest.mark.asyncio
async def test_incremental_output_matches_official_previous_buffer_search() -> None:
    runtime = FakeRuntime([ExecutionResult(stdout="old\nnew")])
    tmux = TmuxSessionAdapter(runtime)  # type: ignore[arg-type]
    tmux._previous_buffer = "old"

    output = await tmux.get_incremental_output()

    assert output == "New Terminal Output:\nold\nnew"
