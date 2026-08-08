from __future__ import annotations

import pytest

from aweagent.scaffold.terminus_2.tmux_tool import TmuxExecuteTool


class FailingTmux:
    async def send_keys(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("tmux transport broke")


class TimingOutTmux:
    async def send_keys(self, *args: object, **kwargs: object) -> None:
        raise TimeoutError("slow")

    async def get_incremental_output(self) -> str:
        return "still running"


class QuietTmux:
    async def send_keys(self, *args: object, **kwargs: object) -> None:
        return None

    async def get_incremental_output(self) -> str:
        return "x" * 20


class RecordingTmux:
    def __init__(self) -> None:
        self.min_timeout_secs: list[float] = []

    async def send_keys(self, *args: object, **kwargs: object) -> None:
        self.min_timeout_secs.append(float(kwargs["min_timeout_sec"]))

    async def get_incremental_output(self) -> str:
        return "ready"


@pytest.mark.asyncio
async def test_transport_failure_propagates_instead_of_becoming_observation() -> None:
    tool = TmuxExecuteTool(FailingTmux())  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="tmux transport broke"):
        await tool.execute(
            {"commands": [{"keystrokes": "pwd\n", "duration": 1.0}]}
        )


@pytest.mark.asyncio
async def test_timeout_uses_official_recovery_observation() -> None:
    tool = TmuxExecuteTool(TimingOutTmux())  # type: ignore[arg-type]

    output = await tool.execute(
        {"commands": [{"keystrokes": "sleep 10\n", "duration": 2.0}]}
    )

    assert output == (
        "Previous command:\n"
        "sleep 10\n\n\n"
        "The previous command timed out after 2.0 seconds\n\n"
        "It is possible that the command is not yet finished executing. "
        "If that is the case, then do nothing. It is also possible that "
        "you have entered an interactive shell and should continue sending "
        "keystrokes as normal.\n\n"
        "Here is the current state of the terminal:\n\n"
        "still running "
    )


@pytest.mark.parametrize(
    ("command", "expected_duration"),
    [
        ({"keystrokes": "true\n", "duration": -0.5}, -0.5),
        ({"keystrokes": "true\n", "duration": 0.0}, 0.0),
        ({"keystrokes": "true\n", "duration": 0.05}, 0.05),
        ({"keystrokes": "true\n", "duration": 0.1}, 0.1),
        ({"keystrokes": "true\n"}, 1.0),
        ({"keystrokes": "true\n", "duration": 75.0}, 60.0),
    ],
)
@pytest.mark.asyncio
async def test_duration_matches_harbor_without_minimum_floor(
    command: dict[str, object],
    expected_duration: float,
) -> None:
    tmux = RecordingTmux()
    tool = TmuxExecuteTool(tmux)  # type: ignore[arg-type]

    await tool.execute({"commands": [command]})

    assert tmux.min_timeout_secs == [expected_duration]


@pytest.mark.asyncio
async def test_output_limit_message_matches_official_template() -> None:
    tool = TmuxExecuteTool(QuietTmux(), max_output_bytes=10)  # type: ignore[arg-type]

    output = await tool.execute({"commands": []})

    assert output == (
        "xxxxx\n[... output limited to 10 bytes; "
        "10 interior bytes omitted ...]\nxxxxx"
    )


@pytest.mark.asyncio
async def test_successful_warning_uses_exact_harbor_observation() -> None:
    tool = TmuxExecuteTool(QuietTmux())  # type: ignore[arg-type]

    output = await tool.execute(
        {"commands": [], "warning": "- Extra text detected before JSON object"}
    )

    assert output == (
        "Previous response had warnings:\n"
        "WARNINGS: - Extra text detected before JSON object\n\n"
        + "x" * 20
    )


@pytest.mark.asyncio
async def test_completion_confirmation_ignores_parser_warning() -> None:
    tool = TmuxExecuteTool(QuietTmux())  # type: ignore[arg-type]

    output = await tool.execute(
        {
            "commands": [],
            "task_complete": True,
            "warning": "- Extra text detected before JSON object",
        }
    )

    assert output == (
        "Current terminal state:\n"
        + "x" * 20
        + "\n\nAre you sure you want to mark the task as complete? "
        "This will trigger your solution to be graded and you won't be able to "
        'make any further corrections. If so, include "task_complete": true '
        "in your JSON response again."
    )
