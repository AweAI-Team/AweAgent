from __future__ import annotations

import json

from aweagent.core.llm.format.terminus_json import TerminusJSONFormat
from aweagent.core.llm.format.xml import CodeActXMLFormat
from aweagent.core.llm.types import LLMResponse
from aweagent.scaffold.terminus_2.parser import TerminusJSONParser


def test_text_formats_keep_default_observation_wrapper() -> None:
    format_ = CodeActXMLFormat()
    assert format_.format_text_observation("bash", "ok") == (
        "OBSERVATION:\n[bash]\nok"
    )


def test_terminus_format_returns_raw_terminal_observation() -> None:
    format_ = TerminusJSONFormat()
    assert format_.format_text_observation("tmux_execute", "screen") == "screen"


def test_reasoning_format_does_not_suppress_parser_warning() -> None:
    format_ = TerminusJSONFormat()
    format_.set_reasoning_format("think_tags")

    calls = format_.parse_response(
        LLMResponse(
            content=(
                "reasoning before JSON\n"
                '{"analysis":"a","plan":"p","commands":[]}'
            )
        )
    )

    arguments = json.loads(calls[0].arguments)
    assert arguments["warning"] == "- Extra text detected before JSON object"


def test_parser_uses_first_balanced_json_object_like_official():
    parser = TerminusJSONParser()
    response = """<think>
I can see the data files now.
Example record: {"id": 101, "full_name": "John Doe", "email": "john@a.com"}
</think>

{
  "analysis": "I can see the three data sources.",
  "plan": "Inspect parquet and CSV, then merge.",
  "commands": [
    {
      "keystrokes": "cat /data/source_b/users.csv\\n",
      "duration": 0.5
    }
  ],
  "task_complete": false
}
"""

    result = parser.parse_response(response)

    assert result.error == "Missing required fields: analysis, plan, commands"
    assert result.commands == []
    assert "Extra text detected before JSON object" in result.warning
    assert "Extra text detected after JSON object" in result.warning


def test_parser_does_not_select_a_later_json_candidate():
    parser = TerminusJSONParser()
    response = """prefix {"id": 101}

{"plan": "missing analysis and commands"}
"""

    result = parser.parse_response(response)

    assert result.error == "Missing required fields: analysis, plan, commands"


def test_parser_preserves_official_validation_warnings_and_reasoning_fields():
    parser = TerminusJSONParser()
    response = """prefix
{
  "commands": [
    {"keystrokes": "pwd", "extra": 1},
    {"keystrokes": "ls\\n", "duration": "fast"}
  ],
  "plan": ["inspect"],
  "analysis": 123,
  "task_complete": 0
}
suffix
"""

    result = parser.parse_response(response)

    assert result.error == ""
    assert result.analysis == 123
    assert result.plan == ["inspect"]
    assert result.is_task_complete == 0
    assert "Extra text detected before JSON object" in result.warning
    assert "Extra text detected after JSON object" in result.warning
    assert "Field 'analysis' should be a string" in result.warning
    assert "Field 'plan' should be a string" in result.warning
    assert "Fields appear in wrong order" in result.warning
    assert "Field 'task_complete' should be a boolean or string" in result.warning
    assert "Command 1: Missing duration field, using default 1.0" in result.warning
    assert "Command 1: Unknown fields: extra" in result.warning
    assert "Command 1 should end with newline" in result.warning
    assert "Command 2: Invalid duration value, using default 1.0" in result.warning


def test_parser_incomplete_json_auto_fix_matches_official_warning():
    parser = TerminusJSONParser()

    result = parser.parse_response(
        '{"analysis":"a","plan":"p","commands":[]'
    )

    assert result.error == ""
    assert result.warning.startswith(
        "- AUTO-CORRECTED: Fixed incomplete JSON by adding missing closing brace "
        "- please fix this in future responses"
    )


def test_task_complete_turns_command_parse_error_into_warning():
    parser = TerminusJSONParser()

    result = parser.parse_response(
        '{"analysis":"done","plan":"done","commands":[1],'
        '"task_complete":true}'
    )

    assert result.error == ""
    assert result.is_task_complete is True
    assert result.commands == []
    assert "Command 1 must be an object" in result.warning
