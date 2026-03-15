"""Terminus JSON parser — extract keystrokes and task_complete from LLM response."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ParsedCommand:
    keystrokes: str
    duration: float


@dataclass
class ParseResult:
    commands: list[ParsedCommand]
    is_task_complete: bool
    error: str
    warning: str


class TerminusJSONParser:
    """Parser for Terminus JSON plain response format.

    Expects JSON with analysis, plan, commands, optional task_complete.
    Aligned with Harbor terminus_json_plain_parser.
    """

    def __init__(self) -> None:
        self.required_fields = ["analysis", "plan", "commands"]

    def parse_response(self, response: str) -> ParseResult:
        """Parse LLM response and extract commands."""
        result = self._try_parse_response(response)

        if result.error:
            for fix_name, fix_fn in self._get_auto_fixes():
                corrected, was_fixed = fix_fn(response, result.error)
                if was_fixed:
                    corrected_result = self._try_parse_response(corrected)
                    if corrected_result.error == "":
                        auto_warn = f"AUTO-CORRECTED: {fix_name}"
                        corrected_result.warning = self._combine_warnings(
                            auto_warn, corrected_result.warning
                        )
                        return corrected_result

        return result

    def _try_parse_response(self, response: str) -> ParseResult:
        warnings: list[str] = []
        json_content, extra_warnings = self._extract_json_content(response)
        warnings.extend(extra_warnings)

        if not json_content:
            return ParseResult(
                [], False, "No valid JSON found in response",
                "- " + "\n- ".join(warnings) if warnings else "",
            )

        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            err = f"Invalid JSON: {e}"
            if len(json_content) < 200:
                err += f" | Content: {repr(json_content)}"
            return ParseResult(
                [], False, err, "- " + "\n- ".join(warnings) if warnings else ""
            )

        validation_err = self._validate_structure(data, json_content, warnings)
        if validation_err:
            return ParseResult(
                [], False, validation_err,
                "- " + "\n- ".join(warnings) if warnings else "",
            )

        is_complete = data.get("task_complete", False)
        if isinstance(is_complete, str):
            is_complete = is_complete.lower() in ("true", "1", "yes")

        commands_data = data.get("commands", [])
        commands, parse_err = self._parse_commands(commands_data, warnings)
        if parse_err:
            if is_complete:
                warnings.append(parse_err)
                return ParseResult(
                    [], True, "", "- " + "\n- ".join(warnings) if warnings else ""
                )
            return ParseResult(
                [], False, parse_err,
                "- " + "\n- ".join(warnings) if warnings else "",
            )

        return ParseResult(
            commands, is_complete, "",
            "- " + "\n- ".join(warnings) if warnings else "",
        )

    def _extract_json_content(self, response: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        json_start = json_end = -1
        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(response):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    if brace_count == 0:
                        json_start = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and json_start != -1:
                        json_end = i + 1
                        break

        if json_start == -1 or json_end == -1:
            return "", ["No valid JSON object found"]

        if response[:json_start].strip():
            warnings.append("Extra text before JSON")
        if response[json_end:].strip():
            warnings.append("Extra text after JSON")

        return response[json_start:json_end], warnings

    def _validate_structure(
        self, data: dict, raw: str, warnings: list[str]
    ) -> str:
        if not isinstance(data, dict):
            return "Response must be a JSON object"
        missing = [f for f in self.required_fields if f not in data]
        if missing:
            return f"Missing required fields: {', '.join(missing)}"
        if not isinstance(data.get("commands"), list):
            return "Field 'commands' must be an array"
        if data.get("task_complete") is not None and not isinstance(
            data["task_complete"], (bool, str)
        ):
            warnings.append("task_complete should be boolean or string")
        return ""

    def _parse_commands(
        self, commands_data: list, warnings: list[str]
    ) -> tuple[list[ParsedCommand], str]:
        commands: list[ParsedCommand] = []
        for i, cmd in enumerate(commands_data):
            if not isinstance(cmd, dict):
                return [], f"Command {i + 1} must be an object"
            if "keystrokes" not in cmd:
                return [], f"Command {i + 1} missing 'keystrokes'"
            ks = cmd["keystrokes"]
            if not isinstance(ks, str):
                return [], f"Command {i + 1} keystrokes must be string"
            dur = 1.0
            if "duration" in cmd:
                d = cmd["duration"]
                if isinstance(d, (int, float)):
                    dur = float(d)
                else:
                    warnings.append(f"Command {i + 1}: invalid duration, using 1.0")
            else:
                warnings.append(f"Command {i + 1}: missing duration, using 1.0")
            commands.append(ParsedCommand(keystrokes=ks, duration=dur))
        return commands, ""

    def _get_auto_fixes(
        self,
    ) -> list[tuple[str, Callable[[str, str], tuple[str, bool]]]]:
        return [
            ("Fixed incomplete JSON", self._fix_incomplete),
            ("Extracted JSON from mixed content", self._fix_mixed),
        ]

    def _fix_incomplete(self, response: str, error: str) -> tuple[str, bool]:
        if any(
            x in error
            for x in ("Invalid JSON", "Expecting", "Unterminated", "No valid JSON")
        ):
            diff = response.count("{") - response.count("}")
            if diff > 0:
                return response + "}" * diff, True
        return response, False

    def _fix_mixed(self, response: str, error: str) -> tuple[str, bool]:
        pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        for m in re.findall(pattern, response, re.DOTALL):
            try:
                json.loads(m)
                return m, True
            except json.JSONDecodeError:
                continue
        return response, False

    def _combine_warnings(self, a: str, b: str) -> str:
        return f"- {a}\n{b}" if b else f"- {a}"
