"""Terminus 2 JSON-plain parser, behaviorally aligned with Harbor."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


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
    analysis: str = ""
    plan: str = ""


class TerminusJSONParser:
    """Parse the official Terminus JSON-plain response format."""

    def __init__(self) -> None:
        self.required_fields = ["analysis", "plan", "commands"]

    def parse_response(self, response: str) -> ParseResult:
        result = self._try_parse_response(response)
        if result.error:
            for fix_name, fix_function in self._get_auto_fixes():
                corrected_response, was_fixed = fix_function(response, result.error)
                if not was_fixed:
                    continue
                corrected_result = self._try_parse_response(corrected_response)
                if corrected_result.error == "":
                    auto_warning = (
                        f"AUTO-CORRECTED: {fix_name} - "
                        "please fix this in future responses"
                    )
                    corrected_result.warning = self._combine_warnings(
                        auto_warning,
                        corrected_result.warning,
                    )
                    return corrected_result
        return result

    def _try_parse_response(self, response: str) -> ParseResult:
        warnings: list[str] = []
        json_content, extra_text_warnings = self._extract_json_content(response)
        warnings.extend(extra_text_warnings)

        if not json_content:
            return ParseResult(
                [],
                False,
                "No valid JSON found in response",
                self._format_warnings(warnings),
                "",
                "",
            )

        try:
            parsed_data = json.loads(json_content)
        except json.JSONDecodeError as error:
            error_msg = f"Invalid JSON: {error}"
            if len(json_content) < 200:
                error_msg += f" | Content: {json_content!r}"
            else:
                error_msg += f" | Content preview: {json_content[:100]!r}..."
            return ParseResult(
                [],
                False,
                error_msg,
                self._format_warnings(warnings),
                "",
                "",
            )

        validation_error = self._validate_json_structure(
            parsed_data,
            json_content,
            warnings,
        )
        if validation_error:
            return ParseResult(
                [],
                False,
                validation_error,
                self._format_warnings(warnings),
                "",
                "",
            )

        is_complete = parsed_data.get("task_complete", False)
        if isinstance(is_complete, str):
            is_complete = is_complete.lower() in ("true", "1", "yes")

        analysis = parsed_data.get("analysis", "")
        plan = parsed_data.get("plan", "")
        commands, parse_error = self._parse_commands(
            parsed_data.get("commands", []),
            warnings,
        )
        if parse_error:
            if is_complete:
                warnings.append(parse_error)
                return ParseResult(
                    [],
                    True,
                    "",
                    self._format_warnings(warnings),
                    analysis,
                    plan,
                )
            return ParseResult(
                [],
                False,
                parse_error,
                self._format_warnings(warnings),
                analysis,
                plan,
            )

        return ParseResult(
            commands,
            is_complete,
            "",
            self._format_warnings(warnings),
            analysis,
            plan,
        )

    @staticmethod
    def _format_warnings(warnings: list[str]) -> str:
        return "- " + "\n- ".join(warnings) if warnings else ""

    def _extract_json_content(self, response: str) -> tuple[str, list[str]]:
        warnings: list[str] = []
        json_start = -1
        json_end = -1
        brace_count = 0
        in_string = False
        escape_next = False

        for index, char in enumerate(response):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if brace_count == 0:
                    json_start = index
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = index + 1
                    break

        if json_start == -1 or json_end == -1:
            return "", ["No valid JSON object found"]

        if response[:json_start].strip():
            warnings.append("Extra text detected before JSON object")
        if response[json_end:].strip():
            warnings.append("Extra text detected after JSON object")
        return response[json_start:json_end], warnings

    def _validate_json_structure(
        self,
        data: dict[str, Any],
        json_content: str,
        warnings: list[str],
    ) -> str:
        if not isinstance(data, dict):
            return "Response must be a JSON object"

        missing_fields = [
            field for field in self.required_fields if field not in data
        ]
        if missing_fields:
            return f"Missing required fields: {', '.join(missing_fields)}"

        if not isinstance(data.get("analysis", ""), str):
            warnings.append("Field 'analysis' should be a string")
        if not isinstance(data.get("plan", ""), str):
            warnings.append("Field 'plan' should be a string")

        if not isinstance(data.get("commands", []), list):
            return "Field 'commands' must be an array"

        self._check_field_order(data, json_content, warnings)

        task_complete = data.get("task_complete")
        if task_complete is not None and not isinstance(task_complete, (bool, str)):
            warnings.append("Field 'task_complete' should be a boolean or string")
        return ""

    def _parse_commands(
        self,
        commands_data: list[dict[str, Any]],
        warnings: list[str],
    ) -> tuple[list[ParsedCommand], str]:
        commands: list[ParsedCommand] = []
        for index, command_data in enumerate(commands_data):
            command_number = index + 1
            if not isinstance(command_data, dict):
                return [], f"Command {command_number} must be an object"
            if "keystrokes" not in command_data:
                return (
                    [],
                    f"Command {command_number} missing required 'keystrokes' field",
                )

            keystrokes = command_data["keystrokes"]
            if not isinstance(keystrokes, str):
                return (
                    [],
                    f"Command {command_number} 'keystrokes' must be a string",
                )

            if "duration" in command_data:
                duration = command_data["duration"]
                if not isinstance(duration, (int, float)):
                    warnings.append(
                        f"Command {command_number}: Invalid duration value, "
                        "using default 1.0"
                    )
                    duration = 1.0
            else:
                warnings.append(
                    f"Command {command_number}: Missing duration field, "
                    "using default 1.0"
                )
                duration = 1.0

            unknown_fields = set(command_data) - {"keystrokes", "duration"}
            if unknown_fields:
                warnings.append(
                    f"Command {command_number}: Unknown fields: "
                    f"{', '.join(unknown_fields)}"
                )

            if (
                index < len(commands_data) - 1
                and keystrokes
                and not keystrokes.endswith("\n")
            ):
                warnings.append(
                    f"Command {command_number} should end with newline when followed "
                    "by another command. Otherwise the two commands will be "
                    "concatenated together on the same line."
                )

            commands.append(
                ParsedCommand(
                    keystrokes=keystrokes,
                    duration=float(duration),
                )
            )
        return commands, ""

    def _get_auto_fixes(
        self,
    ) -> list[tuple[str, Callable[[str, str], tuple[str, bool]]]]:
        return [
            (
                "Fixed incomplete JSON by adding missing closing brace",
                self._fix_incomplete_json,
            ),
            ("Extracted JSON from mixed content", self._fix_mixed_content),
        ]

    @staticmethod
    def _fix_incomplete_json(response: str, error: str) -> tuple[str, bool]:
        if any(
            marker in error
            for marker in (
                "Invalid JSON",
                "Expecting",
                "Unterminated",
                "No valid JSON found",
            )
        ):
            brace_count = response.count("{") - response.count("}")
            if brace_count > 0:
                return response + "}" * brace_count, True
        return response, False

    @staticmethod
    def _fix_mixed_content(response: str, error: str) -> tuple[str, bool]:
        del error
        pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        for match in re.findall(pattern, response, re.DOTALL):
            try:
                json.loads(match)
                return match, True
            except json.JSONDecodeError:
                continue
        return response, False

    @staticmethod
    def _combine_warnings(auto_warning: str, existing_warning: str) -> str:
        if existing_warning:
            return f"- {auto_warning}\n{existing_warning}"
        return f"- {auto_warning}"

    @staticmethod
    def _check_field_order(
        data: dict[str, Any],
        response: str,
        warnings: list[str],
    ) -> None:
        del data
        expected_order = ["analysis", "plan", "commands"]
        positions: dict[str, int] = {}
        for field in expected_order:
            match = re.search(f'"({field})"\\s*:', response)
            if match:
                positions[field] = match.start()

        if len(positions) < 2:
            return
        actual_order = [
            field
            for field, _ in sorted(positions.items(), key=lambda item: item[1])
        ]
        expected_present = [field for field in expected_order if field in positions]
        if actual_order != expected_present:
            warnings.append(
                f"Fields appear in wrong order. Found: {' → '.join(actual_order)}, "
                f"expected: {' → '.join(expected_present)}"
            )
