"""Helpers for translating SWE-bench-Pro rows into official eval assets."""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SWEBenchProEvalAssets:
    """Files uploaded into the isolated evaluation workspace."""

    entryscript_sh: str
    run_script_sh: str
    parser_py: str
    selected_test_targets: list[str]


def normalize_repo_language(raw_language: str | None) -> str:
    """Normalize dataset language labels to AweAgent's canonical values."""
    language = (raw_language or "python").strip().lower()
    if language in {"py", "python"}:
        return "python"
    if language in {"javascript", "js"}:
        return "js"
    if language in {"typescript", "ts"}:
        return "ts"
    if language in {"golang", "go"}:
        return "go"
    return language


def resolve_image(raw: dict[str, Any]) -> str:
    """Resolve the runtime image for a SWE-bench-Pro row.

    Order: explicit ``oci_image`` / ``image`` / ``image_url`` → the row's own
    ``source_image`` (shipped by the public
    ``AweAI-Team/AweAgent-Meta-SWE-Bench-Pro`` dataset).
    """
    explicit_image = raw.get("oci_image") or raw.get("image") or raw.get("image_url")
    if explicit_image:
        return str(explicit_image)
    return str(raw.get("source_image") or "")


def has_prebuilt_eval_assets(raw: dict[str, Any]) -> bool:
    """Whether the row already contains swalm-style eval asset fields."""
    required_keys = {"entryscript_sh", "run_script_sh", "parser_py"}
    return required_keys <= raw.keys()


def extract_prebuilt_eval_assets(raw: dict[str, Any]) -> SWEBenchProEvalAssets | None:
    """Extract prebuilt eval assets from the row when present."""
    if not has_prebuilt_eval_assets(raw):
        return None
    return SWEBenchProEvalAssets(
        entryscript_sh=str(raw.get("entryscript_sh", "")),
        run_script_sh=str(raw.get("run_script_sh", "")),
        parser_py=str(raw.get("parser_py", "")),
        selected_test_targets=extract_selected_test_targets(raw),
    )


def extract_selected_test_targets(raw: dict[str, Any]) -> list[str]:
    """Get the repo-level test targets to execute during evaluation."""
    raw_targets = (
        raw.get("selected_test_files_to_run")
        or raw.get("selected_tests_to_run")
        or raw.get("selected_test_targets")
        or raw.get("selected_test_files")
        or ""
    )
    targets = _parse_jsonish_list(raw_targets)
    if targets:
        return _dedupe(targets)

    # Fallback: derive file paths from explicit test IDs.
    derived: list[str] = []
    for test_id in (
        _parse_swebench_list(raw.get("fail_to_pass", raw.get("FAIL_TO_PASS")))
        + _parse_swebench_list(raw.get("pass_to_pass", raw.get("PASS_TO_PASS")))
    ):
        file_part = str(test_id).split("::", 1)[0].strip()
        if file_part:
            derived.append(file_part)
    return _dedupe(derived)


def strip_binary_hunks(patch: str) -> str:
    """Match the official evaluator's binary-diff stripping behavior."""
    if not patch:
        return patch

    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def parse_swebench_expected_tests(raw: Any) -> list[str]:
    """Parse official SWE-bench-Pro FAIL_TO_PASS / PASS_TO_PASS fields."""
    return _parse_swebench_list(raw)


def _parse_jsonish_list(raw_value: Any) -> list[str]:
    if not raw_value:
        return []
    if isinstance(raw_value, list):
        return [str(v).strip() for v in raw_value if str(v).strip()]

    text = str(raw_value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None

    if isinstance(parsed, list):
        return [str(v).strip() for v in parsed if str(v).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]

    return [
        part.strip()
        for part in text.split(",")
        if part.strip()
    ]


def _parse_swebench_list(raw_value: Any) -> list[str]:
    return _parse_jsonish_list(raw_value)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        norm = os.path.normpath(value)
        if norm in seen:
            continue
        seen.add(norm)
        result.append(value)
    return result
