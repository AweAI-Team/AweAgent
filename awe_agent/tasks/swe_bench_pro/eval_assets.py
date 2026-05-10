"""Helpers for translating SWE-bench-Pro rows into official eval assets."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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


def load_icm_image_index(jsonl_path: str | Path) -> dict[str, str]:
    """Load instance_id → icm_image mapping from a SWE-bench-Pro images.jsonl."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench-Pro images.jsonl not found: {path}")

    index: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dockerfile = str(row.get("dockerfile") or "").strip()
            icm_image = str(row.get("icm_image") or "").strip()
            if not dockerfile or not icm_image:
                continue
            instance_id = (
                dockerfile[: -len(".Dockerfile")]
                if dockerfile.endswith(".Dockerfile")
                else dockerfile
            )
            index[instance_id] = icm_image
    return index


def resolve_image(raw: dict[str, Any], image_index: Mapping[str, str] | None = None) -> str:
    """Resolve the runtime image for a SWE-bench-Pro row from the images.jsonl index."""
    explicit_image = raw.get("oci_image") or raw.get("image") or raw.get("image_url")
    if explicit_image:
        return str(explicit_image)

    if not image_index:
        return ""
    instance_id = str(raw.get("instance_id") or "").strip()
    if not instance_id:
        return ""
    return image_index.get(instance_id, "")


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


def build_source_eval_assets(
    raw: dict[str, Any],
    *,
    workdir: str = "/app",
    official_repo_root: str | None = None,
) -> SWEBenchProEvalAssets:
    """Generate official SWE-bench-Pro eval assets from a raw dataset row."""
    if not official_repo_root:
        raise ValueError("official_repo_root is required for raw SWE-bench-Pro evaluation")

    root = Path(official_repo_root)
    if not root.exists():
        raise FileNotFoundError(f"Official SWE-bench-Pro repo not found: {official_repo_root}")

    instance_id = str(raw.get("instance_id") or "").strip()
    if not instance_id:
        raise ValueError("SWE-bench-Pro row is missing instance_id")

    run_script_sh = _load_official_script(root, instance_id, "run_script.sh")
    parser_py = _load_official_script(root, instance_id, "parser.py")
    entryscript_sh = _build_official_entryscript(raw, root, workdir)
    selected_targets = extract_selected_test_targets(raw)
    return SWEBenchProEvalAssets(
        entryscript_sh=entryscript_sh,
        run_script_sh=run_script_sh,
        parser_py=parser_py,
        selected_test_targets=selected_targets,
    )


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


def _load_official_script(root: Path, instance_id: str, script_name: str) -> str:
    script_path = root / "run_scripts" / instance_id / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Official SWE-bench-Pro script not found: {script_path}")
    return script_path.read_text(encoding="utf-8")


def _load_dockerfile(root: Path, docker_type: str, instance_id: str) -> str:
    dockerfile_path = root / "dockerfiles" / docker_type / instance_id / "Dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Official SWE-bench-Pro Dockerfile not found: {dockerfile_path}")
    return dockerfile_path.read_text(encoding="utf-8")


def _extract_env_exports(*dockerfiles: str) -> str:
    env_cmds: list[str] = []
    for dockerfile_content in dockerfiles:
        for raw_line in dockerfile_content.splitlines():
            line = raw_line.strip()
            if line.startswith("ENV"):
                env_cmds.append(line.replace("ENV", "export", 1))
    return "\n".join(env_cmds)


def _build_official_entryscript(
    raw: dict[str, Any],
    official_repo_root: Path,
    workdir: str,
) -> str:
    base_commit = str(raw.get("base_commit") or raw.get("parent_commit") or "").strip()
    if not base_commit:
        raise ValueError("SWE-bench-Pro row is missing base_commit")

    before_repo_set_cmd = str(raw.get("before_repo_set_cmd") or "").strip()
    before_repo_line = before_repo_set_cmd.splitlines()[-1].strip() if before_repo_set_cmd else ""
    selected_test_files = ",".join(
        _parse_swebench_list(raw.get("selected_test_files_to_run"))
    )

    base_dockerfile = _load_dockerfile(
        official_repo_root,
        "base_dockerfile",
        str(raw["instance_id"]),
    )
    instance_dockerfile = _load_dockerfile(
        official_repo_root,
        "instance_dockerfile",
        str(raw["instance_id"]),
    )
    env_cmds = _extract_env_exports(base_dockerfile, instance_dockerfile)

    lines = [
        env_cmds,
        "# apply patch",
        f"cd {shlex.quote(workdir)}",
        f"git reset --hard {shlex.quote(base_commit)}",
        f"git checkout {shlex.quote(base_commit)}",
        "git apply -v /workspace/patch.diff",
    ]
    if before_repo_line:
        lines.append(before_repo_line)
    lines.extend(
        [
            "# run test and save stdout and stderr to separate files",
            (
                f"bash /workspace/run_script.sh {selected_test_files} "
                "> /workspace/stdout.log 2> /workspace/stderr.log"
            ).strip(),
            "# run parsing script",
            (
                "python /workspace/parser.py /workspace/stdout.log "
                "/workspace/stderr.log /workspace/output.json"
            ),
        ]
    )
    return "\n".join(line for line in lines if line).rstrip() + "\n"


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
