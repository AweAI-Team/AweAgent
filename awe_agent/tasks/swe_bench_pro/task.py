"""SWEBenchProTask — task mapping for the SWE-bench-Pro benchmark."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance
from awe_agent.tasks.swe_bench_pro.eval_assets import (
    build_source_eval_assets,
    extract_prebuilt_eval_assets,
    load_icm_image_index,
    normalize_repo_language,
    resolve_image,
)
from awe_agent.tasks.swe_bench_pro.problem_statement import build_problem_statement

logger = logging.getLogger(__name__)


class SWEBenchProTask(Task):
    """Task implementation for SWE-bench-Pro.

    Accepts:
    - programmatic ``instances=[...]``
    - JSONL / JSON files with one row per instance
    - local HuggingFace dataset folders / parquet files from SWE-bench_Pro
    """

    def __init__(
        self,
        dataset_id: str = "swe_bench_pro",
        data_file: str | None = None,
        instances: list[dict[str, Any]] | None = None,
        images_jsonl: str | None = None,
        official_repo_root: str | None = None,
        default_workdir: str = "/app",
        all_languages: bool = False,
        split_num: int | None = None,
        split_id: int | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.data_file = data_file
        self._raw_instances = instances
        self._images_jsonl = images_jsonl
        self._official_repo_root = official_repo_root
        self._default_workdir = default_workdir
        self._all_languages = all_languages
        self._split_num = split_num
        self._split_id = split_id
        self._loaded: list[dict[str, Any]] | None = None
        self._image_index: dict[str, str] | None = None
        self._validate_split_args()

    def _load_raw(self) -> list[dict[str, Any]]:
        if self._loaded is not None:
            return self._loaded

        if self._raw_instances is not None:
            self._loaded = self._apply_split(self._filter_raw_instances(self._raw_instances))
            return self._loaded

        if not self.data_file:
            raise ValueError("No data source configured. Provide data_file or instances.")

        path = Path(self.data_file)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        if path.is_dir():
            self._loaded = self._load_from_dataset_dir(path)
        elif path.suffix == ".jsonl":
            self._loaded = self._load_jsonl(path)
        elif path.suffix == ".json":
            self._loaded = self._load_json(path)
        elif path.suffix == ".csv":
            self._loaded = self._load_csv(path)
        elif path.suffix in {".yaml", ".yml"}:
            self._loaded = self._load_from_eval_yaml(path)
        elif path.suffix == ".parquet":
            self._loaded = self._load_parquet_files([path])
        else:
            raise ValueError(
                f"Unsupported SWE-bench-Pro data source: {self.data_file}. "
                "Expected a directory, jsonl, json, csv, yaml, or parquet."
            )

        self._loaded = self._apply_split(self._filter_raw_instances(self._loaded))
        logger.info(
            "Loaded %d SWE-bench-Pro instances from %s%s%s",
            len(self._loaded),
            self.data_file,
            "" if self._all_languages else " (language filter: python only)",
            (
                ""
                if self._split_num is None
                else f" (split {self._split_id}/{self._split_num})"
            ),
        )
        return self._loaded

    def _get_image_index(self) -> dict[str, str]:
        if self._image_index is not None:
            return self._image_index

        jsonl_path = self._resolve_images_jsonl_path()
        if jsonl_path is None:
            self._image_index = {}
            logger.info(
                "SWE-bench-Pro images.jsonl not configured; using each row's source_image"
            )
            return self._image_index

        self._image_index = load_icm_image_index(jsonl_path)
        logger.info(
            "Loaded %d SWE-bench-Pro icm images from %s",
            len(self._image_index),
            jsonl_path,
        )
        return self._image_index

    def _resolve_images_jsonl_path(self) -> Path | None:
        if self._images_jsonl:
            return Path(self._images_jsonl)
        if not self.data_file:
            return None
        data_path = Path(self.data_file)
        candidate_root = data_path if data_path.is_dir() else data_path.parent
        candidate = candidate_root / "images.jsonl"
        return candidate if candidate.exists() else None

    def _validate_split_args(self) -> None:
        if self._split_num is None and self._split_id is None:
            return
        if self._split_num is None or self._split_id is None:
            raise ValueError("split_num and split_id must be provided together")
        if self._split_num <= 0:
            raise ValueError("split_num must be a positive integer")
        if not 1 <= self._split_id <= self._split_num:
            raise ValueError("split_id must be within [1, split_num]")

    def _filter_raw_instances(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._all_languages:
            return rows

        filtered = [
            row for row in rows
            if normalize_repo_language(row.get("repo_language") or row.get("language")) == "python"
        ]
        logger.info(
            "Filtered SWE-bench-Pro instances to python only: kept %d/%d",
            len(filtered),
            len(rows),
        )
        return filtered

    def _apply_split(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._split_num is None:
            return rows

        total = len(rows)
        base = total // self._split_num
        remainder = total % self._split_num

        start = 0
        for current_split in range(1, self._split_id):
            start += base + (1 if current_split <= remainder else 0)

        size = base + (1 if self._split_id <= remainder else 0)
        end = start + size
        logger.info(
            "Applying deterministic split %d/%d: selecting rows [%d:%d) out of %d",
            self._split_id,
            self._split_num,
            start,
            end,
            total,
        )
        return rows[start:end]

    def _load_from_dataset_dir(self, root: Path) -> list[dict[str, Any]]:
        parquet_files = sorted((root / "data").glob("*.parquet"))
        if not parquet_files:
            parquet_files = sorted(root.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under SWE-bench-Pro dataset root: {root}"
            )
        return self._load_parquet_files(parquet_files)

    def _load_from_eval_yaml(self, yaml_path: Path) -> list[dict[str, Any]]:
        content = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        dataset_root = yaml_path.parent
        parquet_files: list[Path] = []

        data_files = content.get("configs", [{}])[0].get("data_files", [])
        for item in data_files:
            raw_path = item.get("path", "")
            if not raw_path:
                continue
            parquet_files.extend(sorted(dataset_root.glob(raw_path)))

        if not parquet_files:
            parquet_files = sorted((dataset_root / "data").glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files resolved from {yaml_path}")

        return self._load_parquet_files(parquet_files)

    def _load_parquet_files(self, parquet_files: list[Path]) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading SWE-bench-Pro parquet data requires pyarrow. "
                "Install project dependencies or convert the dataset to JSONL."
            ) from exc

        table = pq.read_table([str(path) for path in parquet_files])
        return table.to_pylist()

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a list in JSON data file: {path}")

    def _load_csv(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({key: value if value is not None else "" for key, value in row.items()})
        return rows

    def _to_instance(self, raw: dict[str, Any]) -> Instance:
        workdir = str(raw.get("workdir") or self._default_workdir)
        language = normalize_repo_language(
            raw.get("repo_language") or raw.get("language"),
        )
        raw_problem_statement = str(raw.get("problem_statement", ""))
        formatted_problem_statement = build_problem_statement(raw)
        fail_to_pass = raw.get("fail_to_pass", raw.get("FAIL_TO_PASS", ""))
        pass_to_pass = raw.get("pass_to_pass", raw.get("PASS_TO_PASS", ""))

        metadata: dict[str, Any] = {
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "repo_language": language,
            "task_type": language,
            "requirements": raw.get("requirements", ""),
            "interface": raw.get("interface", ""),
            "raw_problem_statement": raw_problem_statement,
            "formatted_problem_statement": formatted_problem_statement,
            "before_repo_set_cmd": raw.get("before_repo_set_cmd", ""),
            "selected_test_files_to_run": raw.get("selected_test_files_to_run", ""),
            "dockerhub_tag": raw.get("dockerhub_tag", ""),
            "raw": raw,
        }

        prebuilt_assets = extract_prebuilt_eval_assets(raw)
        if prebuilt_assets is not None:
            metadata.update(
                {
                    "entryscript_sh": prebuilt_assets.entryscript_sh,
                    "run_script_sh": prebuilt_assets.run_script_sh,
                    "parser_py": prebuilt_assets.parser_py,
                    "selected_test_targets": prebuilt_assets.selected_test_targets,
                }
            )
        else:
            try:
                generated_assets = build_source_eval_assets(
                    raw,
                    workdir=workdir,
                    official_repo_root=self._official_repo_root,
                )
            except (FileNotFoundError, NotImplementedError, ValueError) as exc:
                metadata["eval_assets_error"] = str(exc)
            else:
                metadata.update(
                    {
                        "entryscript_sh": generated_assets.entryscript_sh,
                        "run_script_sh": generated_assets.run_script_sh,
                        "parser_py": generated_assets.parser_py,
                        "selected_test_targets": generated_assets.selected_test_targets,
                    }
                )

        return Instance(
            id=str(raw.get("instance_id", "")),
            dataset_id=self.dataset_id,
            repo=str(raw.get("repo", "")),
            base_commit=str(raw.get("base_commit", raw.get("parent_commit", ""))),
            workdir=workdir,
            image=resolve_image(raw, self._get_image_index()),
            language=language,
            problem_statement=formatted_problem_statement,
            gold_patch=str(raw.get("patch", raw.get("golden_patch", ""))),
            test_patch=str(raw.get("test_patch", "")),
            metadata=metadata,
        )

    # ─── Task Protocol ────────────────────────────────────────────────

    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        instances = [self._to_instance(row) for row in self._load_raw()]
        if instance_ids is None:
            return instances

        wanted = {instance_id.lower() for instance_id in instance_ids}
        return [instance for instance in instances if instance.id.lower() in wanted]

    def get_prompt(self, instance: Instance) -> str:
        from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys
        from awe_agent.scaffold.search_swe.prompts.user import get_user_prompt

        _, user_key = resolve_prompt_keys(
            dataset_id=self.dataset_id,
            task_type=None,
            search=False,
        )
        template = get_user_prompt(user_key)
        return template.format(
            workspace_dir=instance.workdir,
            problem_statement=instance.problem_statement,
        )

    def get_image(self, instance: Instance) -> str:
        return instance.image

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        task_info = super().get_task_info(instance)
        task_info["repo_language"] = instance.metadata.get("repo_language", instance.language)
        task_info["requirements"] = instance.metadata.get("requirements", "")
        task_info["interface"] = instance.metadata.get("interface", "")
        # Aligned with official SWE-bench-Pro: patch is the raw
        # ``git diff <base_commit>``; do not inject .gitignore rules into
        # the working tree before the diff (would otherwise pollute the
        # patch with an "AWEAGENT AUTO-GENERATED" .gitignore block).
        task_info["inject_gitignore_in_patch"] = False
        return task_info

    def requires_git_snapshot(self) -> bool:
        """Aligned with official SWE-bench-Pro: no pre-agent snapshotting.

        Official ``swe_bench_pro_eval.py`` evaluates a patch produced by
        ``git diff <base_commit>`` (the SWE-agent ``submit`` tool).  We
        match that by computing the patch directly against ``base_commit``
        instead of an intermediate ``pre-agent commit``.  This also avoids
        the ``remove_future_commits`` step, which is an AweAgent-only
        safeguard that the official flow does not perform.

        Patch extraction itself is preserved via :meth:`requires_patch_extraction`
        (defaults to ``True``).
        """
        return False

    def default_evaluator(self, timeout: int | None = None) -> Evaluator:
        from awe_agent.tasks.swe_bench_pro.evaluator import SWEBenchProEvaluator

        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return SWEBenchProEvaluator(**kwargs)
