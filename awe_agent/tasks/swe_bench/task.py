"""SWEBenchTask — concrete Task implementation for SWE-Bench family datasets.

Supports loading instances from:
1. Local JSONL files (one JSON object per line)
2. HuggingFace datasets (e.g., princeton-nlp/SWE-bench_Verified)
3. Instance dicts passed directly (for programmatic use)

Example YAML config::

    task:
      type: swe_bench
      dataset_id: swe_bench_verified
      data_file: ./data/swe_bench_verified.jsonl    # or HuggingFace name
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance
from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys
from awe_agent.scaffold.search_swe.prompts.user import get_user_prompt

logger = logging.getLogger(__name__)


class SWEBenchTask(Task):
    """Task implementation for SWE-Bench family datasets.

    Loads instances from JSONL files or HuggingFace and generates
    prompts for the agent via the scaffold prompt routing system.

    Args:
        dataset_id: Dataset identifier (e.g. ``"swe_bench_verified"``).
        data_file: Path to JSONL data file.
        hf_dataset: HuggingFace dataset name.
        hf_split: HuggingFace dataset split.
        task_type: Task type string (e.g. ``"issue_resolving"``).
        instances: Raw instance dicts for programmatic use.
        search_mode: Whether search tools are enabled. Affects prompt
            selection via the route table.
    """

    def __init__(
        self,
        dataset_id: str = "swe_bench_verified",
        data_file: str | None = None,
        hf_dataset: str | None = None,
        hf_split: str = "test",
        task_type: str = "issue_resolving",
        instances: list[dict[str, Any]] | None = None,
        search_mode: bool = False,
    ) -> None:
        self.dataset_id = dataset_id
        self.data_file = data_file
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.task_type = task_type
        self._raw_instances = instances
        self._search_mode = search_mode
        self._loaded: list[dict[str, Any]] | None = None

    def _load_raw(self) -> list[dict[str, Any]]:
        """Lazy-load raw instance dicts from the configured source."""
        if self._loaded is not None:
            return self._loaded

        if self._raw_instances is not None:
            self._loaded = self._raw_instances
            return self._loaded

        if self.data_file:
            self._loaded = self._load_jsonl(self.data_file)
            return self._loaded

        if self.hf_dataset:
            self._loaded = self._load_huggingface(self.hf_dataset, self.hf_split)
            return self._loaded

        raise ValueError(
            "No data source configured. Provide data_file, hf_dataset, or instances."
        )

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """Load instances from a JSONL file."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logger.info("Loaded %d instances from %s", len(data), path)
        return data

    @staticmethod
    def _load_huggingface(dataset_name: str, split: str) -> list[dict[str, Any]]:
        """Load instances from a HuggingFace dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Install the 'datasets' package to load from HuggingFace: "
                "pip install datasets"
            )
        ds = load_dataset(dataset_name, split=split)
        data = [dict(row) for row in ds]
        logger.info("Loaded %d instances from HF %s/%s", len(data), dataset_name, split)
        return data

    def _to_instance(self, raw: dict[str, Any]) -> Instance:
        """Convert a raw dict to an Instance."""
        instance_id = raw.get("instance_id", "")

        # Normalize fields across different SWE-Bench formats
        repo = raw.get("repo", "")
        base_commit = (
            raw.get("base_commit")
            or raw.get("pre_agent_commit_id")
            or (raw.get("base", {}) or {}).get("sha", "")
        )
        workdir = raw.get("workdir", "/testbed")
        image = raw.get("image", raw.get("image_url", raw.get("oci_image", "")))
        language = raw.get("language", "python")

        # Problem statement
        problem_statement = raw.get("problem_statement", "")

        # Patches
        gold_patch = raw.get("patch", raw.get("fix_patch", ""))
        test_patch = raw.get("test_patch", "")

        # Test info
        f2p = raw.get("FAIL_TO_PASS", "")
        p2p = raw.get("PASS_TO_PASS", "")

        # Setup commands
        setup_commands = []
        pre_commands = raw.get("pre_commands", {})
        if isinstance(pre_commands, dict):
            exec_cmd = pre_commands.get("execute_command", {})
            if isinstance(exec_cmd, dict):
                setup_commands = exec_cmd.get("commands", [])
        elif isinstance(pre_commands, str):
            setup_commands = [pre_commands]

        return Instance(
            id=instance_id,
            dataset_id=self.dataset_id,
            repo=repo,
            base_commit=base_commit,
            workdir=workdir,
            image=image,
            language=language,
            problem_statement=problem_statement,
            gold_patch=gold_patch,
            test_patch=test_patch,
            setup_commands=setup_commands,
            metadata={
                "FAIL_TO_PASS": f2p,
                "PASS_TO_PASS": p2p,
                "version": raw.get("version", ""),
                "raw": raw,
            },
        )

    # ─── Task Protocol ────────────────────────────────────────────────

    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        """Load and return task instances, optionally filtered by IDs."""
        raw_data = self._load_raw()
        instances = [self._to_instance(r) for r in raw_data]

        if instance_ids is not None:
            id_set = {iid.lower() for iid in instance_ids}
            instances = [i for i in instances if i.id.lower() in id_set]

        return instances

    def get_prompt(self, instance: Instance) -> str:
        """Generate the task prompt via the scaffold prompt routing system."""
        _, user_key = resolve_prompt_keys(
            dataset_id=self.dataset_id,
            task_type=self.task_type,
            search=self._search_mode,
        )
        template = get_user_prompt(user_key)

        return template.format(
            workspace_dir=instance.workdir,
            problem_statement=instance.problem_statement,
            base_commit=instance.base_commit,
            workspace_tree="",
            installed_packages="",
            REPO_DOCUMENT="",
        )

    def get_image(self, instance: Instance) -> str:
        """Get the container image for this instance."""
        return instance.image

    def get_setup_commands(self, instance: Instance) -> list[str]:
        """Get environment setup commands."""
        commands = []
        if instance.base_commit:
            commands.append(
                f"cd {instance.workdir} && git checkout {instance.base_commit}"
            )
        commands.extend(instance.setup_commands)
        return commands

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        """Get task info dict passed to the agent context."""
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "workdir": instance.workdir,
            "language": instance.language,
            "task_type": self.task_type,
        }

    def default_evaluator(self) -> Evaluator:
        """Return a SWEBenchEvaluator for this task."""
        from awe_agent.tasks.swe_bench.evaluator import SWEBenchEvaluator

        return SWEBenchEvaluator()
