"""ScaleSWETask — SWE-bench-style coding benchmark.

Data format (JSONL):
    {
      "instance_id": "...",
      "user": "auth0",
      "repo": "auth0-python",
      "parent_commit": "...",
      "image_url": "...",
      "workdir": "/testbed",
      "problem_statement": "...",
      "pre_commands": "...",
      "f2p_patch": "...",
      "f2p_script": "...",
      "FAIL_TO_PASS": "[...]",
      "PASS_TO_PASS": "[...]",
      ...
    }

Key differences from BeyondSWE:
- No ``task`` field (always issue-resolving)
- ``parent_commit`` instead of ``base_commit``
- ``image_url`` instead of ``image``
- ``user`` + ``repo`` fields combined to form full repo name
- Uses simpler SWE-bench standard prompt
- Always ``enable_search=False`` (OpenHands style)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance

logger = logging.getLogger(__name__)


class ScaleSWETask(Task):
    """Task implementation for the ScaleSWE benchmark.

    Loads ScaleSWE JSONL data and maps its schema to the standard
    :class:`Instance` format.  Prompt selection is delegated to the
    scaffold's route table with ``dataset_id="scale_swe"``.

    Args:
        dataset_id: Dataset identifier (default ``"scale_swe"``).
        data_file: Path to JSONL data file.
        instances: Raw instance dicts for programmatic use.
    """

    def __init__(
        self,
        dataset_id: str = "scale_swe",
        data_file: str | None = None,
        instances: list[dict[str, Any]] | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.data_file = data_file
        self._raw_instances = instances
        self._loaded: list[dict[str, Any]] | None = None

    def _load_raw(self) -> list[dict[str, Any]]:
        if self._loaded is not None:
            return self._loaded

        if self._raw_instances is not None:
            self._loaded = self._raw_instances
            return self._loaded

        if self.data_file:
            path = Path(self.data_file)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            data = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            logger.info("Loaded %d ScaleSWE instances from %s", len(data), self.data_file)
            self._loaded = data
            return self._loaded

        raise ValueError("No data source configured. Provide data_file or instances.")

    def _to_instance(self, raw: dict[str, Any]) -> Instance:
        instance_id = raw.get("instance_id", "")

        # ScaleSWE uses parent_commit instead of base_commit
        base_commit = raw.get("parent_commit", raw.get("base_commit", ""))

        workdir = raw.get("workdir", "/testbed")

        # ScaleSWE uses image_url instead of image
        image = raw.get("image_url", raw.get("image", ""))

        language = raw.get("language", "python")
        problem_statement = raw.get("problem_statement", "")
        gold_patch = raw.get("patch", raw.get("fix_patch", ""))

        # Build full repo name from user + repo fields
        user = raw.get("user", "")
        repo = raw.get("repo", "")
        full_repo = f"{user}/{repo}" if user and repo else repo

        # Setup commands: pre_commands string, strip trailing \n
        setup_commands = []
        pre_commands = raw.get("pre_commands", "")
        if isinstance(pre_commands, str) and pre_commands.strip():
            setup_commands = [pre_commands.strip().removesuffix("\\n")]

        # Test info
        f2p = raw.get("FAIL_TO_PASS", "")
        p2p = raw.get("PASS_TO_PASS", "")

        return Instance(
            id=instance_id,
            dataset_id=self.dataset_id,
            repo=full_repo,
            base_commit=base_commit,
            workdir=workdir,
            image=image,
            language=language,
            problem_statement=problem_statement,
            gold_patch=gold_patch,
            setup_commands=setup_commands,
            metadata={
                "FAIL_TO_PASS": f2p,
                "PASS_TO_PASS": p2p,
                "f2p_patch": raw.get("f2p_patch", ""),
                "f2p_script": raw.get("f2p_script", ""),
                "raw": raw,
            },
        )

    # ─── Task Protocol ────────────────────────────────────────────────

    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        raw_data = self._load_raw()
        instances = [self._to_instance(r) for r in raw_data]

        if instance_ids is not None:
            id_set = {iid.lower() for iid in instance_ids}
            instances = [i for i in instances if i.id.lower() in id_set]

        return instances

    def get_prompt(self, instance: Instance) -> str:
        # Lazy imports to avoid circular dependency:
        # scaffold/prompts/user.py → scale_swe.prompt → scale_swe/__init__ → here
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

    def get_setup_commands(self, instance: Instance) -> list[str]:
        # ScaleSWE's pre_commands already includes git checkout parent_commit,
        # so we do NOT add an extra git checkout here.
        return list(instance.setup_commands)

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "workdir": instance.workdir,
            "language": instance.language,
        }

    def default_evaluator(self, timeout: int | None = None) -> Evaluator:
        """Return a ScaleSWEEvaluator for this task."""
        from awe_agent.tasks.scale_swe.evaluator import ScaleSWEEvaluator

        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return ScaleSWEEvaluator(**kwargs)
