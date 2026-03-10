"""BeyondSWETask — multi-type coding benchmark beyond standard SWE-Bench.

Supports four task types:
- doc2repo:   Build a repository from a specification document
- cross-repo: Fix issues spanning multiple files/modules
- refactor:   Refactor code while preserving functionality
- domain:     Solve domain-specific technical problems

Data format (JSONL):
    {
      "instance_id": "...",
      "task": "doc2repo|cross-repo|refactor|domain",
      "workdir": "/workspace",
      "image_url": "...",
      "problem_statement": "...",        # cross-repo, refactor, domain
      "REPO_DOCUMENT_CONTENT": "...",    # doc2repo
      "base_commit": "...",
      "FAIL_TO_PASS": "[...]",
      "PASS_TO_PASS": "[...]",
      ...
    }
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance
from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys
from awe_agent.scaffold.search_swe.prompts.user import get_user_prompt

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)

# Known BeyondSWE task types (lowercase, no separators).
_KNOWN_TASK_TYPES = {"doc2repo", "crossrepo", "depmigrate", "domainfix"}


def _normalize_task_type(raw_type: str) -> str:
    """Normalize a dataset task type string to lowercase without separators."""
    key = raw_type.lower().replace("_", "").replace("-", "").replace(" ", "")
    if key not in _KNOWN_TASK_TYPES:
        logger.warning(
            "Unknown BeyondSWE task type %r, falling back to 'domainfix'", raw_type,
        )
        return "domainfix"
    return key


class BeyondSWETask(Task):
    """Task implementation for the BeyondSWE benchmark.

    Handles all four BeyondSWE task types with task-specific prompts
    and setup logic.  Prompt selection is delegated to the scaffold's
    route table via ``(dataset_id, task_type, search_mode)``.

    Args:
        dataset_id: Dataset identifier (default ``"beyond_swe"``).
        data_file: Path to JSONL data file.
        instances: Raw instance dicts for programmatic use.
        search_mode: Whether search tools are enabled. Affects prompt
            selection via the route table.
        test_suite_dir: Directory containing test-suite zip files for
            doc2repo evaluation.  Falls back to the
            ``BEYONDSWE_TEST_SUITE_DIR`` environment variable.
    """

    def __init__(
        self,
        dataset_id: str = "beyond_swe",
        data_file: str | None = None,
        instances: list[dict[str, Any]] | None = None,
        search_mode: bool = False,
        test_suite_dir: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.data_file = data_file
        self._raw_instances = instances
        self._search_mode = search_mode
        self._test_suite_dir = test_suite_dir or os.environ.get("BEYONDSWE_TEST_SUITE_DIR", "")
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
            logger.info("Loaded %d BeyondSWE instances from %s", len(data), self.data_file)
            self._loaded = data
            return self._loaded

        raise ValueError("No data source configured. Provide data_file or instances.")

    def _to_instance(self, raw: dict[str, Any]) -> Instance:
        instance_id = raw.get("instance_id", "")
        task_type = _normalize_task_type(raw.get("task", "domainfix"))

        base_commit = (
            raw.get("base_commit")
            or raw.get("pre_agent_commit_id")
            or raw.get("parent_commit")
            or (raw.get("base", {}) or {}).get("sha", "")
        )
        workdir = raw.get("workdir", "/workspace")
        image = raw.get("image", raw.get("image_url", ""))
        language = raw.get("language", "python")

        # Problem statement for non-doc2repo tasks
        problem_statement = raw.get("problem_statement", "")

        # Repo document for doc2repo tasks
        repo_document = raw.get("REPO_DOCUMENT_CONTENT", "")

        # Gold patch
        gold_patch = raw.get("patch", raw.get("fix_patch", ""))

        # Test info
        f2p = raw.get("FAIL_TO_PASS", "")
        p2p = raw.get("PASS_TO_PASS", "")

        # Setup commands
        setup_commands = []
        pre_commands = raw.get("pre_commands", "")
        if isinstance(pre_commands, str) and pre_commands.strip():
            setup_commands = [pre_commands.strip().removesuffix("\\n")]
        elif isinstance(pre_commands, dict):
            exec_cmd = pre_commands.get("execute_command", {})
            if isinstance(exec_cmd, dict):
                setup_commands = exec_cmd.get("commands", [])

        return Instance(
            id=instance_id,
            dataset_id=self.dataset_id,
            repo=raw.get("repo", ""),
            base_commit=base_commit,
            workdir=workdir,
            image=image,
            language=language,
            problem_statement=problem_statement,
            gold_patch=gold_patch,
            setup_commands=setup_commands,
            metadata={
                "task_type": task_type,
                "FAIL_TO_PASS": f2p,
                "PASS_TO_PASS": p2p,
                "REPO_DOCUMENT_CONTENT": repo_document,
                "f2p_patch": raw.get("f2p_patch", ""),
                "f2p_script": raw.get("f2p_script", ""),
                "test_suite": raw.get("test_suite", ""),
                "test_suite_path": self._test_suite_dir,
                "test_suite_num": raw.get("test_suite_num", 0),
                "parent_commit": raw.get("parent_commit", ""),
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
        task_type = instance.metadata.get("task_type", "domainfix")

        # Resolve the user prompt key via the route table
        _, user_key = resolve_prompt_keys(
            dataset_id=self.dataset_id,
            task_type=task_type,
            search=self._search_mode,
        )
        template = get_user_prompt(user_key)

        # Fill template with instance data
        return template.format(
            workspace_dir=instance.workdir,
            problem_statement=instance.problem_statement,
            base_commit=instance.base_commit,
            workspace_tree=instance.metadata.get("workspace_tree", ""),
            installed_packages=instance.metadata.get("installed_packages", ""),
            REPO_DOCUMENT=instance.metadata.get("REPO_DOCUMENT_CONTENT", ""),
        )

    def get_llm_overrides(self, instance: Instance) -> dict[str, Any]:
        task_type = instance.metadata.get("task_type", "domainfix")
        if task_type == "doc2repo":
            return {"max_completion_tokens": 16384}
        return {}

    def get_image(self, instance: Instance) -> str:
        return instance.image

    def get_setup_commands(self, instance: Instance) -> list[str]:
        task_type = instance.metadata.get("task_type", "domainfix")
        commands = []

        if instance.base_commit:
            commands.append(
                f"git checkout {instance.base_commit}"
            )

        # For crossrepo and domainfix, verify parent commit checkout
        parent_commit = instance.metadata.get("parent_commit", "")
        if parent_commit and task_type in ("crossrepo", "domainfix"):
            commands.append(
                f"git log --oneline -1 | grep -q {parent_commit[:8]} || true"
            )

        # PXMeter doc2repo instance requires removing evaluation dir first
        if (
            task_type == "doc2repo"
            and instance.id == "PXMeter_7283384f03718625c13ee447b16db5206b45596b"
        ):
            commands.append("rm -rf /workspace/evaluation")

        commands.extend(instance.setup_commands)
        return commands

    async def prepare_session(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> None:
        """doc2repo: upload repo_document.md and collect installed packages."""
        task_type = instance.metadata.get("task_type", "domainfix")
        if task_type != "doc2repo":
            return

        # 1. Upload repo_document.md into the container
        repo_doc = instance.metadata.get("REPO_DOCUMENT_CONTENT", "")
        if repo_doc:
            doc_path = f"{instance.workdir}/repo_document.md"
            await session.upload_file(doc_path, repo_doc.encode())
            logger.info("Uploaded repo_document.md for %s", instance.id)

        # 2. Collect pip freeze so the prompt knows installed packages
        result = await session.execute("pip freeze", cwd=instance.workdir, timeout=60)
        if result.success:
            instance.metadata["installed_packages"] = result.stdout.strip()
        else:
            logger.warning(
                "pip freeze failed for %s: %s", instance.id, result.stderr[:200],
            )

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            "repo": instance.repo,
            "base_commit": instance.base_commit,
            "workdir": instance.workdir,
            "language": instance.language,
            "task_type": instance.metadata.get("task_type", "domainfix"),
        }

    def default_evaluator(self, timeout: int | None = None) -> Evaluator:
        """Return a BeyondSWEEvaluator for this task."""
        from awe_agent.tasks.beyond_swe.evaluator import BeyondSWEEvaluator

        kwargs = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return BeyondSWEEvaluator(**kwargs)
