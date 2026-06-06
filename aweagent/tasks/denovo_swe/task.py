"""DeNovoSWETask — build repositories from scratch with source-cleaned images.

Similar to BeyondSWE doc2repo, but with key differences:

1. **Image cleaning**: Source code still exists in the image and must be
   cleaned via ``clean.sh`` before agent execution.
2. **Workflow**: clean -> inject README.md -> agent -> apply test patch -> eval
3. **Validate run**: ``--validate-run`` skips agent, goes straight to eval
   to verify that golden patches + unit tests work correctly.

DeNovoSWETask inherits :class:`BeyondSWETask` to reuse the boilerplate
``get_instances`` filtering.  All business-logic methods (prompt rendering,
session prep, evaluator selection) are overridden because the workflow
diverges substantially.  ``__init__`` deliberately does NOT call
``super().__init__()`` — BeyondSWE's constructor reads
``BEYONDSWE_TEST_SUITE_DIR`` from the environment, which is irrelevant here.

Data format (JSONL — output of ``recipes/denovo_swe/extract_patch.py``):
    {
      "instance_id": "...",
      "workdir": "/workspace/...",
      "image": "...",
      "repo": "...",
      "parent_commit": "...",
      "document": "...",
      "passed_ptp": [...],
      "failed_ptp": [...],
      "test_patch": "...",                # text-only unit-test patch
      "test_files": [...],                # all test file paths covered
      "test_binary_archive_b64": "...",   # base64 tar.gz of binary
                                          # fixtures (unified diffs cannot
                                          # carry binaries).  Evaluator
                                          # uploads + extracts this after
                                          # applying test_patch.
      "test_binary_files": [...],         # paths packed inside the archive
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aweagent.core.task.protocol import Evaluator
from aweagent.core.task.types import Instance
from aweagent.core.tool.search.constraints import SearchConstraints
from aweagent.core.tool.search.package_constraints import PackageSearchConstraints
from aweagent.scaffold.search_swe.prompts.config import resolve_prompt_keys
from aweagent.tasks.beyond_swe.task import BeyondSWETask

# ``get_user_prompt`` is imported lazily inside ``get_prompt`` because the
# scaffold's user-prompt aggregator imports from every task module — a
# top-level import here would re-enter that module while it is still
# initialising.
if TYPE_CHECKING:
    from aweagent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)

# Path to clean.sh bundled with this module.  ``abspath`` collapses any
# ``../..`` segments from running this task via a recipe before any
# ``open()`` call — otherwise we get sporadic ``FileNotFoundError`` on
# busy/NFS volumes.
_CLEAN_SH_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "clean.sh"))


class DeNovoSWETask(BeyondSWETask):
    """Task implementation for the DeNovoSWE benchmark.

    Handles doc2repo-style tasks where the agent must build a repository
    from a specification document, but the image still contains source code
    that must be cleaned first.

    Args:
        dataset_id: Dataset identifier (default ``"denovo_swe"``).
        data_file: Path to JSONL data file.
        instances: Raw instance dicts for programmatic use.
        search_mode: Whether search tools are enabled.
        validate_run: If True, skip agent execution and go straight to
            evaluation to verify test patches work correctly.
        del_done_images: Delete the docker image after each instance.
        clean_snapshot_file: Optional JSONL path receiving a snapshot of
            files remaining after ``clean.sh`` ran (for audit).
        prompt_version: ``"v1"`` (original) or ``"v2"`` (default — adds
            the public-API verification gate).
        eval_iters: Run the evaluator N times in independent sandboxes
            and report the mean score (mitigates eval flakiness).
    """

    def __init__(
        self,
        dataset_id: str = "denovo_swe",
        data_file: str | None = None,
        instances: list[dict[str, Any]] | None = None,
        search_mode: bool = False,
        validate_run: bool = False,
        del_done_images: bool = False,
        clean_snapshot_file: str | None = None,
        prompt_version: str = "v2",
        eval_iters: int = 1,
    ) -> None:
        # NOT calling super().__init__() on purpose: BeyondSWETask reads
        # the BEYONDSWE_TEST_SUITE_DIR env var and stores a
        # ``_test_suite_dir`` attribute we do not need.  Reproduce only
        # the fields the inherited ``get_instances`` relies on.
        self.dataset_id = dataset_id
        self.data_file = data_file
        self._raw_instances = instances
        self._search_mode = search_mode
        # DeNovoSWE-specific state
        self._validate_run = validate_run
        self._del_done_images = del_done_images
        self._clean_snapshot_file = clean_snapshot_file
        self._prompt_version = prompt_version
        self._eval_iters = eval_iters
        self._snapshot_lock: asyncio.Lock | None = None
        self._loaded: list[dict[str, Any]] | None = None

    def _load_raw(self) -> list[dict[str, Any]]:
        # Overridden (rather than inherited) so the log line keeps the
        # "DeNovoSWE" label.  Implementation is otherwise identical to
        # ``BeyondSWETask._load_raw``.
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
            logger.info("Loaded %d DeNovoSWE instances from %s", len(data), self.data_file)
            self._loaded = data
            return self._loaded

        raise ValueError("No data source configured. Provide data_file or instances.")

    def _to_instance(self, raw: dict[str, Any]) -> Instance:
        instance_id = raw.get("instance_id", "")
        base_commit = raw.get("parent_commit", "")
        workdir = raw.get("workdir", "/workspace")
        image = raw.get("image", raw.get("image_url", ""))
        language = raw.get("language", "python")

        document = raw.get("document", "")

        # Test patch and test files (from extract_patch step)
        test_patch = raw.get("test_patch", "")
        test_files = raw.get("test_files", [])

        # Binary fixtures — unified text diffs cannot carry binary content,
        # so extract_patch.py packs them into a base64-encoded tar.gz.
        # Older datasets predate this field; absence means "no binary
        # fixtures" — fully backward compatible.
        test_binary_archive_b64 = raw.get("test_binary_archive_b64", "") or ""
        test_binary_files = list(raw.get("test_binary_files", []) or [])

        # Cheating-risk audit field: paths grabbed from the sibling-helper
        # whitelist that shadow the agent's expected package.  We forward
        # to metadata so the evaluator surfaces it without acting on it.
        bug_b_overrides = list(raw.get("bug_b_overrides_agent_code", []) or [])

        passed_ptp = raw.get("passed_ptp", [])
        failed_ptp = raw.get("failed_ptp", [])

        # Package-name fields used by ``get_search_constraints`` to seed
        # the anti-hack bash blocklist.  All optional — every field may
        # be absent on legacy rows.
        pypi_name = str(raw.get("pypi_name") or "")
        pypi_candidates = list(raw.get("pypi_name_candidates") or [])
        import_names = list(raw.get("import_names") or [])

        return Instance(
            id=instance_id,
            dataset_id=self.dataset_id,
            repo=raw.get("repo", ""),
            base_commit=base_commit,
            workdir=workdir,
            image=image,
            language=language,
            problem_statement="",
            gold_patch="",
            test_patch=test_patch,
            metadata={
                "task_type": "doc2repo",
                "document": document,
                "test_patch": test_patch,
                "test_files": test_files,
                "test_binary_archive_b64": test_binary_archive_b64,
                "test_binary_files": test_binary_files,
                "bug_b_overrides_agent_code": bug_b_overrides,
                "passed_ptp": passed_ptp,
                "failed_ptp": failed_ptp,
                "validate_run": self._validate_run,
                "pypi_name": pypi_name,
                "pypi_name_candidates": pypi_candidates,
                "import_names": import_names,
                "raw": raw,
            },
        )

    # ─── Task Protocol ────────────────────────────────────────────────

    # ``get_instances`` is inherited from BeyondSWETask verbatim.  Other
    # base-class methods (``get_image``, ``requires_git_snapshot``,
    # ``requires_patch_extraction``, ``collect_artifact``,
    # ``get_resource_limits``, ``get_docker_environment``) use the
    # ``Task`` defaults, which match the previous DeNovoSWE behavior.

    def get_prompt(self, instance: Instance) -> str:
        from aweagent.scaffold.search_swe.prompts.user import get_user_prompt

        _, user_key = resolve_prompt_keys(
            dataset_id=self.dataset_id,
            task_type="doc2repo",
            search=self._search_mode,
        )
        if self._prompt_version and self._prompt_version != "v1":
            user_key = f"{user_key}_{self._prompt_version}"
        template = get_user_prompt(user_key)

        return template.format(
            workspace_dir=instance.workdir,
            problem_statement="",
            base_commit=instance.base_commit,
            installed_packages=instance.metadata.get("installed_packages", ""),
            REPO_DOCUMENT=instance.metadata.get("document", ""),
        )

    def get_llm_overrides(self, instance: Instance) -> dict[str, Any]:
        return {"max_completion_tokens": 16384}

    def get_search_constraints(self, instance: Instance) -> SearchConstraints | None:
        """Anti-hack bash blocklist targeting the target package.

        The default :meth:`Task.get_search_constraints` only generates
        per-repo patterns (``git clone`` / ``github.com/<repo>``), which
        leaves the whole PyPI/conda/uv ecosystem wide open.  DeNovoSWE's
        ``clean.sh`` uninstalls the target package and wipes its source,
        but nothing stops a motivated agent from running
        ``pip install <pkg>`` / ``git clone <upstream>`` /
        ``curl files.pythonhosted.org/...`` to pull golden source back in.

        We return a :class:`PackageSearchConstraints` seeded from every
        identifier on the JSONL row (``pypi_name`` / ``import_names`` /
        ``pypi_name_candidates`` / repo slug).  Returns ``None`` only
        when every identifier is empty — in that case the agent still
        gets the unconditional patterns from :class:`SearchSWEAgent`.
        """
        pypi_name = instance.metadata.get("pypi_name") or ""
        candidates = instance.metadata.get("pypi_name_candidates") or []
        import_names = instance.metadata.get("import_names") or []

        extras: list[str] = []
        for c in candidates:
            if isinstance(c, dict):
                name = c.get("name")
                if isinstance(name, str):
                    extras.append(name)
            elif isinstance(c, str):
                extras.append(c)
        extras.extend(n for n in import_names if isinstance(n, str))

        # Primary seed: PyPI name if we have it, else the repo slug.
        # Do not fold instance_id into package-name matching: ids include
        # owner/PR tokens and make package-manager rules too broad.
        primary = pypi_name or instance.repo or (extras[0] if extras else "")

        if not primary and not extras:
            return None

        return PackageSearchConstraints.from_package_name(
            package_name=primary,
            instance_id=None,
            extra_identifiers=extras,
            repo=instance.repo or None,
        )

    def get_setup_commands(self, instance: Instance) -> list[str]:
        commands: list[str] = []
        if instance.base_commit:
            commands.append(f"git checkout -f {instance.base_commit}")
        return commands

    async def prepare_session(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> None:
        """Clean source code, inject README.md, and collect installed packages."""
        workdir = instance.workdir

        # 1. Upload and execute clean.sh to remove source code
        with open(_CLEAN_SH_PATH, "rb") as f:
            clean_script = f.read()
        await session.upload_file(f"{workdir}/clean.sh", clean_script)
        result = await session.execute(
            f"bash {workdir}/clean.sh {workdir}",
            cwd=workdir, timeout=300,
        )
        if not result.success:
            logger.error(
                "clean.sh failed for %s (exit=%d): %s",
                instance.id, result.exit_code, result.stderr[:500],
            )
            raise RuntimeError(
                f"clean.sh failed for {instance.id}: {result.stderr[:200]}"
            )
        logger.info("Source code cleaned for %s", instance.id)
        await session.execute(f"rm -f {workdir}/clean.sh", cwd=workdir)

        # 2. Snapshot remaining files after clean (for inspection)
        snapshot_result = await session.execute(
            "find . -type f -not -path './.git/*' | sort",
            cwd=workdir, timeout=60,
        )
        remaining_files: list[str] = []
        git_commits: list[str] = []
        if snapshot_result.success:
            remaining_files = [
                f.strip().removeprefix("./")
                for f in snapshot_result.stdout.strip().split("\n")
                if f.strip()
            ]
            instance.metadata["post_clean_files"] = remaining_files
            logger.info(
                "Post-clean snapshot for %s: %d files remaining",
                instance.id, len(remaining_files),
            )

            git_log_result = await session.execute(
                "git log --all --oneline 2>/dev/null",
                cwd=workdir, timeout=10,
            )
            git_commits = (
                git_log_result.stdout.strip().split("\n")
                if git_log_result.stdout.strip() else []
            )
            instance.metadata["post_clean_git_commits"] = len(git_commits)
            logger.info(
                "Post-clean git for %s: %d commits (should be 1)",
                instance.id, len(git_commits),
            )

        # 3. Optionally dump full snapshot (paths + contents) to file
        if self._clean_snapshot_file and snapshot_result.success:
            if self._snapshot_lock is None:
                self._snapshot_lock = asyncio.Lock()

            file_contents: dict[str, str | None] = {}
            for fpath in remaining_files:
                cat_result = await session.execute(
                    f'cat "{fpath}" 2>/dev/null',
                    cwd=workdir, timeout=10,
                )
                file_contents[fpath] = cat_result.stdout if cat_result.success else None

            snapshot_record = {
                "instance_id": instance.id,
                "repo": instance.repo,
                "workdir": workdir,
                "post_clean_files": remaining_files,
                "post_clean_file_count": len(remaining_files),
                "post_clean_git_commits": len(git_commits),
                "file_contents": file_contents,
            }
            async with self._snapshot_lock:
                with open(self._clean_snapshot_file, "a") as sf:
                    sf.write(json.dumps(snapshot_record, ensure_ascii=False) + "\n")

        # 4. Upload document as README.md and fold it into the baseline
        #    git commit produced by clean.sh.  Without this amend the
        #    agent's patch would carry a spurious +N-line README addition.
        document = instance.metadata.get("document", "")
        if document:
            await session.upload_file(
                f"{workdir}/README.md", document.encode(),
            )
            await session.execute(
                "git add README.md && "
                "git commit --amend --no-edit --allow-empty 2>/dev/null || true",
                cwd=workdir, timeout=30,
            )
            logger.info(
                "Uploaded README.md for %s (folded into baseline commit)",
                instance.id,
            )

        # 5. Fold the runtime-managed ``.gitignore`` block into the
        #    baseline commit too.  Otherwise every agent patch ended with
        #    a ~12-line ``AWEAGENT AUTO-GENERATED`` insertion, training
        #    the model to "edit .gitignore on every task".
        #    ``_update_gitignore`` is idempotent so the call from
        #    ``get_patch`` later is a no-op.
        await session._update_gitignore(workdir, instance.language)
        await session.execute(
            "git add .gitignore && "
            "git commit --amend --no-edit --allow-empty 2>/dev/null || true",
            cwd=workdir, timeout=30,
        )

        # 6. Collect pip freeze so the prompt can list installed packages
        result = await session.execute("pip freeze", cwd=workdir, timeout=60)
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
            "task_type": "doc2repo",
        }

    def default_evaluator(self, timeout: int | None = None) -> Evaluator:
        """Return a DeNovoSWEEvaluator for this task."""
        from aweagent.tasks.denovo_swe.evaluator import DeNovoSWEEvaluator

        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        kwargs["validate_run"] = self._validate_run
        kwargs["del_done_images"] = self._del_done_images
        kwargs["eval_iters"] = self._eval_iters
        return DeNovoSWEEvaluator(**kwargs)
