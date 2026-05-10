"""NL2RepoTask — natural-language → repository implementation benchmark.

Faithful port of the original
`NL2RepoBench <https://github.com/multimodal-art-projection/NL2RepoBench>`_
launcher (``openhands/openhands_app.py`` + ``test_data_service.py``) to the
AweAgent task framework, following the same code style as ``BeyondSWETask``.

Key facts ported from the original benchmark:

* **One docker image per project.**  Every NL2Repo instance is launched in
  its own per-task base image, read from the row's ``evaluation_image``
  field (the same image ``post_processor.create_dockerfile`` uses as
  ``FROM``).  Each instance gets its own isolated sandbox container;
  concurrency is handled by the AweAgent ``TaskRunner``'s semaphore
  (mirroring the ``ThreadPoolExecutor`` / ``max_pool_size`` scheduling
  in ``openhands_app.start_openhands``).
* **Workspace** defaults to ``/workspace`` (matches the original
  ``volumes = "{workspace}:/workspace:rw"`` config in
  ``template/config.template.toml``).
* **Spec file** ``start.md`` is uploaded into the workspace inside
  :meth:`prepare_session`, mirroring the ``shutil.copy2(test_data.md, ...)``
  step in ``openhands_app.start_app``.

Data source
-----------

Each instance is one row with the following fields (matches the upstream
NL2RepoBench CSV schema):

    evaluation_image   — per-instance docker image (used for eval *and*
                         the agent by default)
    instance_id        — unique id / pro_name
    package_name       — Python package name (optional; defaults to instance_id)
    start_instruction  — the natural-language spec uploaded as ``start.md``
    test_cases_num     — expected pytest case count
    verify_cmd         — JSON list of shell commands for evaluation
    verify_files       — JSON list of golden test paths to restore pre-eval

Pass the data via ``data_file`` (constructor) — JSONL or CSV are both
accepted.  For programmatic callers, an ``instances`` list of raw dicts
is also supported.

Image selection
---------------

* ``data_file`` (env ``DATA_FILE``) — required unless ``instances``
  is passed.  Each row's ``evaluation_image`` field is the authoritative
  per-instance eval image, stored in ``Instance.metadata["eval_image"]``
  and used by :class:`NL2RepoEvaluator`.

* ``agent_run_docker`` (env ``AGENT_RUN_DOCKER``) — if non-empty, *every*
  instance's agent container uses this single image regardless of the
  data file.  If empty (the default) the agent runs in the per-instance
  ``evaluation_image``, i.e. agent and evaluator share the container
  image.

The evaluator always substitutes ``metadata["eval_image"]`` for
``Instance.image`` before running its tar-artifact evaluation lifecycle,
so the agent-side override can never affect the evaluation image.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import posixpath
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance
from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.scaffold.search_swe.prompts.config import resolve_prompt_keys
from awe_agent.tasks.nl2repo.constraints import NL2RepoSearchConstraints

if TYPE_CHECKING:
    from awe_agent.core.runtime.protocol import RuntimeSession

logger = logging.getLogger(__name__)

# Workspace path baked into NL2RepoBench's config.template.toml
_DEFAULT_WORKDIR = "/workspace"


_PYTEST_OPTION_TAKES_VALUE = {
    "-k",
    "-m",
    "-n",
    "--basetemp",
    "--confcutdir",
    "--cov",
    "--cov-config",
    "--cov-report",
    "--deselect",
    "--durations",
    "--ignore",
    "--ignore-glob",
    "--junitxml",
    "--maxfail",
    "--rootdir",
    "--tb",
}


def _workspace_relative_path(entry: str, workdir: str = _DEFAULT_WORKDIR) -> str | None:
    """Normalize a verify_files entry as a safe workspace-relative path."""
    raw = str(entry).strip()
    if not raw:
        return None

    # Pytest node IDs are path-like before ``::``.
    raw = raw.split("::", 1)[0]

    normalized_workdir = posixpath.normpath(workdir or _DEFAULT_WORKDIR)
    if raw.startswith("/"):
        raw = posixpath.normpath(raw)
        if raw == normalized_workdir:
            return None
        prefix = f"{normalized_workdir.rstrip('/')}/"
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        else:
            logger.warning("Skipping absolute path outside workspace: %r", entry)
            return None

    rel = posixpath.normpath(raw.lstrip("/"))
    if not rel or rel == "." or rel == ".." or rel.startswith("../"):
        return None
    return rel


def _pytest_targets_from_command(command: str) -> list[str]:
    """Extract explicit pytest path targets from a verify_cmd shell command."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    start = None
    for i, token in enumerate(tokens):
        name = posixpath.basename(token)
        if name in {"pytest", "py.test"}:
            start = i + 1
            break
        if (
            name.startswith("python")
            and i + 2 < len(tokens)
            and tokens[i + 1] == "-m"
            and tokens[i + 2] == "pytest"
        ):
            start = i + 3
            break

    if start is None:
        return []

    targets: list[str] = []
    skip_next = False
    for token in tokens[start:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--":
            continue
        if token.startswith("-"):
            if "=" not in token and token in _PYTEST_OPTION_TAKES_VALUE:
                skip_next = True
            continue
        if "=" in token and "/" not in token and not token.endswith(".py"):
            continue
        targets.append(token)
    return targets


def _looks_like_test_path(relpath: str) -> bool:
    """Return True for pytest targets safe to strip from eval staging."""
    parts = [p for p in posixpath.normpath(relpath).split("/") if p]
    if not parts:
        return False
    basename = parts[-1]
    return (
        any(p in {"test", "tests", "testing"} for p in parts)
        or basename.startswith("test")
        or basename.endswith("_test.py")
        or basename.endswith("_tests.py")
    )


def _test_cleanup_relpaths(
    test_shell: list[str],
    py_test_file_list: list[str],
    workdir: str = _DEFAULT_WORKDIR,
    include_broad_pytest_targets: bool = True,
) -> list[str]:
    """Build the workspace-relative test paths that should be hidden."""
    paths: list[str] = []
    for entry in py_test_file_list:
        rel = _workspace_relative_path(str(entry), workdir)
        if rel is not None:
            paths.append(rel)

    for command in test_shell:
        for entry in _pytest_targets_from_command(str(command)):
            rel = _workspace_relative_path(str(entry), workdir)
            if rel is None:
                continue
            if include_broad_pytest_targets or _looks_like_test_path(rel):
                paths.append(rel)

    return list(dict.fromkeys(paths))


def _coerce_list_field(raw: Any, instance_id: str, field: str) -> list[str]:
    """Decode a list-shaped field, accepting JSON-encoded strings or native lists."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if not isinstance(raw, str):
        return [str(raw)]
    raw = raw.strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(
            "Instance %s has malformed JSON in %s field: %r",
            instance_id, field, raw[:200],
        )
        return []
    if not isinstance(value, list):
        logger.warning(
            "Instance %s: %s is not a list (got %s)",
            instance_id, field, type(value).__name__,
        )
        return []
    return [str(v) for v in value]


class NL2RepoTask(Task):
    """Task implementation for the NL2RepoBench benchmark.

    Each instance corresponds to one project in NL2RepoBench.  The agent
    receives a natural-language specification (``start.md``) inside an
    isolated docker container that already has the project's runtime
    dependencies installed.

    Args:
        dataset_id: Dataset identifier (default ``"nl2repo"``).
        data_file: Path to a JSONL or CSV file containing the per-instance
            rows (``nl2repo_data_info.csv`` shape).  Required unless
            ``instances`` is passed.  When ``None`` the ``DATA_FILE``
            environment variable is consulted.
        agent_run_docker: Optional single image used for every agent
            container.  When ``None`` the ``AGENT_RUN_DOCKER`` environment
            variable is consulted.  An empty string means "run the agent
            in the same image as the evaluator".
        instances: Raw instance dicts for programmatic use; bypasses
            ``data_file``.
        workdir: Container working directory.  Defaults to ``/workspace``,
            matching ``template/config.template.toml`` in NL2RepoBench.
    """

    def __init__(
        self,
        dataset_id: str = "nl2repo",
        data_file: str | None = None,
        agent_run_docker: str | None = None,
        instances: list[dict[str, Any]] | None = None,
        workdir: str = _DEFAULT_WORKDIR,
    ) -> None:
        self.dataset_id = dataset_id
        self.data_file = data_file or os.environ.get("DATA_FILE", "") or None
        # Empty string means "no override"; fall back to the env var when
        # the constructor arg was omitted (None).
        env_agent = os.environ.get("AGENT_RUN_DOCKER", "")
        self.agent_run_docker = (
            agent_run_docker if agent_run_docker is not None else env_agent
        ) or ""
        self._raw_instances = instances
        self._workdir = workdir
        self._loaded: list[dict[str, Any]] | None = None

    # ── Data loading ──────────────────────────────────────────────────

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
            if path.suffix == ".jsonl":
                data = self._load_from_jsonl(path)
            elif path.suffix == ".csv":
                data = self._load_from_csv(path)
            else:
                raise ValueError(
                    f"Unsupported NL2Repo data source: {self.data_file}. "
                    "Expected a .jsonl or .csv file."
                )
            logger.info(
                "Loaded %d NL2Repo instances from %s", len(data), self.data_file,
            )
            self._loaded = data
            return self._loaded

        raise ValueError(
            "No data source configured. Set DATA_FILE (or pass data_file=...) "
            "to a NL2RepoBench JSONL/CSV, or pass instances=... programmatically."
        )

    def _load_from_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Load instances from a JSONL file with NL2RepoBench fields per row."""
        data: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping invalid JSONL row %s:%d: %s", path, line_no, exc,
                    )
                    continue
                normalized = self._normalize_row(row)
                if normalized is not None:
                    data.append(normalized)
        return data

    def _load_from_csv(self, path: Path) -> list[dict[str, Any]]:
        """Load instances from an NL2RepoBench-style CSV.

        Expected columns (``nl2repo_data_info.csv`` layout):

            evaluation_image, instance_id, package_name,
            start_instruction, test_cases_num,
            verify_cmd, verify_files

        ``verify_cmd`` / ``verify_files`` are stored as JSON-encoded list
        strings and map to ``test_shell`` / ``py_test_file_list`` in the
        internal schema.  A UTF-8 BOM on the first header is tolerated
        via ``encoding='utf-8-sig'``.
        """
        data: list[dict[str, Any]] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = self._normalize_row(row)
                if normalized is not None:
                    data.append(normalized)
        return data

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Coerce a raw NL2RepoBench row into the internal schema."""
        instance_id = str(row.get("instance_id") or "").strip()
        if not instance_id:
            return None
        eval_image = str(row.get("evaluation_image") or row.get("eval_image") or "").strip()
        if not eval_image:
            logger.warning(
                "Instance %s has empty evaluation_image; skipping", instance_id,
            )
            return None
        package_name = str(row.get("package_name") or instance_id).strip()
        start_md = row.get("start_instruction") or row.get("start_md") or ""

        try:
            test_case_count = int(str(row.get("test_cases_num") or "0").strip() or 0)
        except ValueError:
            logger.warning(
                "Instance %s has non-integer test_cases_num=%r",
                instance_id, row.get("test_cases_num"),
            )
            test_case_count = 0

        test_shell = _coerce_list_field(
            row.get("verify_cmd") if row.get("verify_cmd") is not None else row.get("test_shell"),
            instance_id, "verify_cmd",
        )
        py_test_file_list = _coerce_list_field(
            row.get("verify_files")
            if row.get("verify_files") is not None
            else row.get("py_test_file_list"),
            instance_id, "verify_files",
        )

        return {
            "instance_id": instance_id,
            "pro_name": package_name or instance_id,
            "eval_image": eval_image,
            "workdir": self._workdir,
            "start_md": start_md,
            "test_case_count": test_case_count,
            "test_shell": test_shell,
            "py_test_file_list": py_test_file_list,
        }

    def _to_instance(self, raw: dict[str, Any]) -> Instance:
        pro_name = raw.get("pro_name") or raw.get("instance_id", "")
        instance_id = raw.get("instance_id") or pro_name

        # Evaluation image: authoritative value from the data row.  For
        # programmatic callers passing raw ``instances=[{...}]`` we also
        # accept a plain ``image`` key for convenience.
        eval_image = raw.get("eval_image") or raw.get("image") or ""
        if not eval_image:
            raise ValueError(
                f"Instance {instance_id!r} has no evaluation image. "
                "Provide evaluation_image in the data row (or eval_image/image "
                "when using instances=...)."
            )

        # Agent image: AGENT_RUN_DOCKER overrides everything; empty means
        # "agent shares the evaluation image".
        agent_image = self.agent_run_docker or eval_image

        workdir = raw.get("workdir") or self._workdir

        return Instance(
            id=instance_id,
            dataset_id=self.dataset_id,
            repo=pro_name,
            base_commit="",  # NL2Repo starts from a clean workspace, no base commit
            workdir=workdir,
            image=agent_image,
            language="python",
            problem_statement="",  # carried via start.md upload, not the prompt
            gold_patch="",
            setup_commands=[],
            metadata={
                "task_type": "nl2repo",
                "pro_name": pro_name,
                "eval_image": eval_image,
                "start_md": raw.get("start_md", ""),
                "test_case_count": int(raw.get("test_case_count", 0) or 0),
                "test_shell": list(raw.get("test_shell", []) or []),
                "py_test_file_list": list(raw.get("py_test_file_list", []) or []),
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
        """Resolve the user prompt via the scaffold route table.

        We deliberately go through ``resolve_prompt_keys`` /
        ``get_user_prompt`` (rather than returning the constant directly)
        so that NL2Repo follows the same routing convention as
        :class:`BeyondSWETask`.  The route is registered in
        :mod:`awe_agent.scaffold.search_swe.prompts.config` as
        ``("nl2repo", "nl2repo", False) -> ("beyondswe", "nl2repo")``.

        The original NL2RepoBench prompt has **no** ``{}`` placeholders, so
        we return it verbatim — calling ``.format(...)`` on it would just
        echo the string back unchanged anyway.
        """
        from awe_agent.scaffold.search_swe.prompts.user import get_user_prompt

        task_type = instance.metadata.get("task_type", "nl2repo")
        _, user_key = resolve_prompt_keys(
            dataset_id=self.dataset_id,
            task_type=task_type,
            search=False,  # NL2RepoBench has no search-mode variant.
        )
        return get_user_prompt(user_key)

    def get_llm_overrides(self, instance: Instance) -> dict[str, Any]:
        # NL2Repo asks the agent to author a whole repository — give it the
        # same generous output budget that ``BeyondSWETask`` grants to its
        # ``doc2repo`` task type.
        return {"max_completion_tokens": 16384}

    def get_search_constraints(self, instance: Instance) -> SearchConstraints | None:
        """Build anti-hack :class:`SearchConstraints` for this instance.

        The default implementation on :class:`Task` returns
        ``SearchConstraints.from_repo(instance.repo)`` which only blocks
        URLs + git fetches containing the repo slug.  For NL2Repo that's
        not enough: a motivated agent can ``pip install <pkg>`` and copy
        the installed source verbatim, inflating the score to 1.0
        without implementing anything.

        We therefore build :class:`NL2RepoSearchConstraints` from the
        ``package_name`` (and ``instance_id`` as a fallback), which fans
        out every sensible name variant (hyphen ↔ underscore, VCS URL
        parsing, etc.) and adds strict regex blocks for ``pip`` /
        ``uv`` / ``poetry`` / ``conda`` / ``pipx`` install & download,
        ``git clone``, ``curl``/``wget`` of PyPI hosts or the package
        name, and common ``python -c 'import …; inspect.getsource…'``
        introspection shortcuts.

        These patterns get concatenated with the agent's other bash
        blocklists inside :meth:`SearchSWEAgent.__init__`, so the check
        happens in :class:`ExecuteBashTool` **before** the command
        reaches the sandbox.
        """
        package_name = instance.metadata.get("pro_name") or instance.repo or ""
        if not package_name:
            return None
        return NL2RepoSearchConstraints.from_package_name(
            package_name=package_name,
            instance_id=instance.id,
        )

    def get_image(self, instance: Instance) -> str:
        return instance.image

    def get_setup_commands(self, instance: Instance) -> list[str]:
        """Ensure the workspace directory exists.

        NL2RepoBench workspaces start empty and this task does not use
        ``git`` — :meth:`requires_git_snapshot` returns ``False`` so the
        runner skips the ``git`` commit / rev-parse steps.

        When the agent is run directly inside the per-instance evaluation
        image, that image can already contain the golden unit tests under
        ``workdir``.  The original NL2RepoBench agent run mounts a fresh host
        workspace over ``/workspace``, so the agent never sees those tests.
        Mirror that property by deleting the ``verify_files`` paths plus
        explicit pytest targets parsed from ``verify_cmd`` from the agent-side
        workspace before ``start.md`` is uploaded.  The
        evaluator still receives the untouched ``py_test_file_list`` metadata
        and runs in a fresh eval image, so the golden tests remain available
        for scoring and agent-submitted test files are still stripped before
        overlay.
        """
        workdir = instance.workdir.rstrip("/") or "/workspace"
        commands = [
            f"mkdir -p {shlex.quote(workdir)}",
            *list(instance.setup_commands),
        ]

        test_paths = _test_cleanup_relpaths(
            test_shell=list(instance.metadata.get("test_shell", []) or []),
            py_test_file_list=list(instance.metadata.get("py_test_file_list", []) or []),
            workdir=workdir,
            include_broad_pytest_targets=True,
        )
        if test_paths:
            paths = [f"{workdir}/{rel}" for rel in test_paths]
            quoted = " ".join(shlex.quote(p) for p in paths)
            commands.append(f"rm -rf {quoted}")

        return commands

    def requires_git_snapshot(self) -> bool:
        """Disable git-based pre-agent snapshotting.

        The NL2Repo submission is the whole workspace captured as a
        tar.gz in :meth:`collect_artifact`, not a git patch — and the
        sandbox image may not even ship ``git``.  Returning ``False``
        tells the runner to skip ``git commit`` / ``git rev-parse HEAD``
        entirely.
        """
        return False

    def requires_patch_extraction(self) -> bool:
        """Disable patch extraction.

        NL2Repo submits a tar.gz of the whole workspace (see
        :meth:`collect_artifact`), not a git diff.  Returning ``False``
        tells the runner to set ``skip_patch_extraction`` in
        ``task_info`` so neither the agent loop nor the runner's
        timeout-recovery path runs ``git`` commands on a sandbox that
        may not even ship git.
        """
        return False

    async def prepare_session(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> None:
        """Drop ``start.md`` into the workspace.

        Mirrors the original ``openhands_app.start_app`` step::

            md_filename = os.path.basename(test_data.md)
            dest_md_path = os.path.join(workspace_path, md_filename)
            shutil.copy2(test_data.md, dest_md_path)

        Every NL2RepoBench instance ships its spec under the file name
        ``start.md`` (verified across all 104 projects), so we always
        upload to ``{workdir}/start.md``.  No git commit step here — the
        submission goes out as a tarball via :meth:`collect_artifact`.
        """
        start_md = instance.metadata.get("start_md", "")
        if not start_md:
            logger.warning("Instance %s has no start.md content", instance.id)
            return
        await session.upload_file(
            f"{instance.workdir.rstrip('/')}/start.md", start_md.encode(),
        )
        logger.info("Uploaded start.md for %s", instance.id)

    async def collect_artifact(
        self,
        instance: Instance,
        session: RuntimeSession,
    ) -> bytes | None:
        """Tar the agent's workspace and return the bytes.

        Mirrors swalm-core's ``run_nl2repo_task``::

            await portal_client.execute_shell(
                'tar -czvf workspace.tar.gz workspace', cwd='/openhands',
            )
            file = await portal_client.download_files([
                '/openhands/workspace.tar.gz',
            ])

        We tar ``workdir`` (e.g. ``/workspace``) into ``/tmp/workspace.tar.gz``
        with the basename as the top-level entry, so the evaluator can
        extract with ``tar -xzf`` and get ``/tmp/workspace/...`` — the
        same directory layout the official ``COPY workspace /workspace``
        Dockerfile step produces.
        """
        workdir = instance.workdir.rstrip("/")
        parent = os.path.dirname(workdir) or "/"
        basename = os.path.basename(workdir) or "workspace"
        archive_path = "/tmp/workspace.tar.gz"

        result = await session.execute(
            f"tar -czf {shlex.quote(archive_path)} "
            f"-C {shlex.quote(parent)} {shlex.quote(basename)}",
            cwd=parent,
            timeout=600,
        )
        if result.exit_code != 0:
            logger.error(
                "tar failed for %s (exit %d): %s",
                instance.id, result.exit_code, result.stderr[:400],
            )
            return None

        try:
            return await session.download_file(archive_path)
        except Exception as e:
            logger.error("download_file failed for %s: %s", instance.id, e)
            return None

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        return {
            "instance_id": instance.id,
            "dataset_id": instance.dataset_id,
            "repo": instance.repo,
            "workdir": instance.workdir,
            "language": instance.language,
            "task_type": instance.metadata.get("task_type", "nl2repo"),
            "pro_name": instance.metadata.get("pro_name", ""),
        }

    def default_evaluator(self, timeout: int | None = None) -> Evaluator:
        """Return an :class:`NL2RepoEvaluator` for this task."""
        from awe_agent.tasks.nl2repo.evaluator import NL2RepoEvaluator

        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return NL2RepoEvaluator(**kwargs)
