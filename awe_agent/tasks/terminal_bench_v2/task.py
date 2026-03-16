"""Terminal Bench V2 Task — task loading for Terminal-Bench 2.0.

Data source: a directory of task folders (each folder = one instance).
Each task folder contains: instruction.md, task.toml, environment/, tests/
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import Instance

# ── TerminalBenchInstance ────────────────────────────────────────────


@dataclass
class TerminalBenchInstance(Instance):
    """Type-safe Instance subclass for Terminal-Bench 2.0 metadata."""

    instruction: str = ""
    task_dir_path: str = ""
    test_files: dict[str, str] = field(default_factory=dict)
    agent_timeout_sec: float = 600.0
    verifier_timeout_sec: float = 600.0
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240

    @classmethod
    def from_instance(cls, instance: Instance) -> TerminalBenchInstance:
        """Convert a generic Instance to TerminalBenchInstance.

        If already a TerminalBenchInstance, return as-is.
        Otherwise, reconstruct from Instance.metadata (e.g. after JSON deserialization).
        """
        if isinstance(instance, TerminalBenchInstance):
            return instance
        return cls(
            id=instance.id,
            dataset_id=instance.dataset_id,
            repo=instance.repo,
            base_commit=instance.base_commit,
            workdir=instance.workdir,
            image=instance.image,
            language=instance.language,
            metadata=instance.metadata,
            instruction=instance.metadata.get("instruction", ""),
            task_dir_path=instance.metadata.get("task_dir", ""),
            test_files=instance.metadata.get("test_files", {}),
            agent_timeout_sec=instance.metadata.get("agent_timeout_sec", 600.0),
            verifier_timeout_sec=instance.metadata.get(
                "verifier_timeout_sec", 600.0
            ),
            cpus=instance.metadata.get("cpus", 1),
            memory_mb=instance.metadata.get("memory_mb", 2048),
            storage_mb=instance.metadata.get("storage_mb", 10240),
        )

logger = logging.getLogger(__name__)

# PyPI index (override via TERMINAL_BENCH_V2_PYPI_INDEX for restricted networks)
INTERNAL_PYPI_INDEX = os.environ.get(
    "TERMINAL_BENCH_V2_PYPI_INDEX",
    "https://pypi.org/simple",
)

# Forward proxy (optional, from env)
PROXY_ENV = {
    k: v
    for k, v in {
        "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
        "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
        "ALL_PROXY": os.environ.get("ALL_PROXY", ""),
        "NO_PROXY": os.environ.get("NO_PROXY", ""),
    }.items()
    if v
}


def _shell_escape(value: str) -> str:
    """Escape value for use in single-quoted shell string."""
    return value.replace("'", "'\"'\"'")


def _parse_size_to_mb(size_val: str | int | float) -> int:
    """Parse size to MB. Supports diverse formats:

    - int/float: treated as MB
    - G/M/K: binary (1024), e.g. "2G", "512M"
    - Gi/Mi/Ki: binary (Kubernetes-style), e.g. "4Gi", "512Mi"
    - GB/MB/KB: decimal (SI), e.g. "4GB", "512MB"
    """
    if isinstance(size_val, (int, float)):
        return int(size_val)
    s = str(size_val).strip()
    if not s:
        return 0
    num_part = ""
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] in ".-"):
        num_part += s[i]
        i += 1
    suffix = s[i:].upper() if i < len(s) else ""
    try:
        val = float(num_part) if num_part else 0.0
    except ValueError:
        return 0
    if suffix in ("G", "GI"):
        return int(val * 1024)
    if suffix in ("M", "MI"):
        return int(val)
    if suffix in ("K", "KI"):
        return max(0, int(val / 1024))
    if suffix == "GB":
        return int(val * 1000)
    if suffix == "MB":
        return int(val)
    if suffix == "KB":
        return max(0, int(val / 1000))
    return int(val)


# ── task.toml config models ─────────────────────────────────────────


class VerifierConfig(BaseModel):
    timeout_sec: float = 600.0
    env: dict[str, str] = {}


class AgentConfig(BaseModel):
    timeout_sec: float = 600.0


class EnvironmentConfig(BaseModel):
    """Supports cpus, memory, storage for Docker resource limits."""

    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int = 10240
    memory: str | int | None = None
    storage: str | int | None = None

    @model_validator(mode="after")
    def _parse_memory_storage(self) -> EnvironmentConfig:
        if self.memory is not None:
            self.memory_mb = _parse_size_to_mb(self.memory)
            self.memory = None
        if self.storage is not None:
            self.storage_mb = _parse_size_to_mb(self.storage)
            self.storage = None
        return self


class TomlTaskConfig(BaseModel):
    version: str = "1.0"
    metadata: dict[str, Any] = {}
    verifier: VerifierConfig = VerifierConfig()
    agent: AgentConfig = AgentConfig()
    environment: EnvironmentConfig = EnvironmentConfig()


# ── TaskInfo ─────────────────────────────────────────────────────────


@dataclass
class TaskInfo:
    instance_id: str
    task_dir: Path
    instruction: str
    config: TomlTaskConfig
    docker_image: str
    test_files: dict[str, str]

    @classmethod
    def from_directory(cls, task_dir: Path | str) -> TaskInfo:
        task_dir = Path(task_dir).resolve()

        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            raise FileNotFoundError(f"instruction.md not found: {instruction_path}")
        instruction = instruction_path.read_text(encoding="utf-8")

        config_path = task_dir / "task.toml"
        if not config_path.exists():
            raise FileNotFoundError(f"task.toml not found: {config_path}")
        with open(config_path, "rb") as f:
            toml_data = tomllib.load(f)
        config = TomlTaskConfig.model_validate(toml_data)

        docker_image = config.environment.docker_image
        if not docker_image:
            raise ValueError(f"task.toml does not contain docker_image: {config_path}")

        test_files = cls._collect_test_files(task_dir / "tests")

        return cls(
            instance_id=task_dir.name,
            task_dir=task_dir,
            instruction=instruction,
            config=config,
            docker_image=docker_image,
            test_files=test_files,
        )

    @staticmethod
    def _collect_test_files(tests_dir: Path) -> dict[str, str]:
        test_files = {}
        if not tests_dir.exists():
            logger.warning("tests directory not found: %s", tests_dir)
            return test_files
        for file_path in tests_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(tests_dir)
                container_path = f"/tests/{relative_path}"
                try:
                    content = file_path.read_bytes()
                    test_files[container_path] = base64.b64encode(content).decode()
                except Exception as e:
                    logger.warning("Failed to read file %s: %s", file_path, e)
        return test_files


# ── Helpers ──────────────────────────────────────────────────────────


def _get_dockerfile_workdir(task_dir: Path) -> str | None:
    """Parse last WORKDIR from Dockerfile for verifier cwd."""
    for candidate in (task_dir / "environment" / "Dockerfile", task_dir / "Dockerfile"):
        if not candidate.exists():
            continue
        try:
            content = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        pattern = re.compile(r"^\s*WORKDIR\s+(.+)$", re.MULTILINE)
        matches = pattern.findall(content)
        if matches:
            wd = matches[-1].strip().strip("'\"")
            if wd:
                return wd
    return None


def _task_info_to_instance(task_info: TaskInfo, dataset_id: str) -> TerminalBenchInstance:
    """Convert TaskInfo to TerminalBenchInstance with typed TB2 fields."""
    workdir = _get_dockerfile_workdir(task_info.task_dir) or "/app"
    env = task_info.config.environment
    return TerminalBenchInstance(
        id=task_info.instance_id,
        dataset_id=dataset_id,
        image=task_info.docker_image,
        workdir=workdir,
        metadata=task_info.config.metadata,
        instruction=task_info.instruction,
        task_dir_path=str(task_info.task_dir),
        test_files=task_info.test_files,
        agent_timeout_sec=task_info.config.agent.timeout_sec,
        verifier_timeout_sec=task_info.config.verifier.timeout_sec,
        cpus=env.cpus,
        memory_mb=env.memory_mb,
        storage_mb=env.storage_mb,
    )


def list_available_tasks(task_data_dir: str | Path) -> list[str]:
    task_data_dir = Path(task_data_dir)
    tasks = []
    for item in task_data_dir.iterdir():
        if item.is_dir():
            if (item / "task.toml").exists() and (item / "instruction.md").exists():
                tasks.append(item.name)
    return sorted(tasks)


# ── Task ─────────────────────────────────────────────────────────────


class TerminalBenchV2Task(Task):
    """Task implementation for Terminal-Bench 2.0.

    Loads instances from a directory of task folders. Each folder contains:
    - instruction.md, task.toml, environment/, tests/
    """

    def __init__(
        self,
        task_data_dir: str,
        data_file: str | None = None,
        dataset_id: str = "terminal_bench_v2",
    ) -> None:
        self.task_data_dir = Path(task_data_dir).resolve()
        self.data_file = data_file
        self.dataset_id = dataset_id
        self._cache: dict[str, TaskInfo] = {}

    def _get_task_info(self, instance_id: str) -> TaskInfo:
        if instance_id not in self._cache:
            task_dir = self.task_data_dir / instance_id
            self._cache[instance_id] = TaskInfo.from_directory(task_dir)
        return self._cache[instance_id]

    def _load_instance_ids(self) -> list[str]:
        if self.data_file:
            path = Path(self.data_file)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError("data_file must be a JSON array of instance IDs")
        return list_available_tasks(self.task_data_dir)

    def get_instances(
        self, instance_ids: list[str] | None = None
    ) -> list[Instance]:
        all_ids = self._load_instance_ids()
        if instance_ids is not None:
            id_set = set(instance_ids)
            all_ids = [iid for iid in all_ids if iid in id_set]

        instances: list[Instance] = []
        for iid in all_ids:
            try:
                task_info = self._get_task_info(iid)
                instances.append(_task_info_to_instance(task_info, self.dataset_id))
            except Exception as e:
                logger.warning("Skip instance %s: %s", iid, e)
        logger.info(
            "Loaded %d Terminal-Bench V2 instances from %s",
            len(instances), self.task_data_dir,
        )
        return instances

    def get_prompt(self, instance: Instance) -> str:
        """Return the initial prompt with an empty terminal_state placeholder.

        The ``Terminus2Agent`` replaces the user message on its first
        ``step()`` call with a version containing the real terminal state
        (read from ``task_info["prompt_template"]``).  This initial value
        serves only as a structural placeholder for ``AgentLoop.run()``.
        """
        from awe_agent.tasks.terminal_bench_v2.prompt import format_prompt

        inst = TerminalBenchInstance.from_instance(instance)
        return format_prompt(
            instruction=inst.instruction,
            terminal_state="",
        )

    def get_image(self, instance: Instance) -> str:
        return instance.image

    def get_setup_commands(self, instance: Instance) -> list[str]:
        """Terminal Bench: inject PyPI, proxy, install tmux and asciinema.

        Installs tmux, asciinema, and configures PyPI/proxy env vars.

        Environment propagation strategy:
        1. Write all env exports to /root/.awe_agent_env (guard-free).
        2. DockerConfig.environment sets BASH_ENV=/root/.awe_agent_env
           so every bash -c auto-sources the file.
        3. Also source from .bashrc and .profile for tmux login shells.
        """
        env_file = "/root/.awe_agent_env"
        commands: list[str] = []

        env_lines: list[str] = []
        env_lines.append(f"export PIP_INDEX_URL='{INTERNAL_PYPI_INDEX}'")
        if PROXY_ENV:
            for k, v in PROXY_ENV.items():
                escaped = _shell_escape(v)
                env_lines.append(f"export {k}='{escaped}'")
                env_lines.append(f"export {k.lower()}='{escaped}'")

        write_env_cmd = (
            f"cat > {env_file} << 'AWEAGENT_ENV_EOF'\n"
            + "\n".join(env_lines)
            + "\n"
            + "AWEAGENT_ENV_EOF"
        )
        commands.append(write_env_cmd)

        source_line = f". {env_file} 2>/dev/null || true"
        for rcfile in ("/root/.bashrc", "/root/.profile"):
            hook_cmd = (
                f'grep -qsF "{env_file}" {rcfile} 2>/dev/null || '
                f"echo '{source_line}' >> {rcfile} 2>/dev/null || true"
            )
            commands.append(hook_cmd)

        tmux_cmd = (
            "which tmux >/dev/null 2>&1 || ("
            "DEBIAN_FRONTEND=noninteractive apt-get update -qq 2>/dev/null && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y tmux 2>/dev/null"
            ") || (yum install -y tmux 2>/dev/null) || (apk add tmux 2>/dev/null) || "
            "(dnf install -y tmux 2>/dev/null) || true"
        )
        commands.append(tmux_cmd)

        asciinema_cmd = (
            "which asciinema >/dev/null 2>&1 || ("
            "DEBIAN_FRONTEND=noninteractive apt-get install -y asciinema 2>/dev/null"
            ") || (yum install -y asciinema 2>/dev/null) || (apk add asciinema 2>/dev/null) || "
            "(pip3 install asciinema 2>/dev/null) || (pip install asciinema 2>/dev/null) || true"
        )
        commands.append(asciinema_cmd)

        return commands

    def get_task_info(self, instance: Instance) -> dict[str, Any]:
        from awe_agent.tasks.terminal_bench_v2.prompt import get_template

        inst = TerminalBenchInstance.from_instance(instance)
        return {
            "instance_id": inst.id,
            "dataset_id": self.dataset_id,
            "workdir": inst.workdir,
            "instruction": inst.instruction,
            "test_files": inst.test_files,
            "agent_timeout_sec": inst.agent_timeout_sec,
            "verifier_timeout_sec": inst.verifier_timeout_sec,
            "cpus": inst.cpus,
            "memory_mb": inst.memory_mb,
            "storage_mb": inst.storage_mb,
            # The raw prompt template; the agent fills in {terminal_state}
            # at runtime.  This avoids scaffold -> tasks layer coupling.
            "prompt_template": get_template(),
        }

    def get_resource_limits(self, instance: Instance) -> dict[str, str] | None:
        """Per-instance limits from task.toml."""
        inst = TerminalBenchInstance.from_instance(instance)
        return {
            "cpu": str(inst.cpus),
            "memory": f"{inst.memory_mb}Mi",
        }

    def get_docker_environment(self, instance: Instance) -> dict[str, str] | None:
        """Set BASH_ENV so every bash -c auto-sources the env file."""
        return {"BASH_ENV": "/root/.awe_agent_env"}

    def requires_git_snapshot(self) -> bool:
        """Terminal Bench has no git repo in workdir."""
        return False

    def get_search_constraints(self, instance: Instance) -> None:
        return None

    def default_evaluator(self, timeout: int | None = None) -> Evaluator | None:
        try:
            from awe_agent.tasks.terminal_bench_v2.evaluator import (
                TerminalBenchV2Evaluator,
            )
            kwargs = {}
            if timeout is not None:
                kwargs["timeout"] = int(timeout)
            return TerminalBenchV2Evaluator(**kwargs)
        except ImportError:
            return None
