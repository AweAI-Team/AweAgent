"""Local runtime backend using the host process and filesystem."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.protocol import Runtime, RuntimeSession
from awe_agent.core.runtime.types import ExecutionResult


class LocalRuntimeSession(RuntimeSession):
    """Runtime session backed by the local host environment."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._config = config
        self._closed = False

    def _resolve_cwd(self, cwd: str | None) -> str:
        candidate = cwd or self._config.workdir or os.getcwd()
        if not os.path.isdir(candidate):
            return os.getcwd()
        return candidate

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-lc",
            command,
            cwd=self._resolve_cwd(cwd),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ExecutionResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s: {command[:200]}",
                exit_code=124,
            )

        return ExecutionResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            exit_code=proc.returncode or 0,
        )

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        path = Path(remote_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    async def download_file(self, remote_path: str) -> bytes:
        path = Path(remote_path)
        if not path.exists():
            raise FileNotFoundError(remote_path)
        return path.read_bytes()

    async def list_files(self, path: str, recursive: bool = False) -> list[str]:
        target = Path(path)
        if not target.exists():
            return []
        if target.is_file():
            return [str(target)]
        if recursive:
            return [str(p) for p in target.rglob("*")]
        return [str(p) for p in target.iterdir()]

    async def close(self) -> None:
        self._closed = True


class LocalRuntime(Runtime):
    """Runtime backed by the local host machine."""

    async def create_session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> RuntimeSession:
        del image, kwargs
        workdir = Path(self.config.workdir or os.getcwd())
        workdir.mkdir(parents=True, exist_ok=True)
        return LocalRuntimeSession(self.config)
