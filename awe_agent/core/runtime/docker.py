"""Docker runtime backend using the Python docker SDK."""

from __future__ import annotations

import io
import logging
import os
import re
import shlex
import tarfile
import uuid
from typing import Any

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.protocol import Runtime, RuntimeSession
from awe_agent.core.runtime.types import ExecutionResult

logger = logging.getLogger(__name__)


class DockerSession(RuntimeSession):
    """A runtime session backed by a Docker container."""

    def __init__(self, container: Any, config: RuntimeConfig) -> None:
        self._container = container
        self._config = config
        self._closed = False

    async def execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        import asyncio

        workdir = cwd or self._config.workdir

        # Build environment
        exec_env = {}
        if env:
            exec_env.update(env)

        # docker SDK exec_run is synchronous; run in thread
        def _run() -> tuple[int, bytes]:
            result = self._container.exec_run(
                ["bash", "-c", command],
                workdir=workdir,
                environment=exec_env or None,
                demux=True,
            )
            return result.exit_code, result.output

        loop = asyncio.get_running_loop()

        # Per-command timeout (e.g. bash_timeout=120); session TTL is
        # enforced at the protocol layer via asyncio.timeout().
        if timeout is not None:
            try:
                exit_code, output = await asyncio.wait_for(
                    loop.run_in_executor(None, _run),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                # Kill orphan processes inside the container to prevent
                # resource leaks.  The exec_run thread may still be blocked
                # but Docker will terminate the exec once we kill its PID.
                try:
                    self._container.exec_run(
                        ["bash", "-c", "kill -9 -1 2>/dev/null || true"],
                        detach=True,
                    )
                except Exception:
                    pass
                return ExecutionResult(
                    stdout="",
                    stderr=f"Command timed out after {timeout}s: {command[:200]}",
                    exit_code=124,
                )
        else:
            exit_code, output = await loop.run_in_executor(None, _run)

        stdout, stderr = "", ""
        if isinstance(output, tuple):
            stdout = (output[0] or b"").decode("utf-8", errors="replace")
            stderr = (output[1] or b"").decode("utf-8", errors="replace")
        elif isinstance(output, bytes):
            stdout = output.decode("utf-8", errors="replace")

        return ExecutionResult(stdout=stdout, stderr=stderr, exit_code=exit_code)

    async def upload_file(self, remote_path: str, content: bytes) -> None:
        import asyncio

        def _upload() -> None:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo(name=os.path.basename(remote_path))
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
            buf.seek(0)
            self._container.put_archive(os.path.dirname(remote_path) or "/", buf)

        await asyncio.get_running_loop().run_in_executor(None, _upload)

    async def download_file(self, remote_path: str) -> bytes:
        import asyncio

        def _download() -> bytes:
            try:
                bits, _ = self._container.get_archive(remote_path)
            except Exception as exc:
                # docker SDK raises NotFound (404) when file does not exist
                raise FileNotFoundError(f"File not found: {remote_path}") from exc
            buf = io.BytesIO()
            for chunk in bits:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f is None:
                    return b""
                return f.read()

        return await asyncio.get_running_loop().run_in_executor(None, _download)

    async def list_files(self, path: str, recursive: bool = False) -> list[str]:
        flag = "-R " if recursive else ""
        result = await self.execute(f"ls {flag}{shlex.quote(path)}")
        if result.success:
            return [line for line in result.stdout.strip().split("\n") if line]
        return []

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        import asyncio

        def _cleanup() -> None:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
            except Exception:
                logger.warning("Failed to cleanup container %s", self._container.id[:12])

        await asyncio.get_running_loop().run_in_executor(None, _cleanup)
        logger.info("Container %s removed", self._container.id[:12])


class DockerRuntime(Runtime):
    """Runtime backed by local Docker engine."""

    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    async def create_session(
        self,
        image: str | None = None,
        **kwargs: object,
    ) -> RuntimeSession:
        import asyncio

        client = self._get_client()
        img = image or self.config.image
        if not img:
            raise ValueError("No image specified for Docker runtime")

        # Container names only allow [a-zA-Z0-9_.-]
        safe_img = re.sub(r"[^a-zA-Z0-9_.-]", "-", img.rsplit("/", 1)[-1])
        container_name = f"awe-agent-{safe_img}-{uuid.uuid4().hex[:12]}"

        def _create() -> Any:
            # Pull if needed
            if self.config.docker.pull_policy == "always":
                logger.info("Pulling image %s", img)
                client.images.pull(img)
            elif self.config.docker.pull_policy == "if_not_present":
                try:
                    client.images.get(img)
                except Exception:
                    logger.info("Image %s not found locally, pulling", img)
                    client.images.pull(img)

            # Parse resource limits
            mem_str = self.config.resource_limits.memory
            mem_bytes = _parse_memory(mem_str)
            cpu_count = int(float(self.config.resource_limits.cpu))

            container = client.containers.run(
                img,
                command="sleep infinity",
                name=container_name,
                detach=True,
                working_dir=self.config.workdir,
                network=self.config.docker.network,
                mem_limit=mem_bytes,
                nano_cpus=cpu_count * 10**9,
                environment=self.config.docker.environment or None,
                volumes=_parse_volumes(self.config.docker.volumes) or None,
            )
            return container

        loop = asyncio.get_running_loop()
        container = await loop.run_in_executor(None, _create)
        logger.info("Created container %s (%s) from %s", container_name, container.id[:12], img)
        return DockerSession(container, self.config)


def _parse_volumes(volume_specs: list[str]) -> dict[str, dict[str, str]]:
    """Parse volume specs like '/host:/container' or '/host:/container:ro'."""
    volumes = {}
    for v in volume_specs:
        parts = v.split(":")
        if len(parts) >= 2:
            host_path = parts[0]
            container_path = parts[1]
            mode = parts[2] if len(parts) >= 3 else "rw"
            volumes[host_path] = {"bind": container_path, "mode": mode}
    return volumes


def _parse_memory(mem_str: str) -> int:
    """Parse memory string like '8Gi' to bytes."""
    mem_str = mem_str.strip()
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
             "K": 1000, "M": 10**6, "G": 10**9, "T": 10**12}
    for suffix, multiplier in sorted(units.items(), key=lambda x: -len(x[0])):
        if mem_str.endswith(suffix):
            return int(float(mem_str[:-len(suffix)]) * multiplier)
    return int(mem_str)
