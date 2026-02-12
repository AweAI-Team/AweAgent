"""Runtime abstraction layer.

Provides pluggable container runtime backends (Docker, K8s, Portal, etc.)

Usage:
    from awe_agent.core.runtime import DockerRuntime, RuntimeConfig

    config = RuntimeConfig(backend="docker", image="python:3.11")
    runtime = DockerRuntime(config)
    async with runtime.session() as session:
        result = await session.execute("python --version")
"""

from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.protocol import Runtime, RuntimeSession
from awe_agent.core.runtime.types import ExecutionResult, FileInfo, RuntimeSessionInfo

__all__ = [
    "ExecutionResult",
    "FileInfo",
    "Runtime",
    "RuntimeConfig",
    "RuntimeSession",
    "RuntimeSessionInfo",
]
