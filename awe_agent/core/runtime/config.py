"""Runtime configuration models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResourceLimits(BaseModel):
    """Resource limits for a runtime container."""

    cpu: str = "4"
    memory: str = "8Gi"


class DockerConfig(BaseModel):
    """Docker-specific configuration."""

    socket: str = "unix:///var/run/docker.sock"
    network: str = "bridge"
    volumes: list[str] = Field(default_factory=list)
    pull_policy: str = "if_not_present"  # always | if_not_present | never


class K8sConfig(BaseModel):
    """Kubernetes-specific configuration."""

    namespace: str = "awe-agent"
    service_account: str = "default"
    storage_class: str = "standard"
    kubeconfig: str | None = None


class RuntimeConfig(BaseModel):
    """Complete runtime configuration.

    Example YAML:
        runtime:
          backend: docker
          image: "ghcr.io/awe-agent/swe-bench:latest"
          timeout: 7200
          resource_limits:
            cpu: "4"
            memory: "8Gi"
          docker:
            network: bridge
    """

    backend: str = "docker"
    image: str = ""
    timeout: int = 7200  # Session TTL in seconds
    workdir: str = "/testbed"
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)

    # Backend-specific configs
    docker: DockerConfig = Field(default_factory=DockerConfig)
    k8s: K8sConfig = Field(default_factory=K8sConfig)

    # Extra backend-specific args
    extra: dict[str, Any] = Field(default_factory=dict)
