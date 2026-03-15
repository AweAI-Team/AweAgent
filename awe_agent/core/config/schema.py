"""Global configuration schema for AweAgent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.runtime.config import RuntimeConfig


class CondenserConfig(BaseModel):
    """Context condensing configuration."""

    type: str = "none"  # "none" | "truncation"
    max_messages: int = 50
    keep_first: int = 2


class AgentConfig(BaseModel):
    """Agent-specific configuration."""

    type: str = "search_swe"
    max_steps: int = 100
    max_context_length: int | None = None
    enable_search: bool = False
    tools: list[str] = Field(default_factory=lambda: ["execute_bash", "str_replace_editor", "think"])
    bash_timeout: int = 180
    bash_max_timeout: int = 600
    max_output_length: int = 32000
    bash_blocklist: list[str] = Field(default_factory=list)
    condenser: CondenserConfig = Field(default_factory=CondenserConfig)
    tool_call_format: str = "openai_function"


class TaskConfig(BaseModel):
    """Task-specific configuration."""

    type: str = "beyond_swe"
    dataset_id: str = "beyond_swe"
    task_type: str = ""
    data_file: str | None = None
    instance_ids: list[str] | None = None
    test_suite_dir: str | None = None
    task_data_dir: str | None = None
    override_agent_timeout: float | None = None


class EvalConfig(BaseModel):
    """Evaluation configuration."""

    enabled: bool = True
    isolated: bool = True
    timeout: int = 3600
    eval_script: str | None = None
    runtime: RuntimeConfig | None = None


class ExecutionConfig(BaseModel):
    """Execution configuration."""

    max_concurrent: int = 50
    max_retries: int = 3
    output_path: str = "./results"
    output_format: str = "jsonl"
    save_trajectories: bool = True


class SecurityConfig(BaseModel):
    """Security configuration.

    Note: core blocklist patterns (git introspection, non-search git fetch)
    are defined in ``SearchSWEAgent`` and applied automatically.  The
    ``bash_blocklist`` here is for *additional* task-specific patterns only.
    """

    bash_blocklist: list[str] = Field(default_factory=list)
    blocked_urls: list[str] = Field(default_factory=list)
    # Search-specific constraint patterns, keyed by field name
    # e.g. {"url": [".*github\\.com/owner/repo.*"], "title": [...]}
    blocked_search_patterns: dict[str, list[str]] = Field(default_factory=dict)


class AweAgentConfig(BaseModel):
    """Top-level configuration for AweAgent.

    This is the master config that controls all behavior.
    Loaded from YAML with env var and CLI overrides.
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Extra fields for custom extensions
    extra: dict[str, Any] = Field(default_factory=dict)
