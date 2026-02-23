"""TaskRunner — batch execution engine for running agents on task instances.

Manages concurrency, retry, result collection, and output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentLoop
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.types import EvalResult, Instance, TaskResult
from awe_agent.plugins.registry import Registry

logger = logging.getLogger(__name__)

# Global registry for runtimes
runtime_registry: Registry[type] = Registry("awe_agent.runtime")

# Built-in runtimes (always available, even without pip install -e .)
from awe_agent.core.runtime.docker import DockerRuntime  # noqa: E402

runtime_registry.register("docker", DockerRuntime)


# ── Helper functions for run directory management ────────────────────


def _sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in directory names.

    e.g. ``Qwen/Qwen3.5-397B-A17B`` → ``Qwen3.5-397B-A17B``
    """
    # Take the last component if there's a slash
    name = model.rsplit("/", 1)[-1]
    # Replace any remaining filesystem-unsafe characters
    name = re.sub(r"[^\w\-.]", "_", name)
    return name


def _build_run_dir(base: Path, model: str) -> Path:
    """Build a unique run directory: ``base / {model}_{YYYYMMDD_HHMMSS}``."""
    sanitized = _sanitize_model_name(model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{sanitized}_{timestamp}"


def _save_run_config(
    run_dir: Path,
    config_snapshot: dict[str, Any],
    instances: list[Instance],
    run_id: str,
) -> None:
    """Write ``run_config.json`` with run metadata (excluding secrets)."""
    # Strip sensitive fields
    safe_config = _strip_secrets(config_snapshot)
    data = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config": safe_config,
        "instance_count": len(instances),
        "instance_ids": [inst.id for inst in instances],
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    )


def _strip_secrets(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove keys that look like secrets."""
    secret_keys = {"api_key", "secret", "token", "password", "credentials"}
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k.lower() in secret_keys:
            continue
        if isinstance(v, dict):
            out[k] = _strip_secrets(v)
        else:
            out[k] = v
    return out


def _build_trajectory_record(result: TaskResult) -> dict[str, Any] | None:
    """Build a trajectory dict for a single instance. Returns None if no agent_result."""
    agent_result = result.agent_result
    if agent_result is None:
        return None

    trajectory_steps = []
    for step in agent_result.trajectory.steps:
        trajectory_steps.append({
            "step": step.step,
            "action": {
                "type": step.action.type,
                "content": step.action.content,
                "tool_calls": step.action.tool_calls,
            },
            "observations": step.observations,
        })

    return {
        "instance_id": result.instance_id,
        "success": result.success,
        "score": result.eval_result.score if result.eval_result else 0.0,
        "finish_reason": agent_result.finish_reason,
        "error": result.error,
        "duration": result.metadata.get("duration"),
        "patch": agent_result.patch,
        "stats": agent_result.metadata.get("stats"),
        "trajectory": trajectory_steps,
        "eval_result": asdict(result.eval_result) if result.eval_result else None,
    }


class TaskRunner:
    """Batch execution engine.

    Runs an agent on multiple task instances concurrently,
    with optional evaluation in isolated containers.

    Example:
        runner = TaskRunner(
            task=BeyondSWETask(data_file="data.jsonl"),
            agent_factory=lambda: SearchSWEAgent(enable_search=True),
            llm_config=LLMConfig(backend="openai", model="gpt-4o"),
            runtime_config=RuntimeConfig(backend="docker"),
            max_concurrent=50,
        )
        results = await runner.run_all()
    """

    def __init__(
        self,
        task: Task,
        agent_factory: Callable[..., Agent],
        llm_config: LLMConfig,
        runtime_config: RuntimeConfig,
        evaluator: Evaluator | None = None,
        eval_runtime_config: RuntimeConfig | None = None,
        max_concurrent: int = 50,
        max_retries: int = 3,
        output_path: str | Path = "./results",
        condenser: Any = None,
        save_trajectories: bool = True,
        config_snapshot: dict[str, Any] | None = None,
        max_steps: int = 100,
        max_context_length: int | None = None,
    ) -> None:
        self.task = task
        self.agent_factory = agent_factory
        self.llm_config = llm_config
        self.runtime_config = runtime_config
        self.evaluator = evaluator or task.default_evaluator()
        self.eval_runtime_config = eval_runtime_config
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.max_context_length = max_context_length
        self.output_path = Path(output_path)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._condenser = condenser
        self._save_trajectories = save_trajectories
        self._config_snapshot = config_snapshot
        self.run_dir: Path | None = None  # set in run_all()

    async def run_all(
        self,
        instance_ids: list[str] | None = None,
    ) -> list[TaskResult]:
        """Run agent on all instances concurrently."""
        instances = self.task.get_instances(instance_ids)
        logger.info("Running %d instances (max_concurrent=%d)", len(instances), self.max_concurrent)

        # Build timestamped run directory
        run_dir = _build_run_dir(self.output_path, self.llm_config.model)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        logger.info("Run directory: %s", run_dir)

        # Save run config snapshot
        if self._config_snapshot:
            try:
                _save_run_config(run_dir, self._config_snapshot, instances, run_dir.name)
            except Exception as e:
                logger.warning("Failed to save run config: %s", e)

        # Trajectories file
        traj_file: Path | None = None
        if self._save_trajectories:
            traj_file = run_dir / "trajectories.jsonl"

        output_file = run_dir / "results.jsonl"
        write_lock = asyncio.Lock()

        tasks = [
            self._run_instance_with_retry(inst, output_file, write_lock, traj_file)
            for inst in instances
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Summarize
        completed = [r for r in results if isinstance(r, TaskResult)]
        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in completed if r.success]
        logger.info(
            "Done: %d/%d succeeded, %d errors",
            len(successes), len(completed), len(errors),
        )

        return [r if isinstance(r, TaskResult) else TaskResult(
            instance_id="unknown", error=str(r)
        ) for r in results]

    async def _run_instance_with_retry(
        self,
        instance: Instance,
        output_file: Path,
        write_lock: asyncio.Lock,
        traj_file: Path | None = None,
    ) -> TaskResult:
        """Run a single instance with retry logic."""
        async with self._semaphore:
            last_error: str | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    result = await self._run_instance(instance)

                    # Write result summary + trajectory (under same lock)
                    async with write_lock:
                        with open(output_file, "a") as f:
                            f.write(json.dumps({
                                "instance_id": result.instance_id,
                                "success": result.success,
                                "score": result.eval_result.score if result.eval_result else 0.0,
                                "error": result.error,
                                "finish_reason": result.agent_result.finish_reason if result.agent_result else "",
                            }) + "\n")

                        # Append trajectory to jsonl
                        if traj_file is not None and result.agent_result is not None:
                            try:
                                record = _build_trajectory_record(result)
                                if record is not None:
                                    with open(traj_file, "a") as f:
                                        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                            except Exception as e:
                                logger.warning("Failed to save trajectory for %s: %s", result.instance_id, e)

                    return result

                except Exception as e:
                    last_error = str(e)
                    logger.warning(
                        "Instance %s attempt %d/%d failed: %s",
                        instance.id, attempt, self.max_retries, e,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(attempt * 2)

            return TaskResult(instance_id=instance.id, error=last_error)

    async def _run_instance(self, instance: Instance) -> TaskResult:
        """Run agent + evaluation on a single instance."""
        start_time = time.monotonic()

        # Create runtime
        runtime_cls = runtime_registry.get(self.runtime_config.backend)
        runtime: Runtime = runtime_cls(self.runtime_config)
        image = self.task.get_image(instance)

        async with runtime.session(image) as session:
            # Setup environment
            for cmd in self.task.get_setup_commands(instance):
                result = await session.execute(cmd)
                if not result.success:
                    logger.warning("Setup command failed: %s -> %s", cmd, result.stderr[:200])

            # Create agent
            constraints = self.task.get_search_constraints(instance)
            agent = self.agent_factory(search_constraints=constraints)
            llm = LLMClient(self.llm_config)
            context = AgentContext(
                llm=llm,
                session=session,
                tools=agent.get_tools(),
                task_info=self.task.get_task_info(instance),
                max_steps=self.max_steps,
                max_context_length=self.max_context_length,
                condenser=self._condenser,
            )
            loop = AgentLoop(agent, context)

            # Run agent
            prompt = self.task.get_prompt(instance)
            agent_result = await loop.run(prompt)

        # Evaluate in isolated container (agent session already released)
        eval_result: EvalResult | None = None
        if self.evaluator and agent_result.patch:
            eval_result = await self._evaluate(instance, agent_result.patch)

        elapsed = time.monotonic() - start_time
        return TaskResult(
            instance_id=instance.id,
            agent_result=agent_result,
            eval_result=eval_result,
            metadata={"duration": elapsed},
        )

    async def _evaluate(self, instance: Instance, patch: str) -> EvalResult:
        """Evaluate in an isolated runtime."""
        if not self.evaluator:
            return EvalResult()

        eval_config = self.eval_runtime_config or self.runtime_config
        eval_runtime_cls = runtime_registry.get(eval_config.backend)
        eval_runtime: Runtime = eval_runtime_cls(eval_config)

        return await self.evaluator.evaluate(instance, patch, eval_runtime)
