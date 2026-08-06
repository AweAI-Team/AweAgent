"""TaskRunner — batch execution engine for running agents on task instances.

Manages concurrency, retry, result collection, and output.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from aweagent.core.agent.context import AgentContext
from aweagent.core.agent.loop import AgentResult
from aweagent.core.agent.protocol import Agent
from aweagent.core.eval.setup import PreAgentSetup
from aweagent.core.llm.client import LLMClient
from aweagent.core.llm.config import LLMConfig
from aweagent.core.runtime.config import RuntimeConfig
from aweagent.core.runtime.protocol import Runtime
from aweagent.core.task.error_kind import infer_error_kind
from aweagent.core.task.protocol import Evaluator, Task
from aweagent.core.task.types import EvalResult, ErrorKind, Instance, TaskResult
from aweagent.plugins.registry import Registry

logger = logging.getLogger(__name__)

# Global registry for runtimes
runtime_registry: Registry[type] = Registry("aweagent.runtime")

# Built-in runtimes (always available, even without pip install -e .)
from aweagent.core.runtime.docker import DockerRuntime  # noqa: E402

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
        action_dict: dict[str, Any] = {
            "type": step.action.type,
            "content": step.action.content,
            "tool_calls": step.action.tool_calls,
        }
        if step.action.reasoning_text is not None:
            action_dict["reasoning_text"] = step.action.reasoning_text
        step_dict: dict[str, Any] = {
            "step": step.step,
            "action": action_dict,
            "observations": step.observations,
        }
        trajectory_steps.append(step_dict)

    record = {
        "instance_id": result.instance_id,
        "success": result.success,
        "score": result.eval_result.score if result.eval_result else 0.0,
        "finish_reason": agent_result.finish_reason,
        "error": result.error,
        "duration": result.metadata.get("duration"),
        "patch": agent_result.patch,
        "stats": agent_result.metadata.get("stats"),
        "trajectory": trajectory_steps,
        "messages": [m.to_full_dict() for m in agent_result.messages],
        "eval_result": asdict(result.eval_result) if result.eval_result else None,
    }
    # Text-answer scaffolds record their submission and its provenance here;
    # patch tasks never set these keys, so the record stays scaffold-agnostic.
    for key in ("final_answer", "answer_provenance"):
        if key in agent_result.metadata:
            record[key] = agent_result.metadata[key]
    return record


def select_instances(
    instances: list[Instance],
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    max_instances: int | None = None,
) -> list[Instance]:
    """Apply global instance range/cap after task loading and ID filtering."""
    if start_index is not None and start_index < 0:
        raise ValueError("start_index must be greater than or equal to 0")
    if end_index is not None and end_index < 0:
        raise ValueError("end_index must be greater than or equal to 0")
    if max_instances is not None and max_instances <= 0:
        raise ValueError("max_instances must be greater than 0")

    start = start_index if start_index is not None else 0
    if end_index is not None and end_index < start:
        raise ValueError("end_index must be greater than or equal to start_index")

    total = len(instances)
    if total == 0:
        if start_index is not None or end_index is not None:
            logger.warning(
                "Instance range start_index=%s end_index=%s was set but no "
                "instances are available; running 0 instances",
                start_index,
                end_index,
            )
        if max_instances is not None:
            logger.warning(
                "max_instances=%d exceeds 0 selected instance(s); running 0",
                max_instances,
            )
        return []

    # Range selection is global and happens after task-specific loading/ID
    # filtering, so every task keeps its own data parsing semantics.
    if start >= total:
        logger.warning(
            "start_index=%d is beyond %d loaded instance(s); running 0 instances",
            start,
            total,
        )
        selected: list[Instance] = []
    else:
        end = end_index if end_index is not None else total - 1
        if end >= total:
            logger.warning(
                "end_index=%d exceeds last loaded index %d; clipping to %d",
                end,
                total - 1,
                total - 1,
            )
            end = total - 1
        selected = instances[start : end + 1]

    if max_instances is not None and max_instances > len(selected):
        logger.warning(
            "max_instances=%d exceeds %d selected instance(s); running %d",
            max_instances,
            len(selected),
            len(selected),
        )
    if max_instances is not None:
        selected = selected[:max_instances]

    return selected


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
        start_index: int | None = None,
        end_index: int | None = None,
        max_instances: int | None = None,
        max_retries: int = 3,
        output_path: str | Path = "./results",
        condenser: Any = None,
        save_trajectories: bool = True,
        config_snapshot: dict[str, Any] | None = None,
        max_steps: int = 100,
        max_context_length: int | None = None,
        workspace_output_dir: str | None = None,
        agent_timeout_override: float | None = None,
        num_rollouts: int = 1,
    ) -> None:
        self.task = task
        self.agent_factory = agent_factory
        self.llm_config = llm_config
        self.runtime_config = runtime_config
        self.evaluator = evaluator or task.default_evaluator()
        self.eval_runtime_config = eval_runtime_config
        self.max_concurrent = max_concurrent
        self.start_index = start_index
        self.end_index = end_index
        self.max_instances = max_instances
        self.max_retries = max_retries
        self.max_steps = max_steps
        self.max_context_length = max_context_length
        self.output_path = Path(output_path)
        self.workspace_output_dir = workspace_output_dir
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._condenser = condenser
        self._save_trajectories = save_trajectories
        self._config_snapshot = config_snapshot
        self._agent_timeout_override = agent_timeout_override
        self.num_rollouts = num_rollouts
        self.run_dir: Path | None = None  # set in run_all()

    async def run_all(
        self,
        instance_ids: list[str] | None = None,
    ) -> list[TaskResult]:
        """Run agent on all instances concurrently."""
        instances = select_instances(
            self.task.get_instances(instance_ids),
            start_index=self.start_index,
            end_index=self.end_index,
            max_instances=self.max_instances,
        )
        logger.info(
            "Running %d rollouts (%d instances x %d rollouts, max_concurrent=%d)",
            len(instances) * self.num_rollouts, len(instances),
            self.num_rollouts, self.max_concurrent,
        )

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

        num_rollouts = self.num_rollouts
        write_lock = asyncio.Lock()

        # INVARIANT: num_rollouts>1 runs N copies of each instance concurrently.
        # deepcopy isolates only the Instance (its metadata is mutated in
        # _run_instance); self.evaluator, self._condenser, and factory-produced
        # agents must be stateless / fresh-per-call.
        def _rollout_paths(k: int) -> tuple[Path, Path | None]:
            base = run_dir if num_rollouts == 1 else (run_dir / f"rollout_{k}")
            if num_rollouts > 1:
                base.mkdir(parents=True, exist_ok=True)
            results_f = base / "results.jsonl"
            traj_f = (base / "trajectories.jsonl") if self._save_trajectories else None
            return results_f, traj_f

        tasks = []
        for k in range(num_rollouts):
            results_f, traj_f = _rollout_paths(k)
            for inst in instances:
                inst_arg = copy.deepcopy(inst) if num_rollouts > 1 else inst
                tasks.append(
                    self._run_instance_with_retry(
                        inst_arg, results_f, write_lock, traj_f, rollout=k
                    )
                )
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
        rollout: int = 0,
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
                                **({"rollout": rollout} if self.num_rollouts > 1 else {}),
                                "dataset_id": instance.dataset_id,
                                "task": instance.metadata.get("task_type", ""),
                                "success": result.success,
                                "score": result.eval_result.score if result.eval_result else 0.0,
                                "error": result.error,
                                "finish_reason": (
                                    result.agent_result.finish_reason
                                    if result.agent_result
                                    else ""
                                ),
                                "error_kind": (
                                    result.eval_result.error_kind
                                    if result.eval_result is not None
                                    else infer_error_kind(
                                        finish_reason=(
                                            result.agent_result.finish_reason
                                            if result.agent_result
                                            else None
                                        ),
                                        eval_result=None,
                                        task_error=result.error,
                                    )
                                ),
                            }) + "\n")

                        # Append trajectory to jsonl
                        if traj_file is not None and result.agent_result is not None:
                            try:
                                record = _build_trajectory_record(result)
                                if record is not None:
                                    with open(traj_file, "a") as f:
                                        f.write(
                                            json.dumps(
                                                record,
                                                ensure_ascii=False,
                                                default=str,
                                            )
                                            + "\n"
                                        )
                            except Exception as e:
                                logger.warning(
                                    "Failed to save trajectory for %s: %s",
                                    result.instance_id,
                                    e,
                                )

                    return result

                except Exception as e:
                    last_error = f"[{type(e).__name__}] {e}"
                    logger.warning(
                        "Instance %s attempt %d/%d failed: [%s] %s",
                        instance.id, attempt, self.max_retries,
                        type(e).__name__, e,
                        exc_info=True,
                    )
                    if attempt < self.max_retries:
                        await asyncio.sleep(attempt * 2)

            # All retries exhausted — record an error row so the failure is
            # visible in ``results.jsonl`` rather than silently swallowed.
            logger.error(
                "Instance %s failed after %d retries: %s",
                instance.id, self.max_retries, last_error,
            )
            error_result = TaskResult(instance_id=instance.id, error=last_error)
            async with write_lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps({
                        "instance_id": instance.id,
                        **({"rollout": rollout} if self.num_rollouts > 1 else {}),
                        "dataset_id": instance.dataset_id,
                        "task": instance.metadata.get("task_type", ""),
                        "success": False,
                        "score": 0.0,
                        "error": last_error,
                        "finish_reason": "error",
                        "error_kind": ErrorKind.INFRA_ERROR.value,
                    }) + "\n")
            return error_result

    @asynccontextmanager
    async def _instance_session(self, instance: Instance):
        """Yield a runtime session for the instance, or ``None`` when the task
        runs no commands (text-answer tasks open no container)."""
        if not self.task.requires_runtime():
            yield None
            return
        runtime_config = self._instance_runtime_config(instance)
        runtime_cls = runtime_registry.get(runtime_config.backend)
        runtime: Runtime = runtime_cls(runtime_config)
        async with runtime.session(self.task.get_image(instance)) as session:
            yield session

    async def _run_instance(self, instance: Instance) -> TaskResult:
        """Run agent + evaluation on a single instance."""
        start_time = time.monotonic()

        # Determine if evaluation must happen inside the agent session
        # (e.g. Terminal Bench modifies container state directly).
        same_session_eval = (
            self.evaluator is not None
            and self.evaluator.requires_same_session
        )

        eval_result: EvalResult | None = None
        artifact_bytes: bytes | None = None

        async with self._instance_session(instance) as session:
            # Session-backed preparation (skipped when the task needs no runtime).
            pre_agent_commit_id: str | None = None
            if session is not None:
                setup = PreAgentSetup(session, instance.workdir)
                await setup.run_setup_commands(self.task.get_setup_commands(instance))

                # Git snapshot (only for tasks that use git repos).
                if self.task.requires_git_snapshot():
                    pre_agent_commit_id = await setup.commit_and_get_id()
                    await setup.remove_future_commits()

                # Task-specific session preparation (e.g. upload files, pip freeze)
                await self.task.prepare_session(instance, session)

            # Create agent
            constraints = self.task.get_search_constraints(instance)
            agent = self.agent_factory(search_constraints=constraints)
            llm_overrides = self.task.get_llm_overrides(instance)
            if llm_overrides:
                llm_config = self.llm_config.model_copy(
                    update={"params": {**self.llm_config.params, **llm_overrides}}
                )
            else:
                llm_config = self.llm_config
            llm = LLMClient(llm_config)
            task_info = self.task.get_task_info(instance)
            if pre_agent_commit_id:
                task_info["pre_agent_commit_id"] = pre_agent_commit_id
            if not self.task.requires_patch_extraction():
                task_info["skip_patch_extraction"] = True
            if self._agent_timeout_override is not None:
                task_info["agent_timeout_sec"] = float(self._agent_timeout_override)
            context = AgentContext(
                llm=llm,
                session=session,
                tools=agent.get_tools(),
                task_info=task_info,
                max_steps=self.max_steps,
                max_context_length=self.max_context_length,
                condenser=self._condenser,
            )
            # The agent supplies its loop, carrying any lifecycle policy.
            loop = agent.create_loop(context)

            # Run agent (with optional per-instance wall-clock timeout).
            prompt = self.task.get_prompt(instance)
            agent_timeout = task_info.get("agent_timeout_sec")
            if agent_timeout is not None:
                try:
                    agent_result = await asyncio.wait_for(
                        loop.run(prompt), timeout=float(agent_timeout),
                    )
                except TimeoutError:
                    logger.warning(
                        "Agent timed out for %s after %ss",
                        instance.id, agent_timeout,
                    )
                    if context.training is not None:
                        context.training.finish_status = "timeout"
                    agent_result = AgentResult(
                        trajectory=context.trajectory,
                        messages=list(context.messages),
                        finish_reason="timeout",
                        error=f"Agent timed out after {agent_timeout}s",
                    )
            else:
                agent_result = await loop.run(prompt)

            if self.task.requires_agent_result_evaluation():
                # Text-answer tasks evaluate metadata["final_answer"] instead of a patch.
                instance.metadata["_agent_final_answer"] = agent_result.metadata.get(
                    "final_answer"
                )
                instance.metadata["_agent_finish_reason"] = agent_result.finish_reason

            if session is not None:
                # Collect a non-patch artifact (e.g. NL2Repo's tarball) while
                # the agent session is still alive.  Stashed onto the instance
                # below so the evaluator can read it through ``metadata``.
                try:
                    artifact_bytes = await self.task.collect_artifact(instance, session)
                except Exception as e:
                    logger.warning(
                        "Failed to collect artifact for %s: %s", instance.id, e,
                    )

                # Same-session evaluation (must happen before session closes).
                if same_session_eval:
                    eval_result = await self._evaluate_same_session(
                        instance, session,
                    )

        # Isolated evaluation (default: agent session already released).
        if not same_session_eval and self.evaluator:
            if self.task.requires_agent_result_evaluation():
                # No patch is needed; the evaluator reads the answer from instance metadata.
                eval_result = await self._evaluate(instance, "")
            elif agent_result.patch or artifact_bytes is not None:
                if artifact_bytes is not None:
                    instance.metadata["_agent_artifact"] = artifact_bytes
                eval_result = await self._evaluate(instance, agent_result.patch)

        elapsed = time.monotonic() - start_time
        # Classify the outcome (infra vs genuine failure) from the signals that
        # already exist, so downstream aggregation can exclude infra failures.
        if eval_result is not None:
            eval_result.error_kind = infer_error_kind(
                finish_reason=(agent_result.finish_reason if agent_result else None),
                eval_result=eval_result,
                task_error=None,
            )
        return TaskResult(
            instance_id=instance.id,
            agent_result=agent_result,
            eval_result=eval_result,
            metadata={"duration": elapsed},
        )

    def _instance_runtime_config(self, instance: Instance) -> RuntimeConfig:
        """Build a runtime config with per-instance overrides."""
        updates: dict[str, Any] = {}

        instance_workdir = instance.workdir or self.runtime_config.workdir
        updates["workdir"] = instance_workdir

        # Per-instance resource limits (e.g. Terminal Bench task.toml).
        limits = self.task.get_resource_limits(instance)
        if limits:
            from aweagent.core.runtime.config import ResourceLimits
            updates["resource_limits"] = ResourceLimits(
                cpu=limits.get("cpu", "1"),
                memory=limits.get("memory", "2048Mi"),
            )

        # Per-instance Docker environment variables (e.g. BASH_ENV).
        env = self.task.get_docker_environment(instance)
        if env:
            docker_update = self.runtime_config.docker.model_copy(
                update={"environment": env},
            )
            updates["docker"] = docker_update

        return self.runtime_config.model_copy(update=updates)

    async def _evaluate_same_session(
        self, instance: Instance, session: Any,
    ) -> EvalResult:
        """Evaluate inside the agent's session (no patch, no isolation)."""
        from dataclasses import replace

        from aweagent.core.runtime.reuse_session import RuntimeWithExistingSession

        eval_instance = instance
        if self.workspace_output_dir:
            meta = dict(instance.metadata) if instance.metadata else {}
            meta["workspace_output_dir"] = self.workspace_output_dir
            eval_instance = replace(instance, metadata=meta)

        eval_runtime = RuntimeWithExistingSession(session)
        return await self.evaluator.evaluate(eval_instance, "", eval_runtime)

    async def _evaluate(self, instance: Instance, patch: str) -> EvalResult:
        """Evaluate in an isolated runtime."""
        if not self.evaluator:
            return EvalResult()

        eval_config = self.eval_runtime_config or self.runtime_config
        eval_runtime_cls = runtime_registry.get(eval_config.backend)
        eval_runtime: Runtime = eval_runtime_cls(eval_config)

        return await self.evaluator.evaluate(instance, patch, eval_runtime)
