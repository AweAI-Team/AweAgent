"""TB2BatchRunner — batch execution engine for Terminal Bench V2.

Custom runner instead of TaskRunner because:
1. Terminus2Agent uses a custom loop (run_async instead of AgentLoop)
2. Evaluation must happen in the same container (no patch, no isolation)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Shared helpers (used by both batch and debug modes) ──────────────


def build_runtime(config, inst, task, agent_timeout):
    """Build a DockerRuntime with per-instance resource limits and BASH_ENV."""
    from awe_agent.core.runtime.config import DockerConfig, ResourceLimits, RuntimeConfig
    from awe_agent.core.runtime.docker import DockerRuntime

    image = task.get_image(inst)
    limits = task.get_resource_limits(inst)
    resource_limits = ResourceLimits(cpu=limits["cpu"], memory=limits["memory"])

    runtime_config = RuntimeConfig(
        backend="docker",
        image=image,
        workdir=inst.workdir,
        timeout=int(agent_timeout) + 600,
        resource_limits=resource_limits,
        docker=DockerConfig(
            pull_policy="if_not_present",
            environment={"BASH_ENV": "/root/.awe_agent_env"},
        ),
    )
    return DockerRuntime(runtime_config), image


def build_agent_context(config, session, task_info, agent):
    """Build AgentContext for a single instance run."""
    from awe_agent.core.agent.context import AgentContext
    from awe_agent.core.condenser import build_condenser
    from awe_agent.core.llm.client import LLMClient

    return AgentContext(
        llm=LLMClient(config.llm),
        session=session,
        tools=agent.get_tools(),
        task_info=task_info,
        max_steps=config.agent.max_steps,
        max_context_length=config.agent.max_context_length,
        condenser=build_condenser(config.agent.condenser),
    )


def resolve_agent_timeout(config, task_info: dict) -> float:
    """Resolve agent timeout: config override > task.toml > default 600s."""
    override = getattr(config.task, "override_agent_timeout", None)
    if override is not None:
        return float(override)
    return float(task_info.get("agent_timeout_sec", 600))


async def run_agent(config, agent, prompt, ctx, agent_timeout):
    """Run agent with timeout. Returns AgentResult or None on timeout."""
    try:
        if getattr(agent, "uses_custom_loop", False):
            return await asyncio.wait_for(
                agent.run_async(prompt, ctx), timeout=agent_timeout
            )
        else:
            from awe_agent.core.agent.loop import AgentLoop
            loop = AgentLoop(agent, ctx)
            return await asyncio.wait_for(loop.run(prompt), timeout=agent_timeout)
    except TimeoutError:
        return None


async def run_evaluation(inst, session, evaluator):
    """Run evaluation in the same session. Returns EvalResult or None."""
    if evaluator is None:
        return None
    from awe_agent.core.runtime.reuse_session import RuntimeWithExistingSession

    eval_runtime = RuntimeWithExistingSession(session)
    return await evaluator.evaluate(inst, "", eval_runtime)


# ── TB2BatchRunner ───────────────────────────────────────────────────


class TB2BatchRunner:
    """Batch runner for Terminal Bench V2."""

    def __init__(
        self,
        config,
        task,
        *,
        skip_eval: bool = False,
        save_trajectories: bool = True,
        max_retries: int = 0,
    ) -> None:
        self.config = config
        self.task = task
        self.skip_eval = skip_eval
        self.save_trajectories = save_trajectories
        self.max_retries = max_retries

        self._evaluator = None if skip_eval else self._make_evaluator()
        self._write_lock = asyncio.Lock()

        model_name = config.llm.model.rsplit("/", 1)[-1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(config.execution.output_path) / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._results_file = self.run_dir / "results.jsonl"
        self._traj_file = (
            self.run_dir / "trajectories.jsonl" if save_trajectories else None
        )

    @staticmethod
    def _make_evaluator():
        from awe_agent.tasks.terminal_bench_v2.evaluator import TerminalBenchV2Evaluator
        return TerminalBenchV2Evaluator()

    async def run_all(self, instance_ids: list[str] | None = None) -> list[dict]:
        """Run all instances concurrently, return result records."""
        instances = self.task.get_instances(instance_ids)
        if not instances:
            print("No instances to run.")
            return []

        self._save_run_config(instances)

        semaphore = asyncio.Semaphore(self.config.execution.max_concurrent)

        async def run_one(inst):
            async with semaphore:
                return await self._run_with_retries(inst)

        results = await asyncio.gather(
            *[run_one(inst) for inst in instances], return_exceptions=True
        )

        final = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                record = {
                    "instance_id": instances[i].id,
                    "success": False,
                    "score": 0.0,
                    "error": str(r),
                }
                final.append(record)
            else:
                final.append(r)

        successes = sum(1 for r in final if r.get("success"))
        errors = sum(1 for r in final if r.get("error"))
        print(f"\nResults: {successes}/{len(final)} accepted, {errors} errors")
        print(f"Output: {self.run_dir}")
        return final

    async def _run_with_retries(self, inst) -> dict:
        """Run a single instance with retry on failure."""
        last_error = None
        for attempt in range(1 + self.max_retries):
            if attempt > 0:
                logger.info(
                    "Retrying %s (attempt %d/%d)",
                    inst.id, attempt + 1, 1 + self.max_retries,
                )
            try:
                return await self._run_single(inst, attempt=attempt)
            except Exception as e:
                last_error = e
                logger.error(
                    "Instance %s attempt %d failed: %s",
                    inst.id, attempt + 1, e, exc_info=True,
                )
        record = {
            "instance_id": inst.id,
            "dataset_id": inst.dataset_id,
            "success": False,
            "score": 0.0,
            "error": f"Failed after {1 + self.max_retries} attempts: {last_error}",
            "finish_reason": "error",
            "duration": 0.0,
        }
        await self._write_result(record)
        return record

    async def _run_single(self, inst, *, attempt: int = 0) -> dict:
        """Run agent + eval on a single TB2 instance (same-session)."""
        from awe_agent.core.eval.setup import PreAgentSetup
        from awe_agent.scaffold.registry import agent_registry

        start = time.monotonic()
        task_info = self.task.get_task_info(inst)
        agent_timeout = resolve_agent_timeout(self.config, task_info)
        runtime, image = build_runtime(self.config, inst, self.task, agent_timeout)

        record = {
            "instance_id": inst.id,
            "dataset_id": inst.dataset_id,
            "success": False,
            "score": 0.0,
            "error": None,
            "finish_reason": "",
            "attempt": attempt,
        }

        eval_result = None
        agent_result = None

        async with runtime.session(image) as session:
            setup = PreAgentSetup(session, inst.workdir)
            await setup.run_setup_commands(self.task.get_setup_commands(inst))

            agent_cls = agent_registry.get(self.config.agent.type)
            agent = agent_cls.from_config(self.config)
            ctx = build_agent_context(self.config, session, task_info, agent)
            prompt = self.task.get_prompt(inst)

            agent_result = await run_agent(
                self.config, agent, prompt, ctx, agent_timeout
            )

            if agent_result is None:
                logger.warning(
                    "Agent timed out for %s after %ds", inst.id, agent_timeout
                )
                record["finish_reason"] = "timeout"
                record["error"] = f"Agent timed out after {agent_timeout}s"
            else:
                record["finish_reason"] = agent_result.finish_reason

            eval_result = await run_evaluation(inst, session, self._evaluator)
            if eval_result is not None:
                record["success"] = eval_result.accepted
                record["score"] = eval_result.score

        record["duration"] = time.monotonic() - start

        await self._write_result(record)
        await self._write_trajectory(inst, agent_result, record, eval_result)

        logger.info(
            "Instance %s done: score=%.1f accepted=%s (%.0fs)",
            inst.id, record["score"], record["success"], record["duration"],
        )
        return record

    async def _write_result(self, record: dict) -> None:
        async with self._write_lock:
            with open(self._results_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    async def _write_trajectory(
        self, inst, agent_result, record: dict, eval_result=None,
    ) -> None:
        if self._traj_file is None:
            return
        try:
            if agent_result is not None:
                traj_steps = [
                    {
                        "step": step.step,
                        "action": {
                            "type": step.action.type,
                            "content": step.action.content,
                            "tool_calls": step.action.tool_calls,
                        },
                        "observations": step.observations,
                    }
                    for step in agent_result.trajectory.steps
                ]
                finish_reason = agent_result.finish_reason
                error = agent_result.error
                patch = agent_result.patch
                stats = agent_result.metadata.get("stats")
            else:
                traj_steps = []
                finish_reason = record.get("finish_reason", "")
                error = record.get("error")
                patch = ""
                stats = None

            eval_dict = None
            if eval_result is not None:
                eval_dict = asdict(eval_result)

            traj_record = {
                "instance_id": inst.id,
                "success": record["success"],
                "score": record["score"],
                "finish_reason": finish_reason,
                "error": error,
                "duration": record.get("duration"),
                "patch": patch,
                "stats": stats,
                "trajectory": traj_steps,
                "eval_result": eval_dict,
            }
            async with self._write_lock:
                with open(self._traj_file, "a") as f:
                    f.write(
                        json.dumps(traj_record, ensure_ascii=False, default=str) + "\n"
                    )
        except Exception as e:
            logger.warning("Failed to save trajectory for %s: %s", inst.id, e)

    def _save_run_config(self, instances) -> None:
        try:
            config_snapshot = json.loads(self.config.model_dump_json())
            for key in ("api_key", "secret", "token", "password"):
                config_snapshot.get("llm", {}).pop(key, None)
            (self.run_dir / "run_config.json").write_text(
                json.dumps({
                    "start_time": datetime.now().isoformat(),
                    "config": config_snapshot,
                    "instance_count": len(instances),
                    "instance_ids": [inst.id for inst in instances],
                }, indent=2) + "\n"
            )
        except Exception as e:
            logger.warning("Failed to save run config: %s", e)
