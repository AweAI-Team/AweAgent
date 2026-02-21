"""TaskRunner — batch execution engine for running agents on task instances.

Manages concurrency, retry, result collection, and output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
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


class TaskRunner:
    """Batch execution engine.

    Runs an agent on multiple task instances concurrently,
    with optional evaluation in isolated containers.

    Example:
        runner = TaskRunner(
            task=SWEBenchTask("swe_bench_verified"),
            agent_factory=lambda: SearchSWEAgent(enable_search=True),
            llm_config=LLMConfig(backend="openai", model="gpt-4o"),
            runtime_config=RuntimeConfig(backend="docker", image="swe-bench:latest"),
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
    ) -> None:
        self.task = task
        self.agent_factory = agent_factory
        self.llm_config = llm_config
        self.runtime_config = runtime_config
        self.evaluator = evaluator or task.default_evaluator()
        self.eval_runtime_config = eval_runtime_config
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.output_path = Path(output_path)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._condenser = condenser

    async def run_all(
        self,
        instance_ids: list[str] | None = None,
    ) -> list[TaskResult]:
        """Run agent on all instances concurrently."""
        instances = self.task.get_instances(instance_ids)
        logger.info("Running %d instances (max_concurrent=%d)", len(instances), self.max_concurrent)

        self.output_path.mkdir(parents=True, exist_ok=True)
        output_file = self.output_path / "results.jsonl"
        write_lock = asyncio.Lock()

        tasks = [
            self._run_instance_with_retry(inst, output_file, write_lock)
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
    ) -> TaskResult:
        """Run a single instance with retry logic."""
        async with self._semaphore:
            last_error: str | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    result = await self._run_instance(instance)

                    # Write result
                    async with write_lock:
                        with open(output_file, "a") as f:
                            f.write(json.dumps({
                                "instance_id": result.instance_id,
                                "success": result.success,
                                "score": result.eval_result.score if result.eval_result else 0.0,
                                "error": result.error,
                                "patch": result.agent_result.patch if result.agent_result else "",
                                "finish_reason": result.agent_result.finish_reason if result.agent_result else "",
                            }) + "\n")

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
                condenser=self._condenser,
            )
            loop = AgentLoop(agent, context)

            # Run agent
            prompt = self.task.get_prompt(instance)
            agent_result = await loop.run(prompt)

            # Evaluate in isolated container
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
