"""Example: Batch evaluation using TaskRunner.

Shows how to run an agent on multiple task instances concurrently.
"""

from __future__ import annotations

import asyncio
from typing import Any

from awe_agent.core.config import load_config
from awe_agent.core.eval.isolation import IsolatedEvaluator
from awe_agent.core.llm import LLMConfig
from awe_agent.core.runtime import RuntimeConfig
from awe_agent.core.task import Evaluator, Instance, Task, TaskRunner
from awe_agent.core.task.types import EvalResult
from awe_agent.scaffold.search_swe import SearchSWEAgent


class MyBenchmark(Task):
    """Example benchmark task."""

    def __init__(self, data_file: str) -> None:
        self._data_file = data_file

    def get_instances(self, instance_ids: list[str] | None = None) -> list[Instance]:
        # In practice, load from JSON/JSONL file
        return [
            Instance(
                id="instance-001",
                dataset_id="my_benchmark",
                repo="myorg/myrepo",
                base_commit="abc123",
                workdir="/testbed",
                image="my-benchmark:latest",
                problem_statement="Fix the failing test in test_utils.py",
            ),
        ]

    def get_prompt(self, instance: Instance) -> str:
        return (
            f"You are working on the repository {instance.repo}.\n\n"
            f"Problem: {instance.problem_statement}\n\n"
            f"Please fix the issue. The repo is at {instance.workdir}."
        )


async def main() -> None:
    # Load config from YAML
    config = load_config("configs/default.yaml")

    # Create runner
    runner = TaskRunner(
        task=MyBenchmark("data/instances.json"),
        agent_factory=lambda: SearchSWEAgent(
            bash_blacklist=config.security.bash_blacklist,
        ),
        llm_config=config.llm,
        runtime_config=config.runtime,
        evaluator=IsolatedEvaluator(
            eval_script="cd /testbed && pytest tests/ -x",
            timeout=config.eval.timeout,
        ),
        eval_runtime_config=RuntimeConfig(
            backend=config.runtime.backend,
            image=config.runtime.image,
        ),
        max_concurrent=config.execution.max_concurrent,
        max_retries=config.execution.max_retries,
        output_path=config.execution.output_path,
    )

    # Run all instances
    results = await runner.run_all()

    # Summary
    successes = sum(1 for r in results if r.success)
    print(f"Results: {successes}/{len(results)} passed")


if __name__ == "__main__":
    asyncio.run(main())
