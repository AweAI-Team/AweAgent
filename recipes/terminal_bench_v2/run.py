"""Terminal Bench V2 recipe — batch execution entry point.

Usage:
    python recipes/terminal_bench_v2/run.py \\
        --task-data-dir data/terminal-bench-2 \\
        --data-file data/terminal-bench-2/instance_ids.json

    # With overrides
    python recipes/terminal_bench_v2/run.py \\
        --task-data-dir data/terminal-bench-2 \\
        --data-file data/terminal-bench-2/instance_ids.json \\
        --model Qwen/Qwen3-32B \\
        --max-steps 50 \\
        --max-concurrent 10 \\
        --instance-ids task_a task_b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from awe_agent.core.config.loader import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Terminal Bench V2 batch runner")
    p.add_argument(
        "--task-data-dir",
        default=os.environ.get("TASK_DATA_DIR"),
        help="Root directory of task folders (or TASK_DATA_DIR env)",
    )
    p.add_argument(
        "--data-file",
        default=os.environ.get("DATA_FILE"),
        help="JSON file with instance ID array (or DATA_FILE env)",
    )
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/terminal_bench_v2.yaml",
        help="Path to YAML config (default: configs/tasks/terminal_bench_v2.yaml)",
    )
    p.add_argument("--instance-ids", nargs="*", default=None, help="Instance IDs to run (filter)")
    p.add_argument("--model", default=None, help="Override LLM model")
    p.add_argument("--max-steps", type=int, default=None, help="Override max agent steps")
    p.add_argument("--max-concurrent", type=int, default=None, help="Override concurrency")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--no-trajectories", action="store_true", help="Disable saving trajectories")
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return p.parse_args()


def _load_config(args: argparse.Namespace):
    overrides: dict = {}
    if args.model is not None:
        overrides.setdefault("llm", {})["model"] = args.model
    if args.max_steps is not None:
        overrides.setdefault("agent", {})["max_steps"] = args.max_steps
    if args.max_concurrent is not None:
        overrides.setdefault("execution", {})["max_concurrent"] = args.max_concurrent
    if args.output is not None:
        overrides.setdefault("execution", {})["output_path"] = args.output
    if args.task_data_dir is not None:
        overrides.setdefault("task", {})["task_data_dir"] = args.task_data_dir
    if args.data_file is not None:
        overrides.setdefault("task", {})["data_file"] = args.data_file
    return load_config(args.config, overrides=overrides)


def _build_task(config):
    from awe_agent.tasks.terminal_bench_v2.task import TerminalBenchV2Task

    task_data_dir = config.task.task_data_dir
    data_file = config.task.data_file
    if not task_data_dir:
        raise ValueError(
            "task_data_dir is required. Set --task-data-dir or TASK_DATA_DIR."
        )
    if not data_file:
        raise ValueError(
            "data_file is required. Set --data-file or DATA_FILE."
        )
    return TerminalBenchV2Task(
        task_data_dir=task_data_dir,
        data_file=data_file,
        dataset_id=config.task.dataset_id,
    )


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _load_config(args)
    task = _build_task(config)

    print(f"LLM:    backend={config.llm.backend}, model={config.llm.model}")
    print(f"Agent:  type={config.agent.type}, max_steps={config.agent.max_steps}")
    print(f"Task:   task_data_dir={config.task.task_data_dir}")

    from awe_agent.core.condenser import build_condenser
    from awe_agent.core.task.runner import TaskRunner
    from awe_agent.scaffold.registry import agent_registry

    agent_cls = agent_registry.get(config.agent.type)

    def agent_factory(search_constraints=None):
        return agent_cls.from_config(config)

    evaluator = None
    if not args.skip_eval:
        evaluator = task.default_evaluator(timeout=config.eval.timeout)

    save_traj = config.execution.save_trajectories and not args.no_trajectories

    runner = TaskRunner(
        task=task,
        agent_factory=agent_factory,
        llm_config=config.llm,
        runtime_config=config.runtime,
        evaluator=evaluator,
        max_concurrent=config.execution.max_concurrent,
        max_retries=config.execution.max_retries,
        output_path=config.execution.output_path,
        condenser=build_condenser(config.agent.condenser),
        save_trajectories=save_traj,
        config_snapshot=json.loads(config.model_dump_json()),
        max_steps=config.agent.max_steps,
        max_context_length=config.agent.max_context_length,
    )

    results = await runner.run_all(args.instance_ids)

    successes = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if r.error)
    print(f"\nResults: {successes}/{len(results)} accepted, {errors} errors")
    print(f"Output: {runner.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
