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
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aweagent.core.config.loader import load_config


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
    p.add_argument(
        "--agent-timeout",
        type=float,
        default=None,
        help="Wall-clock agent timeout in seconds for every instance",
    )
    p.add_argument(
        "--verifier-timeout",
        type=int,
        default=None,
        help="Uniform timeout in seconds for bash /tests/test.sh",
    )
    p.add_argument("--max-concurrent", type=int, default=None, help="Override concurrency")
    p.add_argument(
        "--cpu-milli",
        type=int,
        default=None,
        help="Global Docker CPU limit in millicores (e.g. 16000 = 16 cores)",
    )
    p.add_argument(
        "--memory-mb",
        type=int,
        default=None,
        help="Global Docker memory limit in MiB",
    )
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
    if args.agent_timeout is not None:
        overrides.setdefault("task", {})["override_agent_timeout"] = args.agent_timeout
    if args.verifier_timeout is not None:
        overrides.setdefault("eval", {})["verifier_timeout"] = args.verifier_timeout
    if args.cpu_milli is not None or args.memory_mb is not None:
        resource_limits = overrides.setdefault("runtime", {}).setdefault(
            "resource_limits", {}
        )
        if args.cpu_milli is not None:
            resource_limits["cpu"] = f"{args.cpu_milli}m"
        if args.memory_mb is not None:
            resource_limits["memory"] = f"{args.memory_mb}Mi"
        overrides.setdefault("runtime", {}).setdefault("extra", {})[
            "_force_resource_limits"
        ] = True
    return load_config(args.config, overrides=overrides)


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _load_config(args)
    from aweagent.core.task.pipeline import build_task
    task = build_task(config)

    print(f"LLM:    backend={config.llm.backend}, model={config.llm.model}")
    print(f"Agent:  type={config.agent.type}, max_steps={config.agent.max_steps}")
    resource_limits = config.runtime.resource_limits
    print(
        f"Runtime: backend={config.runtime.backend}, "
        f"cpu={resource_limits.cpu} "
        f"({resource_limits.cpu_milli_value()}m), "
        f"memory={resource_limits.memory}"
    )
    print(f"Task:   task_data_dir={config.task.task_data_dir}")
    agent_timeout_override = config.task.override_agent_timeout
    if agent_timeout_override is not None:
        print(f"Agent wall-clock timeout override: {agent_timeout_override}s")
    if config.eval.verifier_timeout is not None:
        print(f"Verifier timeout override: {config.eval.verifier_timeout}s")

    from aweagent.core.task.pipeline import build_runner

    save_traj = config.execution.save_trajectories and not args.no_trajectories
    runner = build_runner(
        config, task, skip_eval=args.skip_eval, save_trajectories=save_traj
    )

    results = await runner.run_all(args.instance_ids)

    successes = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if r.error)
    print(f"\nResults: {successes}/{len(results)} accepted, {errors} errors")
    print(f"Output: {runner.run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
