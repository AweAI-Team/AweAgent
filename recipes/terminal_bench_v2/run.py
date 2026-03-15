"""Terminal Bench V2 recipe — unified entry point for prompt, debug, and batch runs.

Modes:
  prompt   — Print the generated prompt and task_info for a single instance (no Docker)
  debug    — Full single-instance run (agent + eval) with detailed trace
  batch    — Batch concurrent execution, results in JSONL
  dry-run  — List all instances without executing anything

Key CLI args:
  --task-data-dir    Root directory of task folders (required, or TASK_DATA_DIR)
  --data-file        JSON file with instance ID array (required, or DATA_FILE)
  --config / -c      YAML config (default: configs/tasks/terminal_bench_v2.yaml)
  --instance-id      Single instance ID (prompt/debug)
  --instance-ids     Instance IDs for batch (optional filter)
  --model            Override LLM model
  --max-steps        Override max agent steps
  --max-concurrent   Override concurrency (batch)
  --output           Output directory (batch)
  --skip-eval        Skip evaluation
  --verbose          DEBUG logging

Usage examples:

    # List instances (dry-run)
    python recipes/terminal_bench_v2/run.py \\
        --task-data-dir data/terminal-bench-2 \\
        --data-file data/terminal-bench-2/instance_ids.json \\
        --mode dry-run

    # Debug single instance
    python recipes/terminal_bench_v2/run.py \\
        --task-data-dir data/terminal-bench-2 \\
        --data-file data/terminal-bench-2/instance_ids.json \\
        --instance-id configure-git-webserver --mode debug

    # Batch run
    python recipes/terminal_bench_v2/run.py \\
        --task-data-dir data/terminal-bench-2 \\
        --data-file data/terminal-bench-2/instance_ids.json \\
        --mode batch
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

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Terminal Bench V2 recipe")
    p.add_argument(
        "--task-data-dir",
        default=os.environ.get("TASK_DATA_DIR"),
        help="Root directory of task folders (or TASK_DATA_DIR)",
    )
    p.add_argument(
        "--data-file",
        default=os.environ.get("DATA_FILE"),
        help="JSON file with instance ID array (or DATA_FILE)",
    )
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/terminal_bench_v2.yaml",
        help="Path to YAML config",
    )
    p.add_argument(
        "--mode",
        choices=["prompt", "debug", "batch", "dry-run"],
        default="dry-run",
        help="prompt|debug|batch|dry-run (default: dry-run)",
    )
    p.add_argument("--instance-id", default=None, help="Single instance ID (prompt/debug)")
    p.add_argument("--instance-ids", nargs="*", default=None, help="Instance IDs (batch filter)")
    p.add_argument("--model", default=None, help="Override LLM model")
    p.add_argument("--max-steps", type=int, default=None, help="Override max agent steps")
    p.add_argument("--max-concurrent", type=int, default=None, help="Override concurrency (batch)")
    p.add_argument("--output", default=None, help="Output directory (batch)")
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

    if args.task_data_dir:
        os.environ.setdefault("TASK_DATA_DIR", args.task_data_dir)
    if args.data_file:
        os.environ.setdefault("DATA_FILE", args.data_file)
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


def _print_section(title: str, content: str, max_len: int = 5000) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if len(content) > max_len:
        print(content[:max_len])
        print(f"\n... (truncated, total {len(content)} chars)")
    else:
        print(content)


from awe_agent.core.task.tb2_batch_runner import (
    TB2BatchRunner,
    build_agent_context,
    build_runtime,
    resolve_agent_timeout,
    run_agent,
    run_evaluation,
)


# ── Mode implementations ──────────────────────────────────────────────


def _mode_dry_run(task, instance_ids: list[str] | None) -> None:
    instances = task.get_instances(instance_ids)
    print(f"\nDry run — {len(instances)} instances loaded:")
    for inst in instances:
        limits = task.get_resource_limits(inst)
        res = f"  cpus={limits['cpu']} mem={limits['memory']}" if limits else ""
        print(f"  {inst.id}  workdir={inst.workdir}  image={inst.image}  {res}")


def _mode_prompt(task, instance_id: str) -> None:
    instances = task.get_instances(instance_ids=[instance_id])
    if not instances:
        print(f"ERROR: instance '{instance_id}' not found")
        sys.exit(1)

    inst = instances[0]
    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)
    limits = task.get_resource_limits(inst)

    _print_section("INSTANCE", json.dumps({
        "id": inst.id,
        "workdir": inst.workdir,
        "docker_image": inst.image,
        "resource_limits": limits,
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2, default=str))
    _print_section("PROMPT", prompt)
    _print_section("SETUP COMMANDS", "\n".join(task.get_setup_commands(inst)) or "(none)")


async def _mode_debug(config, task, instance_id: str, skip_eval: bool) -> None:
    from awe_agent.core.eval.setup import PreAgentSetup
    from awe_agent.scaffold.registry import agent_registry
    from awe_agent.tasks.terminal_bench_v2.evaluator import TerminalBenchV2Evaluator

    instances = task.get_instances(instance_ids=[instance_id])
    if not instances:
        print(f"ERROR: instance '{instance_id}' not found")
        sys.exit(1)

    inst = instances[0]
    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)

    _print_section("INSTANCE", json.dumps({
        "id": inst.id, "workdir": inst.workdir, "docker_image": inst.image,
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2, default=str))
    _print_section("PROMPT", prompt[:1500] + "..." if len(prompt) > 1500 else prompt)

    agent_timeout = resolve_agent_timeout(config, task_info)
    runtime, image = build_runtime(config, inst, task, agent_timeout)

    async with runtime.session(image) as session:
        setup = PreAgentSetup(session, inst.workdir)
        await setup.run_setup_commands(task.get_setup_commands(inst))

        agent_cls = agent_registry.get(config.agent.type)
        agent = agent_cls.from_config(config)
        ctx = build_agent_context(config, session, task_info, agent)

        print(
            f"\nStarting agent (max_steps={config.agent.max_steps}, "
            f"model={config.llm.model}, timeout={agent_timeout}s) ..."
        )

        result = await run_agent(config, agent, prompt, ctx, agent_timeout)

        if result is None:
            print(f"\n[agent] Timed out after {agent_timeout}s.")
        else:
            for step in result.trajectory.steps:
                print(f"\n{'─' * 50}")
                print(f"  Step {step.step}  |  action={step.action.type}")
                print(f"{'─' * 50}")
                if step.action.content:
                    print(f"  [content] {step.action.content[:500]}")

            _print_section("RESULT", json.dumps({
                "finish_reason": result.finish_reason,
                "steps": len(result.trajectory.steps),
                "error": result.error,
            }, indent=2))

        # Evaluate in same session — even after agent timeout, the container
        # still has the agent's partial modifications (aligned with Harbor).
        if not skip_eval:
            evaluator = TerminalBenchV2Evaluator()
            eval_result = await run_evaluation(inst, session, evaluator)
            if eval_result:
                _print_section("EVAL RESULT", json.dumps({
                    "accepted": eval_result.accepted,
                    "score": eval_result.score,
                    "duration": eval_result.duration,
                    "details": eval_result.details,
                }, indent=2, default=str))
        else:
            print("\n[eval] Skipped (--skip-eval).")


# ── Main ──────────────────────────────────────────────────────────────


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
    print(f"Mode:   {args.mode}")

    if args.mode == "dry-run":
        _mode_dry_run(task, args.instance_ids)

    elif args.mode == "prompt":
        if not args.instance_id:
            print("ERROR: --instance-id is required for prompt mode")
            sys.exit(1)
        _mode_prompt(task, args.instance_id)

    elif args.mode == "debug":
        if not args.instance_id:
            print("ERROR: --instance-id is required for debug mode")
            sys.exit(1)
        await _mode_debug(config, task, args.instance_id, args.skip_eval)

    elif args.mode == "batch":
        ids = args.instance_ids
        if args.instance_id and not ids:
            ids = [args.instance_id]
        save_traj = (
            config.execution.save_trajectories and not args.no_trajectories
        )
        runner = TB2BatchRunner(
            config,
            task,
            skip_eval=args.skip_eval,
            save_trajectories=save_traj,
            max_retries=config.execution.max_retries,
        )
        await runner.run_all(ids)


if __name__ == "__main__":
    asyncio.run(main())
