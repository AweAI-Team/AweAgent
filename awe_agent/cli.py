"""CLI entry point for AweAgent.

Usage:
    awe-agent run --config config.yaml                     # Run with config
    awe-agent run --config config.yaml --instance-ids X Y  # Run specific instances
    awe-agent info                                         # Show available backends
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from awe_agent import __version__


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="awe-agent",
        description="AweAgent — extensible scaffold framework for code & search agents",
    )
    parser.add_argument("--version", action="version", version=f"awe-agent {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── run command ──────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run agent on task instances")
    run_parser.add_argument(
        "-c", "--config", required=True, help="Path to YAML config file"
    )
    run_parser.add_argument(
        "--instance-ids", nargs="*", help="Specific instance IDs to run"
    )
    run_parser.add_argument(
        "-o", "--output", default=None, help="Output directory (default: from config)"
    )
    run_parser.add_argument(
        "--no-trajectories", action="store_true",
        help="Disable saving per-instance trajectory files",
    )
    run_parser.add_argument(
        "--max-concurrent", type=int, help="Override max concurrent instances"
    )
    run_parser.add_argument(
        "--max-steps", type=int, help="Override max agent steps"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Load config and list instances without running"
    )

    # ── info command ─────────────────────────────────────────────────
    info_parser = subparsers.add_parser("info", help="Show available backends and plugins")

    # Parse
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "info":
        _cmd_info()
    elif args.command == "run":
        asyncio.run(_cmd_run(args))
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_info() -> None:
    """Show available backends, tools, and plugins."""
    from awe_agent.core.llm.client import llm_registry
    from awe_agent.core.task.runner import runtime_registry
    from awe_agent.core.tool.registry import tool_registry
    from awe_agent.scaffold.registry import agent_registry

    print(f"AweAgent v{__version__}\n")

    print("LLM Backends:")
    for name in llm_registry.list_available():
        print(f"  - {name}")

    print("\nRuntime Backends:")
    for name in runtime_registry.list_available():
        print(f"  - {name}")

    print("\nAgent Scaffolds:")
    for name in agent_registry.list_available():
        print(f"  - {name}")

    print("\nTools:")
    for name in tool_registry.list_available():
        print(f"  - {name}")

    print("\nTasks: beyond_swe")


async def _cmd_run(args: argparse.Namespace) -> None:
    """Run agent on task instances."""
    from awe_agent.core.config.loader import load_config
    from awe_agent.core.task.runner import TaskRunner

    logger = logging.getLogger("awe_agent.cli")

    # Build config overrides from CLI args
    overrides: dict[str, Any] = {}
    if args.max_concurrent is not None:
        overrides.setdefault("execution", {})["max_concurrent"] = args.max_concurrent
    if args.max_steps is not None:
        overrides.setdefault("agent", {})["max_steps"] = args.max_steps
    if args.output is not None:
        overrides.setdefault("execution", {})["output_path"] = args.output

    # Load config
    config = load_config(args.config, overrides=overrides)
    logger.info("Config loaded: llm=%s, runtime=%s, agent=%s, task=%s",
                config.llm.backend, config.runtime.backend, config.agent.type, config.task.type)

    # Build task
    task = _build_task(config)
    instances = task.get_instances(args.instance_ids)

    if args.dry_run:
        print(f"\nDry run — {len(instances)} instances loaded:")
        for inst in instances[:20]:
            print(f"  {inst.id} (image={inst.image[:50] if inst.image else 'none'})")
        if len(instances) > 20:
            print(f"  ... and {len(instances) - 20} more")
        return

    logger.info("Running %d instances", len(instances))

    # Build agent factory
    agent_factory = _build_agent_factory(config)

    # Build evaluator (optional)
    evaluator = _build_evaluator(config, task)

    # Build condenser (optional)
    from awe_agent.core.condenser import build_condenser
    condenser = build_condenser(config.agent.condenser)

    # Build config snapshot for saving
    config_snapshot = json.loads(config.model_dump_json())

    # Run
    runner = TaskRunner(
        task=task,
        agent_factory=agent_factory,
        llm_config=config.llm,
        runtime_config=config.runtime,
        evaluator=evaluator,
        max_concurrent=config.execution.max_concurrent,
        max_retries=config.execution.max_retries,
        output_path=config.execution.output_path,
        condenser=condenser,
        save_trajectories=config.execution.save_trajectories and not args.no_trajectories,
        config_snapshot=config_snapshot,
        max_steps=config.agent.max_steps,
    )

    results = await runner.run_all(args.instance_ids)

    # Summary
    successes = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if r.error)
    print(f"\nResults: {successes}/{len(results)} accepted, {errors} errors")
    print(f"Output: {runner.run_dir}")


def _build_task(config: Any):
    """Build a Task instance from config."""
    from awe_agent.tasks.beyond_swe.task import BeyondSWETask

    task_type = config.task.type
    search_mode = config.agent.enable_search

    if task_type == "beyond_swe":
        return BeyondSWETask(
            dataset_id=config.task.dataset_id,
            data_file=config.task.data_file,
            search_mode=search_mode,
        )
    else:
        raise ValueError(f"Unknown task type: {task_type}. Available: beyond_swe.")


def _build_agent_factory(config: Any):
    """Build an agent factory function from config.

    Returns a callable that accepts optional ``search_constraints`` kwarg
    for per-instance constraint injection.
    """
    from awe_agent.scaffold.registry import agent_registry

    agent_cls = agent_registry.get(config.agent.type)

    def factory(search_constraints=None):
        if search_constraints and hasattr(agent_cls, "from_config_with_constraints"):
            return agent_cls.from_config_with_constraints(config, search_constraints)
        return agent_cls.from_config(config)

    return factory


def _build_evaluator(config: Any, task: Any):
    """Build an evaluator from config, or None if eval is disabled.

    Prefers the task's own evaluator (e.g. BeyondSWEEvaluator) over the
    generic IsolatedEvaluator so that task-specific evaluation logic is used.
    """
    if not config.eval.enabled:
        return None

    # Prefer task-specific evaluator
    if hasattr(task, "default_evaluator"):
        task_eval = task.default_evaluator(timeout=config.eval.timeout)
        if task_eval is not None:
            return task_eval

    # Fallback to generic isolated evaluator
    if config.eval.isolated:
        from awe_agent.core.eval.isolation import IsolatedEvaluator
        return IsolatedEvaluator(eval_script=config.eval.eval_script)

    return None


if __name__ == "__main__":
    main()
