"""CLI entry point for AweAgent.

Usage:
    awe-agent run --config config.yaml                     # Run with config
    awe-agent run --config config.yaml --instance-ids X Y  # Run specific instances
    awe-agent info                                         # Show available backends
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from aweagent import __version__


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
        "--start-index",
        type=int,
        help="0-based inclusive start index after instance-id filtering",
    )
    run_parser.add_argument(
        "--end-index",
        type=int,
        help="0-based inclusive end index after instance-id filtering",
    )
    run_parser.add_argument(
        "--max-instances", type=int, help="Run at most N instances after filtering"
    )
    run_parser.add_argument(
        "--max-steps", type=int, help="Override max agent steps"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Load config and list instances without running"
    )

    # ── info command ─────────────────────────────────────────────────
    subparsers.add_parser("info", help="Show available backends and plugins")

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
    from aweagent.core.llm.client import llm_registry
    from aweagent.core.task.registry import task_registry
    from aweagent.core.task.runner import runtime_registry
    from aweagent.core.tool.registry import tool_registry
    from aweagent.scaffold.registry import agent_registry

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

    print("\nTasks:")
    for name in task_registry.list_available():
        print(f"  - {name}")


async def _cmd_run(args: argparse.Namespace) -> None:
    """Run agent on task instances."""
    from aweagent.core.config.loader import load_config
    from aweagent.core.task.pipeline import build_runner, build_task
    from aweagent.core.task.runner import select_instances

    logger = logging.getLogger("aweagent.cli")

    # Build config overrides from CLI args
    overrides: dict[str, Any] = {}
    if args.max_concurrent is not None:
        overrides.setdefault("execution", {})["max_concurrent"] = args.max_concurrent
    if args.start_index is not None:
        overrides.setdefault("execution", {})["start_index"] = args.start_index
    if args.end_index is not None:
        overrides.setdefault("execution", {})["end_index"] = args.end_index
    if args.max_instances is not None:
        overrides.setdefault("execution", {})["max_instances"] = args.max_instances
    if args.max_steps is not None:
        overrides.setdefault("agent", {})["max_steps"] = args.max_steps
    if args.output is not None:
        overrides.setdefault("execution", {})["output_path"] = args.output

    # Load config
    config = load_config(args.config, overrides=overrides)
    logger.info("Config loaded: llm=%s, runtime=%s, agent=%s, task=%s",
                config.llm.backend, config.runtime.backend, config.agent.type, config.task.type)

    # Build task
    task = build_task(config)

    if args.dry_run:
        instances = select_instances(
            task.get_instances(args.instance_ids),
            start_index=config.execution.start_index,
            end_index=config.execution.end_index,
            max_instances=config.execution.max_instances,
        )
        print(f"\nDry run — {len(instances)} instances loaded:")
        for inst in instances[:20]:
            print(f"  {inst.id} (image={inst.image[:50] if inst.image else 'none'})")
        if len(instances) > 20:
            print(f"  ... and {len(instances) - 20} more")
        return

    # Unified runner for all task types (shared pipeline).
    runner = build_runner(
        config, task, save_trajectories=not args.no_trajectories
    )
    results = await runner.run_all(args.instance_ids)

    # Summary
    successes = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if r.error)
    print(f"\nResults: {successes}/{len(results)} accepted, {errors} errors")
    print(f"Output: {runner.run_dir}")


if __name__ == "__main__":
    main()
