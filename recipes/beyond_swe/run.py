"""BeyondSWE recipe — unified entry point for prompt inspection, debug, and batch runs.

Modes:

  prompt   — Print the generated prompt and task_info for a single instance (no Docker)
  debug    — Full single-instance run (agent + eval) with detailed trace
  batch    — Batch concurrent execution via TaskRunner, results in JSONL
  dry-run  — List all instances without executing anything

Key CLI overrides (all optional, default from YAML config):

  --config / -c       YAML config file (default: configs/tasks/beyond_swe_search.yaml)
  --model             LLM model name, e.g. gpt-4o, glm-4.7
  --max-steps         Max agent steps per instance
  --max-concurrent    Max parallel instances (batch mode)
  --output            Output directory (batch mode)
  --enable-search     Force enable search tools  (SearchSWE style)
  --no-search         Force disable search tools (OpenHands style)
  --skip-eval         Skip evaluation after agent run
  --verbose           DEBUG level logging

Environment variables:

  BEYONDSWE_TEST_SUITE_DIR    Directory containing doc2repo test suite zips
  SERPAPI_API_KEY              API key for SerpAPI search backend
  JINA_API_KEY                 API key for Jina reader backend
  SEARCH_BACKEND               Search backend name (default: auto-discover)
  READER_BACKEND               Reader backend name (default: auto-discover)
  LINK_SUMMARY_CONFIG_PATH     Path to LLM config YAML for link summary
  LINK_SUMMARY_MODEL           LLM model for link summary (default: gpt-4o-mini)

Usage examples:

    # Inspect prompt (no Docker needed)
    python recipes/beyond_swe/run.py \\
        --data-file data.jsonl --instance-id inst_001 --mode prompt

    # Debug single instance with custom model and step limit
    python recipes/beyond_swe/run.py \\
        --data-file data.jsonl --instance-id inst_001 --mode debug \\
        --model glm-4.7 --max-steps 30 --verbose

    # Batch run — SearchSWE style (search enabled by default config)
    python recipes/beyond_swe/run.py \\
        --data-file data.jsonl --mode batch

    # Batch run — OpenHands style (disable search)
    python recipes/beyond_swe/run.py \\
        --data-file data.jsonl --mode batch --no-search

    # Batch run — only specific instances
    python recipes/beyond_swe/run.py \\
        --data-file data.jsonl --mode batch \\
        --instance-ids inst_001 inst_002 inst_003

    # List instances
    python recipes/beyond_swe/run.py \\
        --data-file data.jsonl --mode dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from awe_agent.core.config.loader import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BeyondSWE recipe — unified entry point")
    p.add_argument("--data-file", required=True, help="Path to JSONL data file")
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/beyond_swe_search.yaml",
        help="Path to YAML config (default: configs/tasks/beyond_swe_search.yaml)",
    )
    p.add_argument(
        "--mode",
        choices=["prompt", "debug", "batch", "dry-run"],
        default="prompt",
        help="prompt|debug|batch|dry-run (default: prompt)",
    )
    p.add_argument("--instance-id", default=None, help="Single instance ID (prompt/debug)")
    p.add_argument("--instance-ids", nargs="*", default=None, help="Instance IDs (batch, optional)")
    p.add_argument("--model", default=None, help="Override LLM model")
    p.add_argument("--max-steps", type=int, default=None, help="Override max agent steps")
    p.add_argument("--max-concurrent", type=int, default=None, help="Override concurrency (batch)")
    p.add_argument(
        "--enable-search", default=None, action="store_true",
        help="Enable search tools (SearchSWE style)",
    )
    p.add_argument(
        "--no-search", dest="enable_search", action="store_false",
        help="Disable search tools (OpenHands style)",
    )
    p.add_argument("--output", default=None, help="Output directory (batch)")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    p.add_argument("--no-trajectories", action="store_true", help="Disable saving per-instance trajectory files")
    p.add_argument("--verbose", action="store_true", help="DEBUG level logging")
    return p.parse_args()


def _load_config(args: argparse.Namespace):
    """Load and apply CLI overrides to the YAML config."""
    overrides: dict = {}
    if args.model is not None:
        overrides.setdefault("llm", {})["model"] = args.model
    if args.max_steps is not None:
        overrides.setdefault("agent", {})["max_steps"] = args.max_steps
    if args.max_concurrent is not None:
        overrides.setdefault("execution", {})["max_concurrent"] = args.max_concurrent
    if args.enable_search is not None:
        overrides.setdefault("agent", {})["enable_search"] = args.enable_search
    if args.output is not None:
        overrides.setdefault("execution", {})["output_path"] = args.output

    os.environ.setdefault("DATA_FILE", args.data_file)
    return load_config(args.config, overrides=overrides)


def _build_task(config, data_file: str):
    from awe_agent.tasks.beyond_swe.task import BeyondSWETask

    return BeyondSWETask(
        dataset_id=config.task.dataset_id,
        data_file=data_file,
        search_mode=config.agent.enable_search,
        test_suite_dir=config.task.test_suite_dir,
    )


def _print_section(title: str, content: str, max_len: int = 2000) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if len(content) > max_len:
        print(content[:max_len])
        print(f"\n... (truncated, total {len(content)} chars)")
    else:
        print(content)


# ── Mode implementations ──────────────────────────────────────────────


def _mode_dry_run(task, instance_ids: list[str] | None) -> None:
    instances = task.get_instances(instance_ids)
    print(f"\nDry run — {len(instances)} instances loaded:")
    for inst in instances:
        task_type = inst.metadata.get("task_type", "?")
        print(f"  {inst.id}  type={task_type}  image={inst.image[:60] if inst.image else 'none'}")


def _mode_prompt(task, instance_id: str) -> None:
    instances = task.get_instances(instance_ids=[instance_id])
    if not instances:
        print(f"ERROR: instance '{instance_id}' not found")
        sys.exit(1)

    inst = instances[0]
    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)

    _print_section("INSTANCE", json.dumps({
        "id": inst.id,
        "repo": inst.repo,
        "image": inst.image,
        "workdir": inst.workdir,
        "base_commit": inst.base_commit,
        "task_type": inst.metadata.get("task_type"),
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2))
    _print_section("PROMPT", prompt)
    _print_section("SETUP COMMANDS", "\n".join(task.get_setup_commands(inst)) or "(none)")


async def _mode_debug(config, task, instance_id: str, skip_eval: bool) -> None:
    from awe_agent.core.agent import AgentContext, AgentLoop
    from awe_agent.core.condenser import build_condenser
    from awe_agent.core.eval.setup import PreAgentSetup
    from awe_agent.core.llm import LLMClient
    from awe_agent.core.runtime import RuntimeConfig
    from awe_agent.core.runtime.docker import DockerRuntime
    from awe_agent.scaffold.search_swe import SearchSWEAgent

    instances = task.get_instances(instance_ids=[instance_id])
    if not instances:
        print(f"ERROR: instance '{instance_id}' not found")
        sys.exit(1)

    inst = instances[0]

    # Print prompt info first
    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)
    _print_section("INSTANCE", json.dumps({
        "id": inst.id,
        "repo": inst.repo,
        "image": inst.image,
        "workdir": inst.workdir,
        "base_commit": inst.base_commit,
        "task_type": inst.metadata.get("task_type"),
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2))
    _print_section("PROMPT", prompt)

    # Create runtime
    image = task.get_image(inst)
    runtime_config = RuntimeConfig(backend="docker", image=image, workdir=inst.workdir)
    runtime = DockerRuntime(runtime_config)

    async with runtime.session(image) as session:
        # Pre-agent setup
        setup = PreAgentSetup(session, inst.workdir)
        pre_agent_commit_id = await setup.prepare(inst)

        # Build agent
        search_constraints = task.get_search_constraints(inst)
        agent = SearchSWEAgent(
            enable_search=config.agent.enable_search,
            bash_timeout=config.agent.bash_timeout,
            bash_max_timeout=config.agent.bash_max_timeout,
            max_output_length=config.agent.max_output_length,
            bash_blocklist=config.security.bash_blocklist or None,
            search_constraints=search_constraints,
        )
        llm = LLMClient(config.llm)
        condenser = build_condenser(config.agent.condenser)
        if pre_agent_commit_id:
            task_info["pre_agent_commit_id"] = pre_agent_commit_id
        ctx = AgentContext(
            llm=llm,
            session=session,
            tools=agent.get_tools(),
            task_info=task_info,
            max_steps=config.agent.max_steps,
            max_context_length=config.agent.max_context_length,
            condenser=condenser,
        )
        loop = AgentLoop(agent, ctx)

        print(f"\nStarting agent (max_steps={config.agent.max_steps}, "
              f"model={config.llm.model}) ...")
        result = await loop.run(prompt)

        # Print step-by-step trace
        for step in result.trajectory.steps:
            print(f"\n{'─' * 50}")
            print(f"  Step {step.step}  |  action={step.action.type}")
            print(f"{'─' * 50}")

            if step.action.content:
                print(f"  [thinking] {step.action.content[:500]}")

            if step.action.tool_calls:
                for tc in step.action.tool_calls:
                    name = tc.get("name", tc.get("function", {}).get("name", "?"))
                    raw_args = tc.get("arguments", tc.get("function", {}).get("arguments", ""))
                    print(f"  [tool] {name}")
                    print(f"    args: {str(raw_args)[:300]}")

            for i, obs in enumerate(step.observations):
                print(f"  [obs {i}] {obs[:500]}")

        # Summary
        _print_section("RESULT", json.dumps({
            "finish_reason": result.finish_reason,
            "steps": len(result.trajectory.steps),
            "patch_length": len(result.patch),
            "error": result.error,
        }, indent=2))

        if result.patch:
            _print_section("PATCH", result.patch)

    # Evaluate (outside agent session)
    if not skip_eval:
        from awe_agent.tasks.beyond_swe.evaluator import BeyondSWEEvaluator

        eval_runtime = DockerRuntime(RuntimeConfig(
            backend="docker", image=image, workdir=inst.workdir,
        ))
        evaluator = BeyondSWEEvaluator(timeout=3600)
        eval_result = await evaluator.evaluate(inst, result.patch, eval_runtime)
        _print_section("EVAL RESULT", json.dumps({
            "accepted": eval_result.accepted,
            "score": eval_result.score,
            "duration": eval_result.duration,
            "details": eval_result.details,
        }, indent=2, default=str))
    else:
        print("\n[eval] Skipped (--skip-eval).")


async def _mode_batch(
    config, task, instance_ids: list[str] | None, skip_eval: bool,
    save_trajectories: bool = True,
) -> None:
    from awe_agent.core.condenser import build_condenser
    from awe_agent.core.task.runner import TaskRunner
    from awe_agent.scaffold.registry import agent_registry

    agent_cls = agent_registry.get(config.agent.type)

    def agent_factory(search_constraints=None):
        if search_constraints and hasattr(agent_cls, "from_config_with_constraints"):
            return agent_cls.from_config_with_constraints(config, search_constraints)
        return agent_cls.from_config(config)

    condenser = build_condenser(config.agent.condenser)

    evaluator = None if skip_eval else task.default_evaluator(
        timeout=config.eval.timeout,
    )

    config_snapshot = json.loads(config.model_dump_json())

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
        save_trajectories=save_trajectories,
        config_snapshot=config_snapshot,
        max_steps=config.agent.max_steps,
        max_context_length=config.agent.max_context_length,
    )

    results = await runner.run_all(instance_ids)

    successes = sum(1 for r in results if r.success)
    errors = sum(1 for r in results if r.error)
    print(f"\nResults: {successes}/{len(results)} accepted, {errors} errors")
    print(f"Output: {runner.run_dir}")


# ── Main ──────────────────────────────────────────────────────────────


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = _load_config(args)
    task = _build_task(config, args.data_file)

    print(f"LLM:    backend={config.llm.backend}, model={config.llm.model}")
    print(f"Agent:  type={config.agent.type}, max_steps={config.agent.max_steps}, "
          f"search={config.agent.enable_search}")
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
        await _mode_batch(
            config, task, ids, args.skip_eval,
            save_trajectories=config.execution.save_trajectories and not args.no_trajectories,
        )


if __name__ == "__main__":
    asyncio.run(main())
