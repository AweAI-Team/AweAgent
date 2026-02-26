"""Debug a single BeyondSWE instance with SearchSWEAgent.

Three modes controlled by --mode:

  prompt   — Only load data and print the generated prompt + task_info (no Docker needed)
  dry-run  — Start container + setup, but skip agent loop (verify environment)
  run      — Full agent execution

Usage:
    # Check prompt only (no Docker)
    python scripts/debug_beyond_swe.py \
        --config configs/tasks/beyond_swe_search.yaml \
        --data-file /path/to/data.jsonl \
        --instance-id inst_001 \
        --mode prompt

    # Full run with 10 steps
    python scripts/debug_beyond_swe.py \
        --config configs/tasks/beyond_swe_search.yaml \
        --data-file /path/to/data.jsonl \
        --instance-id inst_001 \
        --mode run \
        --max-steps 10 \
        --model glm-4.7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from awe_agent.core.agent import AgentContext, AgentLoop
from awe_agent.core.config.loader import load_config
from awe_agent.core.eval.setup import PreAgentSetup
from awe_agent.core.llm import LLMClient
from awe_agent.core.runtime import RuntimeConfig
from awe_agent.core.runtime.docker import DockerRuntime
from awe_agent.scaffold.search_swe import SearchSWEAgent
from awe_agent.tasks.beyond_swe.task import BeyondSWETask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug a single BeyondSWE instance")
    p.add_argument("--data-file", required=True, help="Path to JSONL data file")
    p.add_argument("--instance-id", required=True, help="Instance ID to debug")
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/beyond_swe_search.yaml",
        help="Path to YAML config file (default: configs/tasks/beyond_swe_search.yaml)",
    )
    p.add_argument(
        "--mode",
        choices=["prompt", "dry-run", "run"],
        default="prompt",
        help="prompt: inspect prompt only; dry-run: setup container; run: full agent (default: prompt)",
    )
    p.add_argument("--model", default=None, help="Override LLM model from config")
    p.add_argument("--max-steps", type=int, default=None, help="Override max agent steps from config")
    p.add_argument("--enable-search", action="store_true", default=None, help="Enable search tools")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation after agent run")
    p.add_argument("--verbose", action="store_true", help="DEBUG level logging")
    return p.parse_args()


def print_section(title: str, content: str, max_len: int = 2000) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    if len(content) > max_len:
        print(content[:max_len])
        print(f"\n... (truncated, total {len(content)} chars)")
    else:
        print(content)


async def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 1. Load config (same system as CLI) ──────────────────────
    overrides: dict = {}
    if args.model is not None:
        overrides.setdefault("llm", {})["model"] = args.model
    if args.max_steps is not None:
        overrides.setdefault("agent", {})["max_steps"] = args.max_steps
    if args.enable_search is not None:
        overrides.setdefault("agent", {})["enable_search"] = args.enable_search

    # Let DATA_FILE env var be available for ${DATA_FILE} in YAML
    os.environ.setdefault("DATA_FILE", args.data_file)

    config = load_config(args.config, overrides=overrides)

    print(f"LLM:    backend={config.llm.backend}, model={config.llm.model}")
    print(f"        base_url={config.llm.base_url}")
    print(f"Agent:  type={config.agent.type}, max_steps={config.agent.max_steps}, "
          f"search={config.agent.enable_search}")

    # ── 2. Load task & instance ──────────────────────────────────
    task = BeyondSWETask(
        data_file=args.data_file,
        search_mode=config.agent.enable_search,
    )
    instances = task.get_instances(instance_ids=[args.instance_id])
    if not instances:
        print(f"ERROR: instance '{args.instance_id}' not found in {args.data_file}")
        sys.exit(1)

    inst = instances[0]

    # ── 3. Inspect prompt & task_info ────────────────────────────
    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)

    print_section("INSTANCE", json.dumps({
        "id": inst.id,
        "repo": inst.repo,
        "image": inst.image,
        "workdir": inst.workdir,
        "base_commit": inst.base_commit,
        "task_type": inst.metadata.get("task_type"),
    }, indent=2))
    print_section("TASK INFO", json.dumps(task_info, indent=2))
    print_section("PROMPT", prompt)
    print_section("SETUP COMMANDS", "\n".join(task.get_setup_commands(inst)) or "(none)")

    if args.mode == "prompt":
        print("\n[prompt mode] Done. Use --mode dry-run or --mode run to go further.")
        return

    # ── 4. Create runtime & container ────────────────────────────
    image = task.get_image(inst)
    runtime_config = RuntimeConfig(backend="docker", image=image, workdir=inst.workdir)
    runtime = DockerRuntime(runtime_config)

    async with runtime.session(image) as session:
        # Pre-agent setup (setup_commands + commit snapshot + remove future commits)
        setup = PreAgentSetup(session, inst.workdir)
        pre_agent_commit_id = await setup.prepare(inst)

        if args.mode == "dry-run":
            r = await session.execute(f"ls {inst.workdir}")
            print_section("CONTAINER LS", r.stdout)
            print("\n[dry-run mode] Container ready. Use --mode run to execute agent.")
            return

        # ── 5. Run agent ─────────────────────────────────────────
        from awe_agent.core.condenser import build_condenser

        # Build search constraints from task instance
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
            condenser=condenser,
        )
        loop = AgentLoop(agent, ctx)

        print(f"\nStarting agent (max_steps={config.agent.max_steps}, "
              f"model={config.llm.model}) ...")
        result = await loop.run(prompt)

        # ── 6. Print step-by-step trace ──────────────────────────
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

        # ── 7. Summary ───────────────────────────────────────────
        print_section("RESULT", json.dumps({
            "finish_reason": result.finish_reason,
            "steps": len(result.trajectory.steps),
            "patch_length": len(result.patch),
            "error": result.error,
        }, indent=2))

        if result.patch:
            print_section("PATCH", result.patch)

    # ── 8. Evaluate (outside agent session — container released) ──
    if args.mode == "run" and not args.skip_eval:
        # if result.patch:
        if True:
            from awe_agent.tasks.beyond_swe.evaluator import BeyondSWEEvaluator

            eval_runtime = DockerRuntime(RuntimeConfig(
                backend="docker", image=image, workdir=inst.workdir,
            ))
            evaluator = BeyondSWEEvaluator(timeout=3600)
            eval_result = await evaluator.evaluate(inst, result.patch, eval_runtime)
            print_section("EVAL RESULT", json.dumps({
                "accepted": eval_result.accepted,
                "score": eval_result.score,
                "duration": eval_result.duration,
                "details": eval_result.details,
            }, indent=2, default=str))
        else:
            print("\n[eval] No patch produced — skipping evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
