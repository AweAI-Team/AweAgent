"""DeNovoSWE recipe — unified entry point for prompt inspection, debug, and batch runs.

Modes:

  prompt   — Print the generated prompt and task_info for a single instance (no Docker)
  debug    — Full single-instance run (agent + eval) with detailed trace
  batch    — Batch concurrent execution via TaskRunner, results in JSONL
  dry-run  — List all instances without executing anything

Key CLI overrides (all optional, default from YAML config):

  --config / -c       YAML config file (default: configs/tasks/denovoswe.yaml)
  --model             LLM model name
  --max-steps         Max agent steps per instance
  --max-concurrent    Max parallel instances (batch mode)
  --output            Output directory (batch mode)
  --enable-search     Force enable search tools
  --no-search         Force disable search tools
  --skip-eval         Skip evaluation after agent run
  --validate-run      Skip agent, run evaluation only (verify test patches)
  --del-done-images   Delete docker image after each instance completes
  --eval-iters        Run evaluator N times per instance and report the mean
  --prompt-version    Prompt version: v1 (original) or v2 (default — adds
                      public-API verification gate)
  --dump-clean-snapshot PATH  Dump post-clean workspace snapshots to JSONL
  --verbose           DEBUG level logging

Usage examples:

    # Inspect prompt (no Docker needed)
    python recipes/denovo_swe/run.py \\
        --data-file data.jsonl --instance-id inst_001 --mode prompt

    # Debug single instance
    python recipes/denovo_swe/run.py \\
        --data-file data.jsonl --instance-id inst_001 --mode debug --verbose

    # Batch run
    python recipes/denovo_swe/run.py \\
        --data-file data.jsonl --mode batch --max-concurrent 50

    # Validate run — verify test patches without agent
    python recipes/denovo_swe/run.py \\
        --data-file data.jsonl --mode batch --validate-run

    # List instances
    python recipes/denovo_swe/run.py \\
        --data-file data.jsonl --mode dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from aweagent.core.config.loader import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeNovoSWE recipe — unified entry point")
    p.add_argument("--data-file", required=True, help="Path to JSONL data file")
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/denovoswe.yaml",
        help="Path to YAML config (default: configs/tasks/denovoswe.yaml)",
    )
    p.add_argument(
        "--mode",
        choices=["prompt", "debug", "batch", "dry-run"],
        default="prompt",
        help="prompt|debug|batch|dry-run (default: prompt)",
    )
    p.add_argument("--instance-id", default=None, help="Single instance ID (prompt/debug)")
    p.add_argument("--instance-ids", nargs="*", default=None, help="Instance IDs (batch, optional)")
    p.add_argument("--llm-config", default=None, help="Path to LLM config YAML (overrides LLM_CONFIG env var)")
    p.add_argument("--model", default=None, help="Override LLM model")
    p.add_argument("--max-steps", type=int, default=None, help="Override max agent steps")
    p.add_argument("--max-concurrent", type=int, default=None, help="Override concurrency (batch)")
    p.add_argument(
        "--enable-search", default=None, action="store_true",
        help="Enable search tools",
    )
    p.add_argument(
        "--no-search", dest="enable_search", action="store_false",
        help="Disable search tools",
    )
    p.add_argument("--output", default=None, help="Output directory (batch)")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    p.add_argument(
        "--validate-run", action="store_true",
        help="Skip agent execution, run evaluation only (verify test patches)",
    )
    p.add_argument(
        "--eval-iters", type=int, default=1,
        help="Run the evaluator N times per instance in separate sandboxes "
             "and report the mean score (mitigates eval flakiness). Default 1.",
    )
    p.add_argument("--no-trajectories", action="store_true", help="Disable saving per-instance trajectory files")
    p.add_argument(
        "--del-done-images", action="store_true", default=False,
        help="Delete docker image after each instance completes (saves disk)",
    )
    p.add_argument(
        "--dump-clean-snapshot", default=None,
        help="Path to JSONL file to dump post-clean file snapshots (for inspection)",
    )
    p.add_argument(
        "--prompt-version", default="v2",
        help="Prompt version: v1 (original) or v2 (with public API verification gate). Default: v2",
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG level logging")
    return p.parse_args()


def _load_config(args: argparse.Namespace):
    """Load and apply CLI overrides to the YAML config."""
    if args.llm_config is not None:
        llm_abs = Path(args.llm_config).resolve()
        task_dir = Path(args.config).resolve().parent
        try:
            os.environ["LLM_CONFIG"] = str(os.path.relpath(llm_abs, task_dir))
        except ValueError:
            os.environ["LLM_CONFIG"] = str(llm_abs)

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


def _build_task(
    config, data_file: str,
    validate_run: bool = False,
    del_done_images: bool = False, clean_snapshot_file: str | None = None,
    prompt_version: str = "v2", eval_iters: int = 1,
):
    from aweagent.tasks.denovo_swe.task import DeNovoSWETask

    return DeNovoSWETask(
        dataset_id=config.task.dataset_id,
        data_file=data_file,
        search_mode=config.agent.enable_search,
        validate_run=validate_run,
        del_done_images=del_done_images,
        clean_snapshot_file=clean_snapshot_file,
        prompt_version=prompt_version,
        eval_iters=eval_iters,
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
        has_patch = bool(inst.metadata.get("test_patch"))
        n_tests = len(inst.metadata.get("passed_ptp", [])) + len(inst.metadata.get("failed_ptp", []))
        print(f"  {inst.id}  tests={n_tests}  has_patch={has_patch}  image={inst.image[:60] if inst.image else 'none'}")


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
        "n_tests": len(inst.metadata.get("passed_ptp", [])) + len(inst.metadata.get("failed_ptp", [])),
        "has_test_patch": bool(inst.metadata.get("test_patch")),
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2))
    _print_section("PROMPT", prompt)
    _print_section("SETUP COMMANDS", "\n".join(task.get_setup_commands(inst)) or "(none)")


def _build_debug_run_dir(output_base: str, instance_id: str) -> Path:
    """``output_base/debug_{instance_id}_{YYYYMMDD_HHMMSS}``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_id = instance_id.replace("/", "_").replace(" ", "_")
    run_dir = Path(output_base) / f"debug_{safe_id}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def _mode_debug(
    config, task, instance_id: str, skip_eval: bool, validate_run: bool,
    eval_iters: int = 1,
) -> None:
    from aweagent.core.agent import AgentContext, AgentLoop
    from aweagent.core.condenser import build_condenser
    from aweagent.core.eval.setup import PreAgentSetup
    from aweagent.core.llm import LLMClient
    from aweagent.core.task.runner import runtime_registry
    from aweagent.scaffold.search_swe import SearchSWEAgent

    instances = task.get_instances(instance_ids=[instance_id])
    if not instances:
        print(f"ERROR: instance '{instance_id}' not found")
        sys.exit(1)

    inst = instances[0]

    output_base = config.execution.output_path
    run_dir = _build_debug_run_dir(output_base, instance_id)
    print(f"Output: {run_dir}")

    log_file = run_dir / "debug.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)
    _print_section("INSTANCE", json.dumps({
        "id": inst.id,
        "repo": inst.repo,
        "image": inst.image,
        "workdir": inst.workdir,
        "base_commit": inst.base_commit,
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2))

    if not validate_run:
        _print_section("PROMPT", prompt)

    with open(run_dir / "run_config.json", "w") as f:
        json.dump({
            "instance_id": instance_id,
            "mode": "debug",
            "validate_run": validate_run,
            "skip_eval": skip_eval,
            "model": config.llm.model,
            "max_steps": config.agent.max_steps,
        }, f, indent=2)

    image = task.get_image(inst)
    runtime_config = config.runtime.model_copy(
        update={"image": image, "workdir": inst.workdir},
    )
    runtime_cls = runtime_registry.get(config.runtime.backend)
    runtime = runtime_cls(runtime_config)

    agent_patch = ""
    agent_timeout = False
    agent_result = None

    if not validate_run:
        try:
            async with runtime.session(image) as session:
                setup = PreAgentSetup(session, inst.workdir)
                await setup.prepare(inst)

                # Dump post-clean snapshot into the debug run dir
                task._clean_snapshot_file = str(run_dir / "post_clean_snapshot.jsonl")
                await task.prepare_session(inst, session)

                # Re-capture commit id after clean (clean.sh reinitializes .git)
                post_clean_commit = await session.execute(
                    "git rev-parse HEAD", cwd=inst.workdir, timeout=10,
                )
                if not post_clean_commit.success or not post_clean_commit.stdout.strip():
                    print(
                        f"ERROR: failed to get post-clean commit id: "
                        f"{post_clean_commit.stderr[:200]}",
                    )
                    sys.exit(1)
                pre_agent_commit_id = post_clean_commit.stdout.strip()

                search_constraints = task.get_search_constraints(inst)
                agent = SearchSWEAgent(
                    enable_search=config.agent.enable_search,
                    bash_timeout=config.agent.bash_timeout,
                    bash_max_timeout=config.agent.bash_max_timeout,
                    max_output_length=config.agent.max_output_length,
                    bash_blocklist=config.security.bash_blocklist or None,
                    search_constraints=search_constraints,
                    tool_call_format=config.agent.tool_call_format,
                )
                llm = LLMClient(config.llm)
                condenser = build_condenser(config.agent.condenser)
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
                agent_result = await loop.run(prompt)

                for step in agent_result.trajectory.steps:
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

                _print_section("RESULT", json.dumps({
                    "finish_reason": agent_result.finish_reason,
                    "steps": len(agent_result.trajectory.steps),
                    "patch_length": len(agent_result.patch),
                    "error": agent_result.error,
                }, indent=2))

                if agent_result.patch:
                    _print_section("PATCH", agent_result.patch)
                agent_patch = agent_result.patch

        except (TimeoutError, asyncio.TimeoutError):
            agent_timeout = True
            print(
                f"\n** SESSION TIMEOUT: agent exceeded runtime.timeout "
                f"({config.runtime.timeout}s) **",
            )
            print("Container killed. Continuing to evaluation with whatever patch was extracted.")

        if agent_result is not None and agent_result.patch:
            (run_dir / "agent.patch").write_text(agent_result.patch)

    else:
        print("\n[validate-run] Skipping agent execution — going straight to evaluation.")

    eval_data = None
    if not skip_eval:
        from aweagent.tasks.denovo_swe.evaluator import DeNovoSWEEvaluator

        eval_runtime_cls = runtime_registry.get(config.runtime.backend)
        eval_runtime = eval_runtime_cls(config.runtime.model_copy(
            update={"image": image, "workdir": inst.workdir},
        ))
        evaluator = DeNovoSWEEvaluator(
            timeout=config.eval.timeout,
            validate_run=validate_run,
            eval_iters=eval_iters,
        )
        eval_result = await evaluator.evaluate(inst, agent_patch, eval_runtime)
        eval_data = {
            "accepted": eval_result.accepted,
            "score": eval_result.score,
            "duration": eval_result.duration,
            "details": eval_result.details,
        }
        _print_section("EVAL RESULT", json.dumps(eval_data, indent=2, default=str))
    else:
        print("\n[eval] Skipped (--skip-eval).")

    finish_reason = "timeout" if agent_timeout else (
        agent_result.finish_reason if agent_result else None
    )

    if agent_result is not None:
        initial_messages = []
        for msg in agent_result.messages:
            if msg.role in ("system", "user"):
                initial_messages.append(msg.to_dict())
            else:
                break

        trajectory_data = {
            "instance_id": inst.id,
            "success": eval_data["accepted"] if eval_data else False,
            "score": eval_data["score"] if eval_data else 0.0,
            "finish_reason": finish_reason,
            "error": agent_result.error,
            "duration": eval_data["duration"] if eval_data else None,
            "initial_messages": initial_messages,
            "patch": agent_result.patch,
            "stats": agent_result.metadata.get("stats"),
            "trajectory": [
                {
                    "step": step.step,
                    "action": {
                        "type": step.action.type,
                        "content": step.action.content
                            if step.action.content and step.action.content.strip() else None,
                        "tool_calls": step.action.tool_calls,
                        "reasoning_text": step.action.thinking,
                    },
                    "observations": step.observations,
                }
                for step in agent_result.trajectory.steps
            ],
            "eval_result": eval_data,
        }
    else:
        trajectory_data = {
            "instance_id": inst.id,
            "success": eval_data["accepted"] if eval_data else False,
            "score": eval_data["score"] if eval_data else 0.0,
            "finish_reason": finish_reason,
            "error": (
                f"Session timeout after {config.runtime.timeout}s"
                if agent_timeout else None
            ),
            "duration": eval_data["duration"] if eval_data else None,
            "initial_messages": [],
            "patch": "",
            "stats": None,
            "trajectory": [],
            "eval_result": eval_data,
        }
    with open(run_dir / "trajectory.json", "w") as f:
        json.dump(trajectory_data, f, indent=2, default=str)

    final_result = {
        "instance_id": inst.id,
        "dataset_id": inst.dataset_id,
        "repo": inst.repo,
        "success": eval_data["accepted"] if eval_data else False,
        "score": eval_data["score"] if eval_data else 0.0,
        "error": "session_timeout" if agent_timeout else None,
        "finish_reason": finish_reason,
        "eval_result": eval_data,
        "validate_run": validate_run,
    }
    with open(run_dir / "results.jsonl", "w") as f:
        f.write(json.dumps(final_result, default=str) + "\n")

    print(f"\nAll outputs saved to: {run_dir}")


async def _mode_batch(
    config, task, instance_ids: list[str] | None, skip_eval: bool,
    save_trajectories: bool = True,
) -> None:
    from aweagent.core.condenser import build_condenser
    from aweagent.core.task.runner import TaskRunner
    from aweagent.scaffold.registry import agent_registry

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
    task = _build_task(
        config, args.data_file,
        validate_run=args.validate_run,
        del_done_images=args.del_done_images,
        clean_snapshot_file=args.dump_clean_snapshot,
        prompt_version=args.prompt_version,
        eval_iters=args.eval_iters,
    )

    print(f"LLM:    backend={config.llm.backend}, model={config.llm.model}")
    print(f"Agent:  type={config.agent.type}, max_steps={config.agent.max_steps}, "
          f"search={config.agent.enable_search}, prompt_version={args.prompt_version}")
    print(f"Mode:   {args.mode}")
    if args.validate_run:
        print("** VALIDATE-RUN mode: agent will be skipped, evaluation only **")

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
        await _mode_debug(
            config, task, args.instance_id, args.skip_eval, args.validate_run,
            eval_iters=args.eval_iters,
        )

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
