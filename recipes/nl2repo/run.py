"""NL2Repo recipe — unified entry point for prompt inspection, debug, and batch runs.

Faithful port of the NL2RepoBench launcher (``main.py`` +
``openhands/openhands_app.start_openhands``) onto the AweAgent task
framework.  Mirrors the structure of ``recipes/scale_swe/run.py`` so that
the four modes (prompt / debug / batch / dry-run) work the same way.

Modes:

  prompt   — Print the generated prompt and task_info for a single instance (no Docker)
  debug    — Full single-instance run (agent + eval) with detailed trace
  batch    — Batch concurrent execution via TaskRunner, results in JSONL
  dry-run  — List all instances without executing anything

Key CLI overrides (all optional, default from YAML config):

  --config / -c        YAML config file (default: configs/tasks/nl2repo.yaml)
  --data-file          JSONL or CSV with per-instance evaluation images / spec /
                       verify cmds (e.g. nl2repo.jsonl).  Required unless
                       DATA_FILE env var is set.
  --agent-run-docker   Single docker image for the *agent* side on every
                       instance.  When set, agents run in this image instead
                       of each instance's evaluation image.  Evaluation still
                       uses the per-instance image from --data-file.
                       Empty string (the default) means the agent shares
                       the evaluation image.
  --instance-id        Single instance ID for prompt/debug
  --instance-ids       Subset of instance IDs for batch
  --model              LLM model name
  --max-steps          Max agent steps per instance
  --max-concurrent     Max parallel instances (batch mode) — equivalent of
                       NL2RepoBench's ``max_pool_size``
  --output             Output directory (batch mode)
  --skip-eval          Skip evaluation after agent run
  --verbose            DEBUG level logging

Environment variables:

  DATA_FILE                Default JSONL/CSV path if --data-file omitted
  AGENT_RUN_DOCKER         Default agent image override if --agent-run-docker
                           omitted (empty string disables the override)

Usage examples:

    # Inspect prompt (no Docker needed)
    python recipes/nl2repo/run.py \\
        --data-file /path/to/data/nl2repo/nl2repo.jsonl \\
        --instance-id aiofiles --mode prompt

    # Debug single instance
    python recipes/nl2repo/run.py \\
        --data-file /path/to/data/nl2repo/nl2repo.jsonl \\
        --instance-id aiofiles --mode debug --verbose

    # Batch run
    python recipes/nl2repo/run.py \\
        --data-file /path/to/data/nl2repo/nl2repo.jsonl \\
        --mode batch

    # List instances
    python recipes/nl2repo/run.py \\
        --data-file /path/to/data/nl2repo/nl2repo.jsonl \\
        --mode dry-run
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

from aweagent.core.config.loader import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NL2Repo recipe — unified entry point")
    p.add_argument("--data-file", default=None, help="Path to JSONL or CSV data file")
    p.add_argument(
        "--config", "-c",
        default="configs/tasks/nl2repo.yaml",
        help="Path to YAML config (default: configs/tasks/nl2repo.yaml)",
    )
    p.add_argument(
        "--agent-run-docker",
        default=None,
        help=(
            "Single docker image for the agent side on every instance. "
            "Evaluation still uses the per-instance image from --data-file."
        ),
    )
    p.add_argument(
        "--mode",
        choices=["prompt", "debug", "batch", "dry-run"],
        default="prompt",
        help="prompt|debug|batch|dry-run (default: prompt)",
    )
    p.add_argument("--instance-id", default=None, help="Single instance ID (prompt/debug)")
    p.add_argument(
        "--instance-ids", nargs="*", default=None,
        help="Instance IDs (batch, optional)",
    )
    p.add_argument("--llm-config", default=None, help="Path to LLM config YAML (overrides LLM_CONFIG env var)")
    p.add_argument("--model", default=None, help="Override LLM model")
    p.add_argument("--max-steps", type=int, default=None, help="Override max agent steps")
    p.add_argument(
        "--max-concurrent", type=int, default=None,
        help="Override concurrency (batch) — equivalent of NL2RepoBench max_pool_size",
    )
    p.add_argument("--output", default=None, help="Output directory (batch)")
    p.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    p.add_argument(
        "--no-trajectories", action="store_true",
        help="Disable saving per-instance trajectory files",
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG level logging")
    return p.parse_args()


def _load_config(args: argparse.Namespace):
    """Load and apply CLI overrides to the YAML config."""
    if args.llm_config is not None:
        from pathlib import Path
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
    if args.output is not None:
        overrides.setdefault("execution", {})["output_path"] = args.output

    if args.data_file is not None:
        os.environ["DATA_FILE"] = args.data_file
        overrides.setdefault("task", {})["data_file"] = args.data_file
    if args.agent_run_docker is not None:
        os.environ["AGENT_RUN_DOCKER"] = args.agent_run_docker
        overrides.setdefault("task", {})["agent_run_docker"] = args.agent_run_docker

    return load_config(args.config, overrides=overrides)


def _build_task(config, args: argparse.Namespace):
    # data_file / agent_run_docker are threaded into config.task.* by
    # _load_config; build via the shared registry.
    from aweagent.core.task.pipeline import build_task

    return build_task(config)


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
        pro_name = inst.metadata.get("pro_name", inst.id)
        n_tests = inst.metadata.get("test_case_count", 0)
        eval_image = inst.metadata.get("eval_image", inst.image)
        agent_marker = (
            "" if inst.image == eval_image else f"  agent={inst.image}"
        )
        print(
            f"  {inst.id}  pro={pro_name}  test_cases={n_tests}  "
            f"eval={eval_image}{agent_marker}"
        )


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
        "pro_name": inst.metadata.get("pro_name"),
        "agent_image": inst.image,
        "eval_image": inst.metadata.get("eval_image"),
        "workdir": inst.workdir,
        "test_case_count": inst.metadata.get("test_case_count"),
        "test_shell": inst.metadata.get("test_shell"),
        "py_test_file_list": inst.metadata.get("py_test_file_list"),
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2))
    _print_section("PROMPT", prompt)
    _print_section("START.MD (preview)", inst.metadata.get("start_md", ""), max_len=1500)


async def _mode_debug(config, task, instance_id: str, skip_eval: bool) -> None:
    from aweagent.core.agent import AgentContext, AgentLoop
    from aweagent.core.condenser import build_condenser
    from aweagent.core.eval.setup import PreAgentSetup
    from aweagent.core.llm import LLMClient
    from aweagent.core.runtime import RuntimeConfig
    from aweagent.core.runtime.docker import DockerRuntime
    from aweagent.scaffold.search_swe import SearchSWEAgent

    instances = task.get_instances(instance_ids=[instance_id])
    if not instances:
        print(f"ERROR: instance '{instance_id}' not found")
        sys.exit(1)

    inst = instances[0]

    prompt = task.get_prompt(inst)
    task_info = task.get_task_info(inst)
    _print_section("INSTANCE", json.dumps({
        "id": inst.id,
        "pro_name": inst.metadata.get("pro_name"),
        "agent_image": inst.image,
        "eval_image": inst.metadata.get("eval_image"),
        "workdir": inst.workdir,
    }, indent=2))
    _print_section("TASK INFO", json.dumps(task_info, indent=2))
    _print_section("PROMPT", prompt)

    image = task.get_image(inst)
    runtime_config = RuntimeConfig(backend="docker", image=image, workdir=inst.workdir)
    runtime = DockerRuntime(runtime_config)

    artifact_bytes: bytes | None = None
    async with runtime.session(image) as session:
        setup = PreAgentSetup(session, inst.workdir)
        await setup.run_setup_commands(task.get_setup_commands(inst))
        await task.prepare_session(inst, session)

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

        if not task.requires_patch_extraction():
            task_info["skip_patch_extraction"] = True

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

        print(
            f"\nStarting agent (max_steps={config.agent.max_steps}, "
            f"model={config.llm.model}) ..."
        )
        result = await loop.run(prompt)

        for step in result.trajectory.steps:
            print(f"\n{'─' * 50}")
            print(f"  Step {step.step}  |  action={step.action.type}")
            print(f"{'─' * 50}")
            if step.action.content:
                print(f"  [content] {step.action.content[:500]}")
            if step.action.tool_calls:
                for tc in step.action.tool_calls:
                    name = tc.get("name", tc.get("function", {}).get("name", "?"))
                    raw_args = tc.get("arguments", tc.get("function", {}).get("arguments", ""))
                    print(f"  [tool] {name}")
                    print(f"    args: {str(raw_args)[:300]}")
            for i, obs in enumerate(step.observations):
                print(f"  [obs {i}] {obs[:500]}")

        _print_section("RESULT", json.dumps({
            "finish_reason": result.finish_reason,
            "steps": len(result.trajectory.steps),
            "patch_length": len(result.patch),
            "error": result.error,
        }, indent=2))

        if result.patch:
            _print_section("PATCH", result.patch)

        try:
            artifact_bytes = await task.collect_artifact(inst, session)
        except Exception as e:
            print(f"\n[collect_artifact] failed: {e}")

    if not skip_eval:
        from aweagent.tasks.nl2repo.evaluator import NL2RepoEvaluator

        if artifact_bytes is not None:
            inst.metadata["_agent_artifact"] = artifact_bytes

        eval_runtime = DockerRuntime(RuntimeConfig(
            backend="docker", image=image, workdir=inst.workdir,
        ))
        evaluator = NL2RepoEvaluator(timeout=config.eval.timeout)
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
    from aweagent.core.task.pipeline import build_runner

    runner = build_runner(
        config, task, skip_eval=skip_eval, save_trajectories=save_trajectories
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
    task = _build_task(config, args)

    print(f"LLM:    backend={config.llm.backend}, model={config.llm.model}")
    print(
        f"Agent:  type={config.agent.type}, max_steps={config.agent.max_steps}, "
        f"search={config.agent.enable_search}"
    )
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
