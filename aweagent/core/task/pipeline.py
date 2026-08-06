"""Shared run pipeline — the single place that wires config → runner → results.

Before this module, the assembly (agent factory, evaluator selection, condenser,
TaskRunner construction, run_all) was copy-pasted across ``cli._cmd_run`` and the
four ``recipes/*/run.py`` ``_mode_batch`` functions. They drifted. This module is
the one source of truth; ``cli`` and the recipes call into it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from aweagent.core.condenser import build_condenser
from aweagent.core.task.registry import task_registry
from aweagent.core.task.runner import TaskRunner
from aweagent.scaffold.registry import agent_registry

if TYPE_CHECKING:
    from aweagent.core.agent.protocol import Agent
    from aweagent.core.config.schema import AweAgentConfig
    from aweagent.core.task.protocol import Evaluator, Task
    from aweagent.core.task.types import TaskResult


def build_task(config: AweAgentConfig) -> Task:
    """Resolve and construct the task for ``config.task.type`` via the registry."""
    return task_registry.get(config.task.type).from_config(config)


def build_agent_factory(config: AweAgentConfig) -> Callable[..., Agent]:
    """Build the agent factory closure (formerly duplicated 5x).

    Returns a callable accepting an optional ``search_constraints`` kwarg for
    per-instance constraint injection.
    """
    agent_cls = agent_registry.get(config.agent.type)

    def factory(search_constraints=None):
        if search_constraints and hasattr(agent_cls, "from_config_with_constraints"):
            return agent_cls.from_config_with_constraints(config, search_constraints)
        return agent_cls.from_config(config)

    return factory


def build_evaluator(
    config: AweAgentConfig,
    task: Task,
    *,
    skip_eval: bool = False,
) -> Evaluator | None:
    """Select the evaluator: task-specific first, generic isolated as fallback."""
    if skip_eval or not config.eval.enabled:
        return None

    task_eval = task.default_evaluator(timeout=config.eval.timeout)
    if task_eval is not None:
        return task_eval

    if config.eval.isolated:
        from aweagent.core.eval.isolation import IsolatedEvaluator

        return IsolatedEvaluator(eval_script=config.eval.eval_script)

    return None


def build_runner(
    config: AweAgentConfig,
    task: Task,
    *,
    skip_eval: bool = False,
    save_trajectories: bool = True,
) -> TaskRunner:
    """Construct a TaskRunner from config (replaces 5 duplicated call sites)."""
    return TaskRunner(
        task=task,
        agent_factory=build_agent_factory(config),
        llm_config=config.llm,
        runtime_config=config.runtime,
        evaluator=build_evaluator(config, task, skip_eval=skip_eval),
        max_concurrent=config.execution.max_concurrent,
        start_index=config.execution.start_index,
        end_index=config.execution.end_index,
        max_instances=config.execution.max_instances,
        max_retries=config.execution.max_retries,
        output_path=config.execution.output_path,
        condenser=build_condenser(config.agent.condenser),
        save_trajectories=save_trajectories and config.execution.save_trajectories,
        num_rollouts=config.execution.num_rollouts,
        config_snapshot=json.loads(config.model_dump_json()),
        max_steps=config.agent.max_steps,
        max_context_length=config.agent.max_context_length,
        agent_timeout_override=config.task.override_agent_timeout,
        evaluation_enabled=not skip_eval and config.eval.enabled,
    )


async def run_pipeline(
    config: AweAgentConfig,
    *,
    instance_ids: list[str] | None = None,
    skip_eval: bool = False,
    save_trajectories: bool = True,
) -> tuple[TaskRunner, list[TaskResult]]:
    """Full path: build task via registry → build runner → run_all.

    Returns the runner (for ``run_dir``) and the list of results.
    """
    task = build_task(config)
    runner = build_runner(
        config, task, skip_eval=skip_eval, save_trajectories=save_trajectories
    )
    results = await runner.run_all(instance_ids)
    return runner, results
