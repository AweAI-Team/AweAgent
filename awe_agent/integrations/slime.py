"""Slime RL framework integration.

Provides a rollout interface that bridges AweAgent with the Slime training
framework.  The Slime ``RolloutManager`` starts SGLang engines and a router,
then invokes the rollout function with:

- ``args`` — training configuration (tokenizer path, concurrency, sampling
  params, SGLang router endpoint, etc.)
- ``rollout_id`` — monotonically increasing counter.
- ``data_buffer`` — Slime's sample lifecycle manager
  (``.get_samples()`` / ``.add_samples()``).

This module provides two layers:

1. **Core** — :func:`generate_single` processes one :class:`Sample` through
   the full AweAgent pipeline (runtime → setup → agent → evaluate) and fills
   the Sample with token-level RL data.

2. **Entry point** — :func:`generate_rollout_fully_async` is the continuous
   async rollout function that Slime calls.  It creates a background
   ``AsyncRolloutWorker`` that pulls samples from ``data_buffer`` and
   processes them concurrently.

Data flow::

    Slime RolloutManager
      │
      ├── manages SGLang engines + router
      │
      └── calls generate_rollout_fully_async(args, rollout_id, data_buffer)
            │
            └── AsyncRolloutWorker (daemon thread, continuous event loop)
                  │
                  └── for each Sample group from data_buffer:
                        generate_single(args, sample, sampling_params)
                          ├── create runtime session
                          ├── PreAgentSetup
                          ├── AgentLoop.run() with TrainingState
                          ├── evaluate patch → reward
                          └── fill Sample fields from TrainingState
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import os.path as osp
import queue
import threading
import time
import traceback
from argparse import Namespace
from typing import Any

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentLoop, AgentResult
from awe_agent.core.agent.training import TrainingState
from awe_agent.core.eval.setup import PreAgentSetup
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.runtime.protocol import Runtime
from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.runner import runtime_registry
from awe_agent.core.task.types import Instance

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

POS_REWARD = 1.0
NEG_REWARD = -1.0
MAX_POD_RETRY = int(os.getenv("AWE_AGENT_MAX_POD_RETRY", "3"))
MAX_ITERATIONS = int(os.getenv("AWE_AGENT_MAX_ITERATIONS", "100"))


# ── Singleton state ───────────────────────────────────────────────────


class _SingletonMeta(type):
    """Metaclass for lazy singletons — instantiated once per ``args`` set."""

    _instances: dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class GenerateState(metaclass=_SingletonMeta):
    """Global state for the generation process (singleton).

    Holds the tokenizer (loaded from ``args.hf_checkpoint``), a concurrency
    semaphore, and the SGLang sampling parameters.  Mirrors the
    ``GenerateState`` in the RL rollout generator.
    """

    def __init__(self, args: Namespace) -> None:
        from transformers import AutoTokenizer

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.hf_checkpoint, trust_remote_code=True,
        )

        total_concurrency = (
            args.sglang_server_concurrency
            * args.rollout_num_gpus
            // args.rollout_num_gpus_per_engine
        )
        self.semaphore = asyncio.Semaphore(total_concurrency)
        logger.info("Rollout concurrency: %d", total_concurrency)

        self.sampling_params: dict[str, Any] = {
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "top_k": args.rollout_top_k,
            "max_new_tokens": args.rollout_max_response_len,
            "stop": getattr(args, "rollout_stop", None),
            "stop_token_ids": getattr(args, "rollout_stop_token_ids", None),
            "skip_special_tokens": getattr(args, "rollout_skip_special_tokens", False),
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

        # Shared runtime — one instance (and underlying client / connection pool)
        # reused across all concurrent rollout sessions.
        self._runtime_config = _get_runtime_config()
        runtime_cls = runtime_registry.get(self._runtime_config.backend)
        self.runtime: Runtime = runtime_cls(self._runtime_config)

    def reset(self) -> None:
        self.aborted = False

    aborted: bool = False


# ── Task / Agent factories ────────────────────────────────────────────
#
# These functions create the Task, Agent, Runtime, and Evaluator objects
# from environment variables.  They allow the rollout function to be
# configured without touching code via AGENT_CLASS / MAX_ITERATIONS env vars.

def _get_task() -> Task:
    """Instantiate the task from environment configuration."""
    task_class = os.getenv("AWE_AGENT_TASK_CLASS", "ScaleSWETask")
    data_file = os.getenv("AWE_AGENT_DATA_FILE", "")

    if task_class == "ScaleSWETask":
        from awe_agent.tasks.scale_swe.task import ScaleSWETask
        return ScaleSWETask(data_file=data_file)
    elif task_class == "BeyondSWETask":
        from awe_agent.tasks.beyond_swe.task import BeyondSWETask
        search_mode = os.getenv("AWE_AGENT_SEARCH_MODE", "false").lower() in ("true", "1")
        test_suite_dir = os.getenv("BEYONDSWE_TEST_SUITE_DIR", "")
        return BeyondSWETask(
            data_file=data_file,
            search_mode=search_mode,
            test_suite_dir=test_suite_dir,
        )
    else:
        raise ValueError(f"Unknown task class: {task_class}")


def _create_agent():
    """Create a SearchSWEAgent from environment configuration."""
    from awe_agent.scaffold.search_swe.agent import SearchSWEAgent

    enable_search = os.getenv("AWE_AGENT_ENABLE_SEARCH", "false").lower() in ("true", "1")
    tool_call_format = os.getenv("AWE_AGENT_TOOL_CALL_FORMAT", "codeact_xml")
    return SearchSWEAgent(
        enable_search=enable_search,
        tool_call_format=tool_call_format,
    )


def _get_runtime_config() -> RuntimeConfig:
    """Build runtime config from environment.

    For the ``portal`` backend, populates ``extra`` with portal-specific
    configuration read from ``AWE_AGENT_PORTAL_*`` / ``AWE_AGENT_SANDBOX_*``
    environment variables.  These keys are only interpreted by ``PortalRuntime``
    in AweAgent-internal.
    """
    backend = os.getenv("AWE_AGENT_RUNTIME_BACKEND", "docker")
    image = os.getenv("AWE_AGENT_RUNTIME_IMAGE", "")
    timeout = int(os.getenv("AWE_AGENT_RUNTIME_TIMEOUT", "14400"))
    extra: dict[str, Any] = {}
    if backend == "portal":
        extra = {
            "region": os.getenv("AWE_AGENT_PORTAL_REGION", ""),
            "sandbox_psm": os.getenv("AWE_AGENT_SANDBOX_PSM", ""),
            "sandbox_id": os.getenv("AWE_AGENT_SANDBOX_ID", ""),
            "zti_token": os.getenv("AWE_AGENT_ZTI_TOKEN", ""),
            "portal_wait_timeout": int(os.getenv("AWE_AGENT_PORTAL_WAIT_TIMEOUT", "120")),
        }
    return RuntimeConfig(backend=backend, image=image, timeout=timeout, extra=extra)


# ── Core: single-sample processing ───────────────────────────────────


async def generate_single(
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
    *,
    task: Task | None = None,
    evaluator: Evaluator | None = None,
    evaluation: bool = False,
) -> Any:
    """Process a single Slime Sample through the full AweAgent pipeline.

    This is the core building block.  It:

    1. Creates a runtime session with the task's Docker image.
    2. Runs ``PreAgentSetup`` (setup commands + commit snapshot).
    3. Creates an ``AgentContext`` with a ``TrainingState`` that tracks
       token-level RL data.
    4. Runs the agent loop.
    5. Evaluates the patch and sets the reward.
    6. Fills the ``Sample`` with tokens, loss_mask, logprobs, etc.

    Args:
        args: Slime training args namespace.
        sample: A Slime ``Sample`` object.
        sampling_params: SGLang sampling parameters dict.
        task: Task instance (created from env if None).
        evaluator: Evaluator instance (task default if None).
        evaluation: Whether this is an evaluation rollout.

    Returns:
        The same ``sample`` object, with RL fields populated.
    """
    state = GenerateState(args)

    if task is None:
        task = _get_task()
    if evaluator is None:
        evaluator = task.default_evaluator()

    instance_id = sample.metadata.get("instance_id", "")

    # Resolve the task instance
    instances = task.get_instances([instance_id])
    if not instances:
        logger.error("Instance not found: %s", instance_id)
        sample.status = sample.Status.ABORTED
        return sample

    instance = instances[0]

    # Build LLM config pointing to the SGLang router
    sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    llm_config = LLMConfig(
        backend="sglang",
        base_url=sglang_url,
        model="SGLANG_ENGINE",
        return_tokens=True,
        return_logprobs=True,
        params={
            "sampling_params": sampling_params,
        },
    )

    # Retry loop for transient runtime failures
    result: AgentResult | None = None
    training_state: TrainingState | None = None
    last_exception: Exception | None = None

    for retry in range(MAX_POD_RETRY):
        # Fresh TrainingState per attempt — a failed attempt may leave
        # stale tokens in the accumulator.
        training_state = TrainingState(
            tokenizer=state.tokenizer,
            max_new_tokens=sampling_params.get("max_new_tokens", 32768),
        )
        try:
            result = await _run_agent_on_instance(
                task, instance, llm_config, training_state, state.runtime,
            )
            break
        except Exception as e:
            last_exception = e
            logger.error(
                "Instance %s retry %d/%d failed: %s",
                instance_id, retry + 1, MAX_POD_RETRY,
                traceback.format_exc(),
            )
            if retry < MAX_POD_RETRY - 1:
                await asyncio.sleep((retry + 1) * 2)

    # All retries exhausted
    if result is None:
        logger.error(
            "Instance %s failed after %d retries: %s",
            instance_id, MAX_POD_RETRY, last_exception,
        )
        sample.status = sample.Status.ABORTED
        return sample

    # ── Evaluate and set reward ───────────────────────────────────────
    reward = NEG_REWARD if not evaluation else 0.0
    if evaluator and result.patch:
        try:
            eval_result = await evaluator.evaluate(instance, result.patch, state.runtime)
            if eval_result.accepted:
                reward = POS_REWARD
            elif eval_result.score == 0.0:
                reward = NEG_REWARD if not evaluation else 0.0
        except Exception as e:
            logger.error("Evaluation failed for %s: %s", instance_id, e)

    sample.reward = reward

    # ── Fill Sample from TrainingState ────────────────────────────────
    rl_data = training_state.to_rl_data()
    prompt_ids = rl_data["prompt_token_ids"]
    response_ids = rl_data["response_token_ids"]

    sample.tokens = prompt_ids + response_ids
    sample.response_length = len(response_ids)
    sample.response = rl_data["response_text"]
    sample.loss_mask = rl_data["loss_mask"]
    sample.rollout_log_probs = rl_data["rollout_log_probs"]
    sample.train_metadata = {
        "messages": [m.to_full_dict() for m in result.messages],
        "patch": result.patch,
        "finish_reason": result.finish_reason,
    }
    for wv in rl_data["weight_versions"]:
        sample.weight_versions.append(wv)

    # ── Truncation handling ───────────────────────────────────────────
    # If the total sequence exceeds the token budget, truncate the
    # response portion and mark the sample as TRUNCATED.  Positive
    # rewards are demoted since the agent's answer may be incomplete.
    max_budget = sampling_params.get("max_new_tokens", 32768)
    if len(sample.tokens) > max_budget:
        logger.warning(
            "Instance %s truncated: %d tokens > budget %d",
            instance_id, len(sample.tokens), max_budget,
        )
        if sample.reward == POS_REWARD:
            logger.warning(
                "Instance %s had POS_REWARD but is truncated, demoting to NEG_REWARD",
                instance_id,
            )
            sample.reward = NEG_REWARD

        available = max_budget - len(prompt_ids) - 64
        if available < 0:
            logger.error(
                "Prompt too long for instance %s: %d > %d",
                instance_id, len(prompt_ids), max_budget,
            )
            sample.status = sample.Status.ABORTED
            return sample

        response_ids = response_ids[:available]
        sample.tokens = prompt_ids + response_ids
        sample.response_length = len(response_ids)
        sample.loss_mask = sample.loss_mask[:len(response_ids)]
        sample.rollout_log_probs = sample.rollout_log_probs[:len(response_ids)]
        sample.status = sample.Status.TRUNCATED
    else:
        # Map internal finish status to Slime Sample.Status
        status_map = {
            "stop": sample.Status.COMPLETED,
            "length": sample.Status.TRUNCATED,
            "abort": sample.Status.ABORTED,
        }
        sample.status = status_map.get(
            rl_data["status"], sample.Status.COMPLETED,
        )

    return sample


async def _run_agent_on_instance(
    task: Task,
    instance: Instance,
    llm_config: LLMConfig,
    training_state: TrainingState,
    runtime: Runtime,
) -> AgentResult:
    """Run setup, execute agent, return result using *shared* runtime."""
    image = task.get_image(instance)

    async with runtime.session(image) as session:
        # Pre-agent setup
        setup = PreAgentSetup(session, instance.workdir)
        await setup.run_setup_commands(task.get_setup_commands(instance))
        pre_agent_commit_id = await setup.commit_and_get_id()
        await setup.remove_future_commits()

        # Create agent and context
        agent = _create_agent()
        llm = LLMClient(llm_config)
        task_info = task.get_task_info(instance)
        if pre_agent_commit_id:
            task_info["pre_agent_commit_id"] = pre_agent_commit_id

        context = AgentContext(
            llm=llm,
            session=session,
            tools=agent.get_tools(),
            task_info=task_info,
            max_steps=MAX_ITERATIONS,
            training=training_state,
        )

        loop = AgentLoop(agent, context)
        prompt = task.get_prompt(instance)
        return await loop.run(prompt)


# ── Batch processing ─────────────────────────────────────────────────


async def generate_and_evaluate(
    args: Namespace,
    sample: Any,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Any:
    """Process a single sample (with semaphore for concurrency control).

    Wraps :func:`generate_single` with the global semaphore and abort
    detection from :class:`GenerateState`.
    """
    state = GenerateState(args)

    if state.aborted:
        sample.status = sample.Status.ABORTED
        return sample

    async with state.semaphore:
        if state.aborted:
            sample.status = sample.Status.ABORTED
            return sample

        return await generate_single(
            args, sample, sampling_params, evaluation=evaluation,
        )


async def generate_and_evaluate_group(
    args: Namespace,
    group: list[Any],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> list[Any]:
    """Process a group of samples concurrently."""
    tasks = [
        generate_and_evaluate(args, sample, sampling_params, evaluation=evaluation)
        for sample in group
    ]
    return list(await asyncio.gather(*tasks))


# ── Async rollout worker ─────────────────────────────────────────────


_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args: Namespace, data_buffer: Any) -> AsyncRolloutWorker:
    """Get or create the global continuous rollout worker."""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.is_alive():
            logger.info("Creating new global async rollout worker")
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    """Stop the global worker (called at interpreter exit)."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """Continuous async rollout worker running in a daemon thread.

    Pulls sample groups from ``data_buffer``, processes them through
    AweAgent, and pushes completed groups to an output queue.
    """

    def __init__(self, args: Namespace, data_buffer: Any) -> None:
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Any]]] = queue.Queue(maxsize=1000)
        self._thread: threading.Thread | None = None
        self._state = GenerateState(args)

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._thread_main, daemon=True,
            )
            self._thread.start()
            logger.info("Started continuous async rollout worker thread")

    def stop(self) -> None:
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Stopped async rollout worker thread")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_completed(self) -> list[tuple[int, list[Any]]]:
        """Drain all completed groups from the output queue."""
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    @property
    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # ── Internal ──────────────────────────────────────────────────────

    def _thread_main(self) -> None:
        asyncio.run(self._worker_loop())

    async def _worker_loop(self) -> None:
        logger.info("Continuous async rollout worker started")
        active_tasks: set[asyncio.Task[Any]] = set()
        max_concurrent = self.args.rollout_batch_size
        group_counter = 0

        while self.running:
            try:
                # Clean up finished tasks
                done = {t for t in active_tasks if t.done()}
                for t in done:
                    try:
                        t.result()
                    except Exception as e:
                        logger.error("Rollout task failed: %s", e)
                active_tasks -= done

                # Submit new tasks
                while len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break  # No samples available — wait in the outer sleep
                    for group in groups:
                        gid = group_counter
                        group_counter += 1

                        task = asyncio.create_task(
                            generate_and_evaluate_group(
                                self.args,
                                group,
                                self._state.sampling_params.copy(),
                            ),
                        )

                        def _on_done(t: asyncio.Task[Any], gid: int = gid) -> None:
                            try:
                                self.output_queue.put((gid, t.result()))
                            except Exception:
                                pass

                        task.add_done_callback(_on_done)
                        active_tasks.add(task)
                        break

                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Error in worker loop: %s", e)
                await asyncio.sleep(1)

        # Drain remaining tasks
        if active_tasks:
            logger.info("Waiting for %d remaining tasks", len(active_tasks))
            await asyncio.wait(active_tasks)

        logger.info("Continuous async rollout worker stopped")


# ── Slime entry points ───────────────────────────────────────────────


async def _generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_buffer: Any,
) -> list[list[Any]]:
    """Async implementation of the fully-async rollout generation."""
    worker = _get_global_worker(args, data_buffer)
    target_size = args.rollout_batch_size

    data: list[list[Any]] = []
    completed_groups: dict[int, list[Any]] = {}
    last_progress = time.time()
    no_progress_timeout = 30.0

    logger.info(
        "Rollout %d: collecting %d groups (queue=%d)",
        rollout_id, target_size, worker.queue_size,
    )

    while len(data) < target_size:
        # Collect results
        made_progress = False
        for gid, group in worker.get_completed():
            completed_groups[gid] = group
            made_progress = True

        if made_progress:
            last_progress = time.time()

        # Process completed groups
        for gid in list(completed_groups):
            if len(data) >= target_size:
                break
            group = completed_groups.pop(gid)

            # Return aborted groups to the data buffer for retry
            try:
                from slime.utils.types import Sample as SlimeSample
                any_aborted = any(
                    s.status == SlimeSample.Status.ABORTED for s in group
                )
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    data_buffer.add_samples([group])
                    logger.info("Returned aborted group %d to data buffer", gid)
                except Exception as e:
                    logger.error("Failed to return group %d: %s", gid, e)
                continue

            data.append(group)

        # Progress watchdog
        if time.time() - last_progress > no_progress_timeout:
            logger.warning(
                "No progress for %.0fs. Queue=%d, collected=%d/%d",
                no_progress_timeout, worker.queue_size, len(data), target_size,
            )
            last_progress = time.time()

        if not made_progress:
            await asyncio.sleep(0.01)

    logger.info("Rollout %d completed: %d groups", rollout_id, len(data))

    # Sort by original index for deterministic training
    data.sort(key=lambda group: group[0].index if group else 0)
    return data


def generate_rollout_fully_async(
    args: Namespace,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
) -> list[list[Any]]:
    """Slime rollout function entry point.

    Matches the signature expected by Slime's ``RolloutManager``::

        def rollout_fn(args, rollout_id, data_buffer, evaluation=False)
            -> Union[RolloutFnTrainOutput, list[list[Sample]]]

    Configure via environment variables:

    - ``AWE_AGENT_TASK_CLASS``: Task class name (``ScaleSWETask`` / ``BeyondSWETask``).
    - ``AWE_AGENT_DATA_FILE``: Path to JSONL data file.
    - ``AWE_AGENT_RUNTIME_BACKEND``: Runtime backend (``docker`` / ``portal``).
    - ``AWE_AGENT_RUNTIME_IMAGE``: Default container image.
    - ``AWE_AGENT_RUNTIME_TIMEOUT``: Session TTL in seconds (default 14400).
    - ``AWE_AGENT_ENABLE_SEARCH``: Enable search tools (``true`` / ``false``).
    - ``AWE_AGENT_TOOL_CALL_FORMAT``: Tool call format (``codeact_xml``).
    - ``AWE_AGENT_MAX_ITERATIONS``: Max agent steps per instance.
    - ``AWE_AGENT_MAX_POD_RETRY``: Max retries for runtime failures.

    Portal-specific (when ``AWE_AGENT_RUNTIME_BACKEND=portal``):

    - ``AWE_AGENT_PORTAL_REGION``: Portal region.
    - ``AWE_AGENT_SANDBOX_PSM``: Sandbox PSM identifier.
    - ``AWE_AGENT_SANDBOX_ID``: Sandbox ID.
    - ``AWE_AGENT_ZTI_TOKEN``: ZTI authentication token.
    - ``AWE_AGENT_PORTAL_WAIT_TIMEOUT``: Sandbox readiness timeout (default 120s).
    """
    if evaluation:
        raise ValueError("Evaluation mode not yet supported in AweAgent async rollout")

    from slime.utils.async_utils import run
    completed = run(_generate_rollout_async(args, rollout_id, data_buffer))

    # Save rollout data
    _save_rollout_data(args, rollout_id, completed, evaluation)
    return completed


# ── Persistence ───────────────────────────────────────────────────────


def _save_rollout_data(
    args: Namespace,
    rollout_id: int,
    data: list[list[Any]],
    evaluation: bool,
) -> None:
    """Save rollout data and info to disk."""
    save_path = osp.join(args.save, "eval_data" if evaluation else "rollout_data")
    os.makedirs(save_path, exist_ok=True)

    flat = [
        {k: v for k, v in s.to_dict().items() if k != "spec_info"}
        for group in data
        for s in group
    ]

    if evaluation:
        with open(osp.join(save_path, f"eval_rollout_{rollout_id}.jsonl"), "a") as f:
            for item in flat:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        info = {
            "rollout_id": rollout_id,
            "reward": [s.get("reward") for s in flat],
            "truncated": [
                1 if s.get("status") == "truncated" else 0 for s in flat
            ],
        }
        info_path = osp.join(save_path, "rollout_info.jsonl")
        with open(info_path, "a") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        data_path = osp.join(save_path, f"rollout_{rollout_id}.json")
        with open(data_path, "a") as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)


# ── Cleanup ───────────────────────────────────────────────────────────

import atexit  # noqa: E402

atexit.register(_stop_global_worker)
