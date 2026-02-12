"""Slime RL framework integration.

Provides a clean rollout interface for RL training. Replaces the complex
global-state + dual-event-loop design in swalm_fully_async_rollout.py.

Key design:
- No global state — instance-based lifecycle via context manager
- Single event loop — all async in one loop
- LLM requests automatically routed to SGLang via config
- Trajectory data automatically collected for training

Usage:
    rollout = AweAgentRollout(
        agent_factory=lambda: SearchSWEAgent(enable_search=True),
        task=SWEBenchTask("scaleswe"),
        runtime_config=RuntimeConfig(backend="portal"),
        max_concurrent=50,
    )

    # Called by Slime each training iteration
    training_data = await rollout.generate(samples, sglang_url="http://...")
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.loop import AgentLoop
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.runtime.config import RuntimeConfig
from awe_agent.core.task.protocol import Evaluator, Task
from awe_agent.core.task.runner import runtime_registry

logger = logging.getLogger(__name__)


class AweAgentRollout:
    """Slime-compatible rollout for RL training.

    Each call to generate() runs the agent on a batch of samples,
    collecting trajectory data (token_ids, logprobs, rewards) for training.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Agent],
        task: Task,
        runtime_config: RuntimeConfig,
        evaluator: Evaluator | None = None,
        eval_runtime_config: RuntimeConfig | None = None,
        max_concurrent: int = 50,
        agent_max_steps: int = 100,
        positive_reward: float = 1.0,
        negative_reward: float = 0.0,
    ) -> None:
        self.agent_factory = agent_factory
        self.task = task
        self.runtime_config = runtime_config
        self.evaluator = evaluator
        self.eval_runtime_config = eval_runtime_config
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._agent_max_steps = agent_max_steps
        self._pos_reward = positive_reward
        self._neg_reward = negative_reward

    async def generate(
        self,
        samples: list[Any],
        sglang_url: str,
        llm_overrides: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run agent rollout for a batch of samples.

        Args:
            samples: Slime Sample objects. Each must have .metadata with
                     "instance_id" and "dataset_id".
            sglang_url: SGLang router URL provided by Slime.
            llm_overrides: Optional LLM param overrides.

        Returns:
            List of training data dicts compatible with Slime.
        """
        # Build LLM config pointing to SGLang
        llm_config = LLMConfig(
            backend="sglang",
            base_url=sglang_url,
            model="SGLANG_ENGINE",
            return_tokens=True,
            return_logprobs=True,
            params={
                "temperature": 0.7,
                "max_tokens": 4096,
                **(llm_overrides or {}),
            },
        )

        tasks = [
            self._process_sample(sample, llm_config)
            for sample in samples
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        training_data = []
        for sample, result in zip(samples, results):
            if isinstance(result, Exception):
                logger.error("Sample %s failed: %s", getattr(sample, "id", "?"), result)
                training_data.append(self._make_failed_result(sample))
            else:
                training_data.append(result)

        return training_data

    async def _process_sample(
        self,
        sample: Any,
        llm_config: LLMConfig,
    ) -> dict[str, Any]:
        """Process a single sample: run agent + evaluate + collect data."""
        async with self._semaphore:
            start_time = time.monotonic()
            metadata = getattr(sample, "metadata", {})
            instance_id = metadata.get("instance_id", "")

            # Get task instance
            instances = self.task.get_instances([instance_id])
            if not instances:
                raise ValueError(f"Instance not found: {instance_id}")
            instance = instances[0]

            # Create runtime session
            runtime_cls = runtime_registry.get(self.runtime_config.backend)
            runtime = runtime_cls(self.runtime_config)
            image = self.task.get_image(instance)

            async with runtime.session(image) as session:
                # Setup
                for cmd in self.task.get_setup_commands(instance):
                    await session.execute(cmd)

                # Run agent
                agent = self.agent_factory()
                llm = LLMClient(llm_config)
                context = AgentContext(
                    llm=llm,
                    session=session,
                    tools=agent.get_tools(),
                    task_info=self.task.get_task_info(instance),
                    max_steps=self._agent_max_steps,
                )
                loop = AgentLoop(agent, context)
                agent_result = await loop.run(self.task.get_prompt(instance))

            # Evaluate
            reward = self._neg_reward
            if self.evaluator and agent_result.patch:
                eval_config = self.eval_runtime_config or self.runtime_config
                eval_runtime_cls = runtime_registry.get(eval_config.backend)
                eval_runtime = eval_runtime_cls(eval_config)
                eval_result = await self.evaluator.evaluate(
                    instance, agent_result.patch, eval_runtime
                )
                if eval_result.accepted:
                    reward = self._pos_reward

            # Build training data
            trajectory_data = agent_result.trajectory.to_training_format()
            trajectory_data["reward"] = reward
            trajectory_data["instance_id"] = instance_id
            trajectory_data["patch"] = agent_result.patch
            trajectory_data["duration"] = time.monotonic() - start_time
            trajectory_data["finish_reason"] = agent_result.finish_reason

            return trajectory_data

    def _make_failed_result(self, sample: Any) -> dict[str, Any]:
        """Create a failed result placeholder."""
        return {
            "response_token_ids": [],
            "logprobs": [],
            "reward": self._neg_reward,
            "instance_id": getattr(sample, "metadata", {}).get("instance_id", ""),
            "error": True,
        }
