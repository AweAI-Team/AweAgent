"""Example: RL training integration with Slime.

Shows how AweAgent integrates with Slime for RL training.
The only difference from inference is the LLM config pointing to SGLang.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from awe_agent.core.runtime import RuntimeConfig
from awe_agent.integrations.slime import AweAgentRollout
from awe_agent.scaffold.search_swe import SearchSWEAgent


# Minimal mock of Slime's Sample type
@dataclass
class MockSample:
    prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# Minimal mock Task for demonstration
class MockTask:
    def get_instances(self, instance_ids=None):
        from awe_agent.core.task.types import Instance
        return [Instance(
            id="test-001",
            dataset_id="scaleswe",
            workdir="/testbed",
            image="python:3.11-slim",
            problem_statement="Fix the bug in utils.py",
        )]

    def get_prompt(self, instance):
        return instance.problem_statement

    def get_image(self, instance):
        return instance.image

    def get_setup_commands(self, instance):
        return []

    def get_task_info(self, instance):
        return {"instance_id": instance.id, "workdir": instance.workdir}


async def main() -> None:
    """Demonstrate RL rollout (won't actually work without SGLang + Docker)."""
    # Create rollout — this is what Slime calls
    rollout = AweAgentRollout(
        agent_factory=lambda: SearchSWEAgent(),
        task=MockTask(),
        runtime_config=RuntimeConfig(backend="docker", image="python:3.11-slim"),
        max_concurrent=50,
        agent_max_steps=100,
    )

    # These samples would come from Slime's data source
    samples = [
        MockSample(metadata={"instance_id": "test-001", "dataset_id": "scaleswe"}),
    ]

    # In real training, sglang_url is provided by Slime's training loop
    sglang_url = "http://localhost:30000"

    print("AweAgent Rollout configured:")
    print(f"  Agent: SearchSWEAgent")
    print(f"  Task: MockTask")
    print(f"  SGLang URL: {sglang_url}")
    print(f"  Max concurrent: 50")
    print()
    print("In real training, call:")
    print("  training_data = await rollout.generate(samples, sglang_url)")
    print()
    print("The training_data contains:")
    print("  - response_token_ids: all generated tokens")
    print("  - logprobs: log probabilities for each token")
    print("  - reward: 1.0 (pass) or 0.0 (fail)")
    print("  - patch: the code changes made")


if __name__ == "__main__":
    asyncio.run(main())
