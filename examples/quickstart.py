"""Quickstart example — run an agent on a single task.

Usage:
    AWE_AGENT__LLM__API_KEY=sk-xxx python examples/quickstart.py
"""

import asyncio

from awe_agent.core.agent import AgentContext, AgentLoop
from awe_agent.core.llm import LLMClient, LLMConfig, Message
from awe_agent.core.runtime import RuntimeConfig
from awe_agent.core.runtime.docker import DockerRuntime
from awe_agent.scaffold.search_swe import SearchSWEAgent


async def main() -> None:
    # Configure LLM
    llm_config = LLMConfig(
        backend="openai",
        model="gpt-4o",
        params={"temperature": 0.0, "max_tokens": 4096},
    )

    # Configure runtime
    runtime_config = RuntimeConfig(
        backend="docker",
        image="python:3.11-slim",
        workdir="/workspace",
    )

    # Create components
    agent = SearchSWEAgent()
    runtime = DockerRuntime(runtime_config)

    async with runtime.session() as session:
        llm = LLMClient(llm_config)
        context = AgentContext(
            llm=llm,
            session=session,
            tools=agent.get_tools(),
            max_steps=20,
        )
        loop = AgentLoop(agent, context)

        # Run the agent
        result = await loop.run(
            "Create a Python file called fibonacci.py that prints the first 20 Fibonacci numbers. "
            "Then run it to verify it works."
        )

        print(f"Finish reason: {result.finish_reason}")
        print(f"Steps taken: {len(result.trajectory.steps)}")
        print(f"Patch:\n{result.patch}")


if __name__ == "__main__":
    asyncio.run(main())
