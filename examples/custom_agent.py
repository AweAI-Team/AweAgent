"""Example: Creating a custom agent.

Shows how to implement the Agent protocol to build your own agent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from awe_agent.core.agent import Action, Agent, AgentContext, AgentLoop
from awe_agent.core.llm import LLMClient, LLMConfig, Message
from awe_agent.core.runtime import RuntimeConfig
from awe_agent.core.runtime.docker import DockerRuntime
from awe_agent.core.tool.builtin.bash import BashTool
from awe_agent.core.tool.protocol import Tool


class MySimpleAgent(Agent):
    """A simple custom agent that uses only bash."""

    def get_system_prompt(self, task_info: dict[str, Any]) -> str:
        return (
            "You are a helpful coding assistant. Use the bash tool to "
            "accomplish tasks. Be concise and efficient."
        )

    def get_tools(self) -> list[Tool]:
        return [BashTool(timeout=60)]

    async def step(self, context: AgentContext) -> Action:
        response = await context.llm.chat(
            messages=context.messages,
            tools=context.get_tool_schemas(),
        )

        if response.tool_calls:
            return Action(
                type="tool_call",
                content=response.content,
                tool_calls=[tc.to_dict() for tc in response.tool_calls],
            )

        return Action(type="finish", content=response.content)


async def main() -> None:
    llm_config = LLMConfig(backend="openai", model="gpt-4o")
    runtime_config = RuntimeConfig(backend="docker", image="python:3.11-slim")

    agent = MySimpleAgent()
    runtime = DockerRuntime(runtime_config)

    async with runtime.session() as session:
        llm = LLMClient(llm_config)
        context = AgentContext(
            llm=llm, session=session, tools=agent.get_tools(), max_steps=10
        )
        loop = AgentLoop(agent, context)
        result = await loop.run("What Python version is installed? Show the output.")
        print(f"Result: {result.finish_reason}")
        for step in result.trajectory.steps:
            print(f"  Step {step.step}: {step.action.type}")


if __name__ == "__main__":
    asyncio.run(main())
