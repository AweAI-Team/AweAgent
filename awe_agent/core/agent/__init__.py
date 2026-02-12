"""Agent abstraction layer.

Provides the Agent protocol, execution loop, context, and trajectory types.

Usage:
    from awe_agent.core.agent import Agent, AgentLoop, AgentContext, AgentResult
"""

from awe_agent.core.agent.context import AgentContext, BashConstraints
from awe_agent.core.agent.loop import AgentLoop, AgentResult
from awe_agent.core.agent.protocol import Agent
from awe_agent.core.agent.trajectory import Action, Trajectory, TrajectoryStep

__all__ = [
    "Action",
    "Agent",
    "AgentContext",
    "AgentLoop",
    "AgentResult",
    "BashConstraints",
    "Trajectory",
    "TrajectoryStep",
]
