"""Agent abstraction layer.

Provides the Agent protocol, execution loop, context, and trajectory types.

Usage:
    from aweagent.core.agent import Agent, AgentLoop, AgentContext, AgentResult
"""

from aweagent.core.agent.context import AgentContext, BashConstraints
from aweagent.core.agent.loop import AgentLoop, AgentResult
from aweagent.core.agent.protocol import Agent
from aweagent.core.agent.stats import RunStats
from aweagent.core.agent.training import TrainingState
from aweagent.core.agent.trajectory import Action, Trajectory, TrajectoryStep

__all__ = [
    "Action",
    "Agent",
    "AgentContext",
    "AgentLoop",
    "AgentResult",
    "BashConstraints",
    "RunStats",
    "Trajectory",
    "TrajectoryStep",
    "TrainingState",
]
