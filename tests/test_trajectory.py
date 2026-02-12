"""Tests for trajectory recording and export."""

from __future__ import annotations

from awe_agent.core.agent.trajectory import Action, Trajectory, TrajectoryStep


def test_action_creation():
    action = Action(
        type="tool_call",
        content="Let me fix this.",
        tool_calls=[{"name": "bash", "arguments": '{"command": "ls"}'}],
    )
    assert action.type == "tool_call"
    assert len(action.tool_calls) == 1


def test_trajectory_add_step():
    traj = Trajectory()
    action = Action(type="tool_call", content="step 1")
    traj.add_step(step=0, action=action)
    assert len(traj.steps) == 1
    assert traj.steps[0].step == 0
    assert traj.steps[0].action.content == "step 1"


def test_trajectory_multiple_steps():
    traj = Trajectory()
    for i in range(5):
        traj.add_step(step=i, action=Action(type="tool_call", content=f"step {i}"))
    assert len(traj.steps) == 5


def test_trajectory_with_observations():
    traj = Trajectory()
    action = Action(type="tool_call", content="run test")
    traj.add_step(step=0, action=action)
    traj.steps[0].observations = ["test output here"]
    assert traj.steps[0].observations == ["test output here"]


def test_trajectory_training_format():
    traj = Trajectory()
    action = Action(
        type="tool_call",
        content="fix",
        token_ids=[1, 2, 3],
        logprobs=[-0.1, -0.2, -0.3],
    )
    traj.add_step(step=0, action=action)
    traj.final_reward = 1.0

    data = traj.to_training_format()
    assert data["reward"] == 1.0
    assert data["response_token_ids"] == [1, 2, 3]
    assert data["logprobs"] == [-0.1, -0.2, -0.3]
    assert data["num_steps"] == 1


def test_trajectory_training_format_no_tokens():
    traj = Trajectory()
    traj.add_step(step=0, action=Action(type="finish", content="done"))
    traj.final_reward = 0.0

    data = traj.to_training_format()
    assert data["reward"] == 0.0
    assert data["response_token_ids"] == []
