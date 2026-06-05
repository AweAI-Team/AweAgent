"""Example: RL training shares the same agent code as inference.

The core idea: the *only* thing that turns an inference run into an RL rollout
is a single optional field on :class:`AgentContext` — ``training``.  When it is
``None`` (the default) the agent runs as pure inference at zero cost.  When it
holds a :class:`TrainingState`, the *same* :class:`AgentLoop` accumulates
token-level RL data (prompt/response token ids, a per-token ``loss_mask``, and
rollout logprobs) as a side effect of normal execution — no fork of the agent,
the loop, or the tools.

This file shows two layers:

1. ``demo_training_seam()`` — the framework-level seam: build an
   ``AgentContext`` with a ``TrainingState`` and read back ``to_rl_data()``.
   This is what makes "develop == deploy == train" hold.
2. ``production_entry_point()`` — how the Slime post-training framework drives
   real rollouts via :func:`awe_agent.integrations.slime.generate_rollout_fully_async`.

Running an actual rollout needs an SGLang server (for token ids + logprobs), a
container runtime (Docker), and a Slime checkout, so ``main()`` only prints the
wiring rather than executing it.
"""

from __future__ import annotations

import asyncio

from awe_agent.core.agent.context import AgentContext
from awe_agent.core.agent.training import TrainingState
from awe_agent.core.llm.client import LLMClient
from awe_agent.core.llm.config import LLMConfig
from awe_agent.scaffold.search_swe.agent import SearchSWEAgent


def build_training_context(tokenizer, session, max_new_tokens: int = 32768) -> AgentContext:
    """Build an ``AgentContext`` wired for RL training.

    The LLM points at an SGLang engine (``backend="sglang"``) so the loop
    receives raw token ids and logprobs; ``return_tokens`` / ``return_logprobs``
    enable the token-level RL data path.  Attaching a ``TrainingState`` is the
    single switch that turns this into a rollout — everything else (agent,
    loop, tools) is identical to inference.
    """
    agent = SearchSWEAgent(enable_search=False, tool_call_format="codeact_xml")

    llm = LLMClient(LLMConfig(
        backend="sglang",
        base_url="http://localhost:30000",
        model="SGLANG_ENGINE",
        return_tokens=True,
        return_logprobs=True,
    ))

    return AgentContext(
        llm=llm,
        session=session,                 # a RuntimeSession (e.g. DockerRuntime)
        tools=agent.get_tools(),
        max_steps=100,
        # The one and only difference from inference:
        training=TrainingState(tokenizer=tokenizer, max_new_tokens=max_new_tokens),
    )


async def demo_training_seam() -> None:
    """Sketch of the framework-level seam (does not execute — needs SGLang + Docker).

    The shape of the code that runs a training rollout is exactly the inference
    code, plus reading ``ctx.training.to_rl_data()`` at the end::

        from awe_agent.core.agent.loop import AgentLoop

        ctx = build_training_context(tokenizer, session)
        loop = AgentLoop(agent, ctx)
        result = await loop.run(task_prompt)        # SAME loop as inference

        rl_data = ctx.training.to_rl_data()
        # rl_data maps 1:1 onto a Slime Sample:
        #   sample.tokens            = rl_data["prompt_token_ids"] + rl_data["response_token_ids"]
        #   sample.response_length   = len(rl_data["response_token_ids"])
        #   sample.loss_mask         = rl_data["loss_mask"]        # model tokens=1, env tokens=0
        #   sample.rollout_log_probs = rl_data["rollout_log_probs"]
        #   sample.weight_versions   = rl_data["weight_versions"]

    Advantage estimation and the gradient step live in Slime, not here.
    """


def production_entry_point() -> None:
    """How Slime drives real rollouts.

    ``awe_agent.integrations.slime`` exposes the rollout function Slime's
    ``RolloutManager`` calls each iteration::

        from awe_agent.integrations.slime import generate_rollout_fully_async
        # def rollout_fn(args, rollout_id, data_buffer, evaluation=False) -> list[list[Sample]]

    It is configured entirely through environment variables (no code changes):

        AWE_AGENT_TASK_CLASS        ScaleSWETask | BeyondSWETask
        AWE_AGENT_DATA_FILE         path to the JSONL dataset
        AWE_AGENT_RUNTIME_BACKEND   docker | portal
        AWE_AGENT_RUNTIME_IMAGE     default container image
        AWE_AGENT_ENABLE_SEARCH     true | false
        AWE_AGENT_TOOL_CALL_FORMAT  codeact_xml
        AWE_AGENT_MAX_ITERATIONS    max agent steps per instance

    Under the hood it reuses the *same* pipeline as evaluation —
    ``runtime.session`` -> ``PreAgentSetup`` -> ``AgentLoop.run`` -> ``Evaluator``
    — and fills each Slime ``Sample`` with the token-level data above plus a
    reward from the evaluator.
    """


async def main() -> None:
    # Importing the Slime bridge does not require Slime itself (it is imported
    # lazily at call time), so this stays safe to run anywhere.
    from awe_agent.integrations import slime

    print("AweAgent RL training: same agent, one switch.\n")
    print("Inference vs training differ by a single AgentContext field:")
    print("  inference : AgentContext(..., training=None)        # zero-cost")
    print("  training  : AgentContext(..., training=TrainingState(tokenizer=...))")
    print()
    print("The same AgentLoop accumulates, per step:")
    print("  - model tokens        -> loss_mask = 1")
    print("  - environment tokens  -> loss_mask = 0")
    print("  - logprobs + weight_version (SGLang /generate)")
    print()
    print("Production entry point (called by Slime's RolloutManager):")
    print(f"  awe_agent.integrations.slime.{slime.generate_rollout_fully_async.__name__}"
          "(args, rollout_id, data_buffer)")
    print()
    print("End-to-end training additionally requires: an SGLang server, a Docker")
    print("runtime, and a Slime checkout. See awe_agent/integrations/slime.py.")


if __name__ == "__main__":
    asyncio.run(main())
