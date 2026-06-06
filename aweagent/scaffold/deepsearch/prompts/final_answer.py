"""Forced final-answer extraction prompts for DeepSearch."""

from __future__ import annotations

from dataclasses import dataclass

OMITTED_TOOL_RESULT = "Tool result is omitted to save tokens."
NO_FINAL_ANSWER_FALLBACK = "No final answer generated."


@dataclass(frozen=True)
class FinalAnswerPrompts:
    """Prompt bundle used when DeepSearch must extract an answer without tools."""

    current_history: str
    folded_history: str


DEFAULT_FINAL_ANSWER_PROMPTS = FinalAnswerPrompts(
    current_history="""You are extracting the final answer for a search QA task.

The agent did not submit an answer with finish(answer=...) before the rollout ended.
Based only on the conversation history above, provide the most likely final answer now.
Do not call tools. Do not request more information.

Return a concise final answer. If useful, include brief evidence, but make the exact
answer easy to identify.
""",
    folded_history="""You are extracting the final answer for a search QA task.

The previous extraction attempt did not produce a usable answer. The conversation
history above preserves the agent's replies and tool-call structure, but older tool
results may be folded to save context.

Based only on that history, provide the most likely final answer now. Do not call
tools. Do not request more information. Return a concise final answer.
""",
)

FINAL_ANSWER_PROMPTS: dict[str, FinalAnswerPrompts] = {
    "default": DEFAULT_FINAL_ANSWER_PROMPTS,
}


def get_final_answer_prompts(key: str) -> FinalAnswerPrompts:
    """Get a DeepSearch forced-answer prompt bundle by key."""
    if key not in FINAL_ANSWER_PROMPTS:
        raise KeyError(
            f"Unknown DeepSearch final-answer prompt key: {key!r}. "
            f"Available: {list(FINAL_ANSWER_PROMPTS.keys())}"
        )
    return FINAL_ANSWER_PROMPTS[key]


__all__ = [
    "FINAL_ANSWER_PROMPTS",
    "FinalAnswerPrompts",
    "NO_FINAL_ANSWER_FALLBACK",
    "OMITTED_TOOL_RESULT",
    "get_final_answer_prompts",
]
