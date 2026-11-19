"""Follow-up instruction shared by search steps and final-answer recovery."""

NO_TOOL_CALL_PROMPT = (
    "CRITICAL: Continue by calling one of the available tools. "
    "Use search/read tools to keep investigating, or call finish(answer=...) "
    "if you are ready to submit the final answer."
)
