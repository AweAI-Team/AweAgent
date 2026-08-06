"""User prompt template for the CalibForge-Eval scaffold."""

# ruff: noqa: E501

USER_PROMPT_TEMPLATE = """Task:
{instruction}

Working directory: {workdir}

Complete the task in the current runtime environment. The final state of this environment will be evaluated after you call finish."""
