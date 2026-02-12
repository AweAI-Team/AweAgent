"""LLM abstraction layer.

Provides a config-driven, pluggable LLM client with middleware support.

Usage:
    from awe_agent.core.llm import LLMClient, LLMConfig, Message

    config = LLMConfig(backend="openai", model="gpt-4o")
    async with LLMClient(config) as client:
        response = await client.chat([Message(role="user", content="Hello")])
"""

from awe_agent.core.llm.client import LLMClient, llm_registry
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.protocol import LLMBackend
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

__all__ = [
    "LLMBackend",
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "TokenUsage",
    "ToolCall",
    "llm_registry",
]
