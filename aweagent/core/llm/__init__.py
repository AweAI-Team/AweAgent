"""LLM abstraction layer.

Provides a config-driven, pluggable LLM client with middleware support.

Usage:
    from aweagent.core.llm import LLMClient, LLMConfig, Message

    config = LLMConfig(backend="openai", model="gpt-4o")
    async with LLMClient(config) as client:
        response = await client.chat([Message(role="user", content="Hello")])
"""

from aweagent.core.llm.client import LLMClient, create_async_client, llm_registry
from aweagent.core.llm.config import LLMConfig, ReasoningConfig
from aweagent.core.llm.protocol import LLMBackend
from aweagent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

__all__ = [
    "LLMBackend",
    "LLMClient",
    "create_async_client",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "ReasoningConfig",
    "TokenUsage",
    "ToolCall",
    "llm_registry",
]
