"""OpenAI Response API backend.

Uses the ``/v1/responses`` endpoint (newer API) which supports
input items, function_call output items, and reasoning items.

Requires openai>=1.66.0 (pinned in pyproject.toml).
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message, TokenUsage, ToolCall

logger = logging.getLogger(__name__)


def _normalize_response_status(
    status: str | None,
    incomplete_details: Any = None,
) -> str | None:
    """Map Response API status to framework-standard finish_reason.

    For ``incomplete`` responses, checks ``incomplete_details.reason`` to
    distinguish token truncation from other causes (e.g. content_filter).
    Only token-related reasons are mapped to ``"length"``; others are
    preserved as-is so the agent loop doesn't misinterpret them as
    truncation.
    """
    if status is None:
        return None
    if status == "completed":
        return "stop"
    if status == "failed":
        return "error"
    if status == "incomplete":
        reason = None
        if incomplete_details is not None:
            reason = getattr(incomplete_details, "reason", None)
        # Only map to "length" when the reason is token-related
        if reason in ("max_output_tokens", "max_tokens"):
            return "length"
        # Return the specific reason if available, otherwise "incomplete"
        return reason or "incomplete"
    return status


class OpenAIResponseBackend:
    """Backend for the OpenAI Response API (/v1/responses)."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = self._create_client(config)

    @staticmethod
    def _create_client(config: LLMConfig) -> AsyncOpenAI:
        """Create the appropriate async client.

        Uses AsyncAzureOpenAI when ``extra.api_version`` is set (Azure-
        compatible endpoints), otherwise plain
        AsyncOpenAI.
        """
        api_version = config.extra.get("api_version")
        if api_version:
            return AsyncAzureOpenAI(
                api_key=config.api_key or "dummy",
                azure_endpoint=config.base_url or "",
                api_version=api_version,
                timeout=config.timeout,
            )
        return AsyncOpenAI(
            api_key=config.api_key or "dummy",
            base_url=config.base_url,
            timeout=config.timeout,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        instructions, input_items, prev_response_id = self._serialize_input(messages)

        params: dict[str, Any] = {
            "model": kwargs.pop("model", self.config.model),
            "input": input_items,
        }

        if instructions:
            params["instructions"] = instructions

        # Use previous_response_id for efficient multi-turn continuation
        if prev_response_id:
            params["previous_response_id"] = prev_response_id

        # Merge config params
        for k, v in self.config.params.items():
            if k not in params:
                params[k] = v
        params.update(kwargs)

        # Reasoning effort
        reasoning_cfg = self.config.reasoning
        if reasoning_cfg.effort and "reasoning" not in params:
            params["reasoning"] = {"effort": reasoning_cfg.effort}
            if reasoning_cfg.summary:
                params["reasoning"]["summary"] = reasoning_cfg.summary

        # Tool schemas
        if tools:
            params["tools"] = self._convert_tools(tools)

        # max_tokens / max_completion_tokens → max_output_tokens (Response API naming)
        if "max_tokens" in params and "max_output_tokens" not in params:
            params["max_output_tokens"] = params.pop("max_tokens")
        if "max_completion_tokens" in params and "max_output_tokens" not in params:
            params["max_output_tokens"] = params.pop("max_completion_tokens")

        # Remove keys not accepted by Response API
        params.pop("stop", None)
        params.pop("response_format", None)
        params.pop("thinking", None)  # Response API uses reasoning, not thinking

        response = await self._client.responses.create(**params)
        return self._parse_response(response)

    def _serialize_input(
        self, messages: list[Message]
    ) -> tuple[str, list[dict[str, Any]], str | None]:
        """Convert framework Messages to Response API input items.

        Two mutually exclusive strategies:

        1. **previous_response_id** (incremental): The API already holds
           conversation history.  Only items *after* the last assistant
           turn that carried a response_id are sent as new input.

        2. **Manual replay** (no response_id): All items — including
           reasoning items from assistant turns — are sent so the API
           has full context.

        Returns:
            (instructions, input_items, previous_response_id)
        """
        instructions_parts: list[str] = []
        items: list[dict[str, Any]] = []
        last_response_id: str | None = None
        # Track where items from the last response-id turn end,
        # so we can trim to incremental input when using continuation.
        last_response_item_end: int = 0

        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    instructions_parts.append(m.content)
                continue

            if m.role == "user":
                items.append({
                    "type": "message",
                    "role": "user",
                    "content": m.content or "",
                })

            elif m.role == "assistant":
                # Extract response_id from reasoning_raw
                has_response_id = False
                if (
                    m.reasoning_raw
                    and isinstance(m.reasoning_raw, dict)
                ):
                    if m.reasoning_raw.get("response_id"):
                        last_response_id = m.reasoning_raw["response_id"]
                        has_response_id = True

                    # Replay reasoning items (used only in manual mode;
                    # trimmed away if previous_response_id is found later).
                    for ri in m.reasoning_raw.get("reasoning_items") or []:
                        items.append(ri)

                # Text content as assistant message
                if m.content:
                    items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": m.content,
                    })
                # Tool calls as function_call items
                if m.tool_calls:
                    for tc in m.tool_calls:
                        items.append({
                            "type": "function_call",
                            "call_id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        })

                if has_response_id:
                    # Mark: everything up to here is already known to the API
                    last_response_item_end = len(items)

            elif m.role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id or "",
                    "output": m.content or "",
                })

        # Strategy selection:
        # - If we have a previous_response_id, the API already holds all
        #   items up to last_response_item_end.  Only send new input.
        # - Otherwise, send everything (manual replay mode).
        # Continuation can be disabled via extra.use_continuation=false
        # for endpoints that don't persist server-side response state.
        use_continuation = self.config.extra.get("use_continuation", True)
        if not use_continuation:
            last_response_id = None
        elif last_response_id is not None:
            items = items[last_response_item_end:]

        instructions = "\n\n".join(instructions_parts) if instructions_parts else ""
        return instructions, items, last_response_id

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse Response API output items."""
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        thinking_parts: list[str] = []
        reasoning_items: list[dict[str, Any]] = []  # Full reasoning for training

        for item in response.output:
            if item.type == "message":
                # Message output item contains content parts
                for part in getattr(item, "content", []) or []:
                    if hasattr(part, "type"):
                        if part.type == "output_text":
                            content_parts.append(part.text)
                        elif part.type == "text":
                            content_parts.append(part.text)
            elif item.type == "function_call":
                tool_calls.append(ToolCall(
                    id=getattr(item, "call_id", "") or getattr(item, "id", ""),
                    name=item.name,
                    arguments=item.arguments,
                ))
            elif item.type == "reasoning":
                # Store full reasoning item for training/trajectory.
                # Capture all available attributes — the API may expose
                # content, encrypted_content, or other fields in future.
                reasoning_entry: dict[str, Any] = {
                    "type": "reasoning",
                    "id": getattr(item, "id", None),
                }
                # Preserve encrypted_content if present (needed for
                # round-trip when manually managing conversation state)
                encrypted_content = getattr(item, "encrypted_content", None)
                if encrypted_content is not None:
                    reasoning_entry["encrypted_content"] = encrypted_content
                content_field = getattr(item, "content", None)
                if content_field is not None:
                    reasoning_entry["content"] = content_field
                summary = getattr(item, "summary", None)
                if summary:
                    summary_texts = []
                    for s in summary:
                        if hasattr(s, "text"):
                            summary_texts.append(s.text)
                            thinking_parts.append(s.text)
                    reasoning_entry["summary"] = summary_texts
                reasoning_items.append(reasoning_entry)

        content_text = "\n".join(content_parts) if content_parts else None
        thinking_text = "\n".join(thinking_parts) if thinking_parts else None

        # Build reasoning_raw: includes reasoning items + response ID for continuation
        reasoning_raw = None
        if reasoning_items or hasattr(response, "id"):
            reasoning_raw = {
                "response_id": getattr(response, "id", None),
                "reasoning_items": reasoning_items or None,
            }

        # Normalize finish_reason: Response API uses "status" field
        raw_status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        finish_reason = _normalize_response_status(raw_status, incomplete_details)

        # Parse usage
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, "input_tokens", 0)
                or getattr(response.usage, "prompt_tokens", 0),
                completion_tokens=getattr(response.usage, "output_tokens", 0)
                or getattr(response.usage, "completion_tokens", 0),
                total_tokens=getattr(response.usage, "total_tokens", 0),
            )

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            reasoning_text=thinking_text,
            reasoning_raw=reasoning_raw,
            usage=usage,
            finish_reason=finish_reason,
            raw=response,
        )

    def _convert_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI Chat Completions tool schema to Response API format."""
        response_tools: list[dict[str, Any]] = []
        for t in tools:
            func = t.get("function", t)
            response_tools.append({
                "type": "function",
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "strict": self.config.extra.get("strict_tools", False),
            })
        return response_tools

    async def close(self) -> None:
        """Clean up client resources."""
        if hasattr(self._client, "close"):
            await self._client.close()
