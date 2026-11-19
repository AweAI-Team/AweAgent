"""Inference-time folding and tools for the FoldSearch scaffold."""

from __future__ import annotations

import copy
import json
import uuid
from functools import lru_cache
from typing import Any

from aweagent.core.llm.types import Message, ToolCall
from aweagent.core.tool.protocol import Tool
from aweagent.core.tool.search import (
    SearchConstraints,
    WebFetchRawTool,
    WebFetchTool,
    WebSearchTool,
)

FOLD_TEXT = "Content folded due to space limitation"

@lru_cache(maxsize=8)
def _load_tokenizer(tokenizer_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "agent.tool_options.fold_search.tokenizer_path requires transformers"
        ) from exc
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)


def _as_positive_int(value: Any, *, default: int, field_name: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _build_tools(
    tool_options: dict[str, dict[str, Any]],
    search_constraints: SearchConstraints | None,
) -> list[Tool]:
    search_options = tool_options.get("web_search", {})
    fetch_options = tool_options.get("web_fetch", {})

    reader = None
    reader_backend = fetch_options.get("reader_backend")
    if reader_backend:
        reader = WebFetchRawTool(
            constraints=search_constraints,
            max_content_tokens=fetch_options.get("max_content_tokens", 100000),
            max_attempts=fetch_options.get("reader_max_attempts", 3),
            backend=reader_backend,
        )

    return [
        WebSearchTool(
            engine=search_options.get("engine"),
            constraints=search_constraints,
            max_attempts=search_options.get("max_attempts", 1),
            result_scheme=search_options.get("result_scheme"),
            backend=search_options.get("backend"),
        ),
        WebFetchTool(
            constraints=search_constraints,
            max_content_tokens=fetch_options.get("max_content_tokens", 25000),
            max_attempts=fetch_options.get("max_attempts", 3),
            system_prompt=fetch_options.get("system_prompt"),
            llm_config_path=fetch_options.get("llm_config_path"),
            llm_config=fetch_options.get("llm") or fetch_options.get("llm_config"),
            llm_params=fetch_options.get("llm_params"),
            reader=reader,
        ),
    ]


def _message_to_dict(message: Message) -> dict[str, Any]:
    d = message.to_dict()
    if message.reasoning_raw is not None:
        d["reasoning_content"] = message.reasoning_raw
    return d


def _dict_to_message(message: dict[str, Any]) -> Message:
    return Message.from_dict({
        "role": message["role"],
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
        "tool_call_id": message.get("tool_call_id"),
        "name": message.get("name"),
        "reasoning_raw": message.get("reasoning_content"),
    })


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text))


class RuntimeTokenContextRewriter:
    """Fold older tool observations using the fold_search.py token policy."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_context_length: int,
        target_context_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.target_context_length = target_context_length
        self.fold_text = FOLD_TEXT
        self.mask_token_length = _token_count(tokenizer, self.fold_text)

    def process(self, messages: list[Message]) -> list[Message]:
        message_dicts = [_message_to_dict(message) for message in messages]
        total_tool_tokens = 0
        msg_lengths: dict[int, int] = {}
        tool_indices: list[int] = []

        for index, message in enumerate(message_dicts):
            if message.get("role") != "tool":
                continue
            length = _token_count(self.tokenizer, str(message.get("content") or ""))
            msg_lengths[index] = length
            total_tool_tokens += length
            tool_indices.append(index)

        if total_tool_tokens <= self.max_context_length:
            return [_dict_to_message(message) for message in copy.deepcopy(message_dicts)]

        processed = copy.deepcopy(message_dicts)
        current_tool_tokens = total_tool_tokens
        masked_count = 0
        total_tools = len(tool_indices)

        for index in tool_indices:
            remaining_tools = total_tools - masked_count
            if remaining_tools <= 2:
                break
            if processed[index].get("content") == self.fold_text:
                masked_count += 1
                continue
            saved = msg_lengths[index] - self.mask_token_length
            if saved > 0:
                processed[index]["content"] = self.fold_text
                current_tool_tokens -= saved
                masked_count += 1
            if current_tool_tokens <= self.target_context_length:
                break

        return [_dict_to_message(message) for message in processed]


def count_messages_tokens(tokenizer: Any, messages: list[Message]) -> int:
    total = 0
    for message in messages:
        total += _token_count(tokenizer, str(message.content or ""))
    return total


def _tool_call_dict(tool_call: ToolCall) -> dict[str, Any]:
    d = tool_call.to_dict()
    if not d.get("id"):
        d["id"] = f"call_{uuid.uuid4().hex}"
    return d


def _parse_arguments(tool_name: str, raw_args: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    try:
        loaded = json.loads(raw_args)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        if tool_name == "code_interpreter":
            return {"code": raw_args}
        if tool_name == "web_search":
            return {"queries": [raw_args]}
        return {}
