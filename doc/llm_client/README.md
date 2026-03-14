# AweAgent LLM Client

A config-driven, multi-backend LLM abstraction layer with full reasoning preservation for training and inference.

## Architecture

```
                        LLMClient
                           |
                     +-----+------+
                     | Middleware  |
                     | (retry,    |
                     |  trace)    |
                     +-----+------+
                           |
          +----------------+----------------+
          |                |                |
     LLMBackend       LLMBackend       LLMBackend
     (protocol)       (protocol)       (protocol)
          |                |                |
    +-----+-----+   +-----+-----+   +-----+-----+
    |  OpenAI   |   | Anthropic |   |   Ark     |   ...
    |  Backend  |   |  Backend  |   |  Backend  |
    +-----------+   +-----------+   +-----------+
```

### Design Principles

- **Config-driven**: All behavior is determined by YAML configuration. Backends never make implicit assumptions or silent fallbacks.
- **Protocol-decoupled**: Every backend implements the `LLMBackend` protocol (`chat()` + `close()`). Adding a new provider requires zero changes to existing code.
- **Semantic layering**: Reasoning data is split into two fields with distinct purposes: `reasoning_text` (human-readable, for logging and training) and `reasoning_raw` (opaque provider payload, for multi-turn round-trip).
- **Full reasoning preservation**: Even when the provider API discards reasoning across turns (e.g., DeepSeek, interleaved thinking), the framework preserves it in `Action.reasoning_raw` and exports it through `Trajectory` for training/inference replay.

## Reasoning Data Flow

```
Provider API response
    |
LLMResponse
    |-- reasoning_text: str      <-- human-readable (logs, training, observation)
    |-- reasoning_raw: Any       <-- opaque provider payload (round-trip)
    |
Action (trajectory.py)
    |-- reasoning_text            <-- persisted permanently
    |-- reasoning_raw             <-- persisted permanently
    |
Message (types.py)
    |-- reasoning_raw             <-- sent back to provider on subsequent turns
    |
Backend serialization
    |-- OpenAI-compatible: controlled by reasoning.preserve and reasoning.format
    |-- Other backends: provider-specific round-trip rules
```

**Training export:**
- `Trajectory.to_messages()` exports both `reasoning_text` and `reasoning_raw`.
- `Trajectory.to_training_format()` collects per-step `reasoning_text` + `reasoning_raw` into a `reasoning` field.

## Supported Backends

| Backend | Source | Protocol | Use Case |
|---------|--------|----------|----------|
| `openai` | `backends/openai.py` | Chat Completions | GPT-series + all OpenAI-compatible models |
| `openai_response` | `backends/openai_response.py` | Responses API | GPT-4.1+, incremental continuation |
| `anthropic` | `backends/anthropic.py` | Messages API | Claude-series + Anthropic-compatible models |
| `ark` | `backends/ark.py` | Volcengine Ark | Models hosted on Volcengine Ark |
| `sglang` | `backends/sglang.py` | SGLang | Self-hosted models (RL training) |
| `azure` | `backends/azure.py` | Azure Chat Completions | Azure-hosted OpenAI models |

## Configuration Presets

### Production presets (ready to use)

| File | Model / Service |
|------|-----------------|
| `configs/llm/anthropic.yaml` | Claude Sonnet 4.6 (adaptive thinking) |
| `configs/llm/openai.yaml` | GPT-4o (Chat Completions) |
| `configs/llm/openai_response.yaml` | GPT-4.1 (Responses API) |
| `configs/llm/ark.yaml` | Volcengine Ark |
| `configs/llm/sglang.yaml` | SGLang (RL mode) |
| `configs/llm/azure.yaml` | Azure OpenAI |

### Model examples (reference / copy-and-adapt)

| File | Model | Protocol |
|------|-------|----------|
| `configs/llm/examples/kimi.yaml` | Kimi K2.5 | OpenAI-compatible |
| `configs/llm/examples/glm5.yaml` | GLM-5 | OpenAI-compatible |
| `configs/llm/examples/qwen.yaml` | Qwen 3.5 | OpenAI-compatible |
| `configs/llm/examples/deepseek.yaml` | DeepSeek Reasoner | OpenAI-compatible |
| `configs/llm/examples/minimax.yaml` | MiniMax M2.5 | OpenAI-compatible |
| `configs/llm/examples/minimax_anthropic.yaml` | MiniMax M2.5 | Anthropic-compatible |

---

## Backend Reference

### OpenAI Chat Completions (`backend: openai`)

Standard `chat.completions.create` interface. Also serves as the backend for all OpenAI-compatible providers (Kimi, GLM-5, Qwen, DeepSeek, MiniMax).

```yaml
# configs/llm/openai.yaml
backend: openai
base_url: ${OPENAI_BASE_URL:-https://api.openai.com/v1}
api_key: ${OPENAI_API_KEY}
model: gpt-4o
params:
  temperature: 0.0
  max_tokens: 4096
```

**Reasoning-related config fields:**

| Field | Purpose |
|-------|---------|
| `reasoning.preserve` | Whether to send reasoning back to the API on subsequent turns |
| `reasoning.format` | Field name for round-trip: `reasoning_content`, `reasoning_details`, or `auto` |
| `extra` | Provider-specific parameters, injected into `extra_body` |

**Provider-specific `extra` handling:**

| Key | Behavior | Used by |
|-----|----------|---------|
| `clear_thinking` | Merged into the `thinking` dict (not sent as sibling) | GLM-5 |
| `enable_thinking` | Passed through in `extra_body` | Qwen 3.5 |
| `reasoning_split` | Passed through in `extra_body` | MiniMax (OpenAI) |

### OpenAI Responses API (`backend: openai_response`)

OpenAI's latest `responses.create` interface with built-in state management and structured reasoning.

```yaml
# configs/llm/openai_response.yaml
backend: openai_response
api_key: ${OPENAI_API_KEY}
model: gpt-4.1
params:
  temperature: 0.0
  max_tokens: 4096
reasoning:
  effort: medium    # low | medium | high
  summary: auto     # auto | concise
```

**Conversation state management (two mutually exclusive strategies):**

| Strategy | Trigger | Behavior |
|----------|---------|----------|
| **Continuation** | `previous_response_id` found in history | Only new input items are sent. The API already holds prior context. |
| **Manual replay** | No `response_id` in history | All items are sent, including reasoning items from previous turns. |

The backend auto-selects the strategy based on whether any assistant message in the history carries a `response_id`. The two strategies are never mixed.

**Reasoning preservation:** `_parse_response()` stores reasoning items (including `encrypted_content`) alongside the `response_id` in `reasoning_raw`, enabling both training export and round-trip replay.

### Anthropic Messages API (`backend: anthropic`)

Native Anthropic Messages API for Claude models. Also works with any Anthropic-compatible provider (e.g., MiniMax) via `base_url`.

```yaml
# configs/llm/anthropic.yaml
backend: anthropic
api_key: ${ANTHROPIC_API_KEY}
model: claude-sonnet-4-6
params:
  max_tokens: 16384
thinking: true
thinking_type: adaptive
```

**Thinking control (strict-explicit, no silent defaults):**

| Configuration | API Parameter | Constraint |
|---------------|--------------|------------|
| `thinking_type: adaptive` | `{"type": "adaptive"}` | Claude 4.6 models only (`claude-opus-4-6*`, `claude-sonnet-4-6*`) |
| `thinking_budget: 10000` | `{"type": "enabled", "budget_tokens": 10000}` | Must be a positive integer (Pydantic `gt=0`) |
| Neither set | **ValueError** | Backend refuses to guess |
| `thinking_type: adaptive` on non-4.6 | **ValueError** | No silent downgrade |

**Type safety:**
- `thinking_type` is `Literal["adaptive", "enabled"]`. Typos like `"adpative"` are rejected at config construction by Pydantic.
- `thinking_budget` is `Annotated[int, Field(gt=0)]`. Zero and negative values are rejected at config construction.
- Runtime `budget_tokens <= 0` passed via `thinking_config` raises `ValueError` as a second line of defense.

**Content block round-trip:** Anthropic's interleaved thinking returns `thinking`, `text`, `tool_use`, and `redacted_thinking` blocks. `_parse_response()` stores all blocks in original order in `reasoning_raw`. `_serialize_messages()` detects full blocks and replays them losslessly, preserving `signature` fields required by the API.

**Anthropic-compatible providers (e.g., MiniMax):** Configure with `backend: anthropic` and `base_url` pointing to the compatible endpoint. Do **not** set `thinking: true` — that would trigger Claude-specific thinking control, which these providers do not support. Content block parsing and round-trip work automatically since they are protocol-level, not provider-specific.

```yaml
# configs/llm/examples/minimax_anthropic.yaml
backend: anthropic
base_url: https://api.minimax.io/anthropic
api_key: ${MINIMAX_API_KEY}
model: MiniMax-M2.5
params:
  max_tokens: 8192
  temperature: 1.0
reasoning:
  preserve: true
```

### Volcengine Ark (`backend: ark`)

Volcengine Ark-hosted models. Uses an independent SDK (`volcenginesdkarkruntime`) with an OpenAI-compatible interface.

```yaml
# configs/llm/ark.yaml
backend: ark
base_url: ${ARK_BASE_URL}
api_key: ${ARK_API_KEY}
model: ${ARK_MODEL_ID}
params:
  max_tokens: 16384
thinking: true
```

Supports `reasoning_content` extraction from responses and cross-turn preservation. `_serialize_messages()` writes `reasoning_content` on assistant messages for models that require it.

### SGLang (`backend: sglang`)

Self-hosted inference engine, primarily for RL training scenarios.

```yaml
# configs/llm/sglang.yaml
backend: sglang
base_url: http://localhost:30000
model: SGLANG_ENGINE
params:
  temperature: 0.7
  max_tokens: 4096
return_tokens: true
return_logprobs: true
```

RL-specific fields on `LLMResponse`: `prompt_token_ids`, `completion_token_ids`, `logprobs`, `weight_version`, `finish_status`.

---

## Reasoning Modes

Different models handle reasoning in fundamentally different ways. The framework unifies them through `ReasoningConfig`.

### Per-Model Overview

| Model | Reasoning Format | Preserve? | Field Name | Key Config |
|-------|-----------------|-----------|------------|------------|
| Claude 4.6 | Interleaved content blocks | Yes (signature required) | content blocks | `thinking: true`, `thinking_type: adaptive` |
| Kimi K2.5 | `reasoning_content` field | Yes (tool call turns) | `reasoning_content` | `reasoning.preserve: true` |
| GLM-5 | `reasoning_content` + `clear_thinking` | Yes (Preserved Thinking) | `reasoning_content` | `reasoning.preserve: true`, `extra.clear_thinking: false` |
| MiniMax M2.5 (OpenAI) | `reasoning_details` list | Yes | `reasoning_details` | `reasoning.format: reasoning_details`, `extra.reasoning_split: true` |
| MiniMax M2.5 (Anthropic) | Content blocks | Yes | content blocks | `reasoning.preserve: true` |
| Qwen 3.5 | `reasoning_content` field | No | `reasoning_content` | `reasoning.preserve: false`, `extra.enable_thinking: true` |
| DeepSeek Reasoner | `reasoning_content` field | No (400 if sent back) | `reasoning_content` | `reasoning.preserve: false` |
| GPT-4.1 (Response API) | Reasoning items | Auto (continuation) | Response API items | `reasoning.effort: medium` |

### `ReasoningConfig` Fields

```yaml
reasoning:
  preserve: true              # Send reasoning back on subsequent turns
  format: reasoning_content   # Field name: reasoning_content | reasoning_details | think_tags | auto
  effort: medium              # Responses API only: low | medium | high
  summary: auto               # Responses API only: auto | concise
```

### Common Configuration Patterns

**Pattern A: Preserve reasoning across turns (Kimi, GLM-5, MiniMax)**

```yaml
reasoning:
  preserve: true
  format: reasoning_content    # or reasoning_details for MiniMax OpenAI
```

**Pattern B: Strip reasoning (Qwen, DeepSeek)**

```yaml
reasoning:
  preserve: false
  format: reasoning_content
```

**Pattern C: Anthropic content block round-trip (Claude, MiniMax Anthropic)**

No `reasoning.format` needed. The Anthropic backend handles content block replay automatically.

```yaml
reasoning:
  preserve: true
```

**Pattern D: Responses API with effort control (GPT-4.1)**

```yaml
reasoning:
  effort: medium
  summary: auto
```

---

## LLMConfig Reference

All fields with types and defaults:

```yaml
# ── Connection ──
backend: openai                   # openai | openai_response | anthropic | ark | sglang | azure
base_url: null                    # str | null — API base URL
api_key: null                     # str | null — API key
model: gpt-4o                    # str — model name or endpoint ID

# ── Generation Parameters ──
params:                           # dict[str, Any] — passed directly to the API
  temperature: 0.0
  max_tokens: 4096
stop: null                        # list[str] | null — stop sequences
response_format: null             # dict | null — structured output format

# ── Thinking Control (Anthropic) ──
thinking: false                   # bool — enable thinking mode
thinking_type: null               # Literal["adaptive", "enabled"] | null — validated enum
thinking_budget: null             # int > 0 | null — token budget for manual mode

# ── Reasoning ──
reasoning:
  preserve: null                  # bool | null — preserve reasoning across turns
  format: auto                   # str — reasoning_content | reasoning_details | think_tags | auto
  effort: null                   # str | null — Responses API: low | medium | high
  summary: null                  # str | null — Responses API: auto | concise

# ── Reliability ──
retry:
  max_attempts: 5
  backoff: exponential            # exponential | linear | fixed
  base_delay: 1.0
  max_delay: 60.0
cache:
  enabled: false
  ttl: 3600
timeout: 1200.0                   # float — request timeout in seconds

# ── RL Training ──
return_tokens: false              # bool — return token IDs in LLMResponse
return_logprobs: false            # bool — return log probabilities

# ── Provider Extensions ──
extra: {}                         # dict[str, Any] — injected into extra_body
```

---

## Adding a New Backend

1. Create `awe_agent/core/llm/backends/your_backend.py` implementing the `LLMBackend` protocol:

```python
from awe_agent.core.llm.config import LLMConfig
from awe_agent.core.llm.types import LLMResponse, Message

class YourBackend:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        # Initialize your client here

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 1. Serialize messages to your API format
        # 2. Call your API
        # 3. Parse the response into LLMResponse
        #    - Set reasoning_text for human-readable reasoning
        #    - Set reasoning_raw for provider-specific payload (round-trip)
        return LLMResponse(content=..., reasoning_text=..., reasoning_raw=...)

    async def close(self) -> None:
        # Clean up client resources
        pass
```

2. Register the entry point in `pyproject.toml`:

```toml
[project.entry-points."awe_agent.llm_backend"]
your_backend = "awe_agent.core.llm.backends.your_backend:YourBackend"
```

3. Create a YAML preset at `configs/llm/your_backend.yaml`.

4. Use it:

```python
config = LLMConfig(backend="your_backend", model="your-model")
async with LLMClient(config) as client:
    response = await client.chat([Message(role="user", content="Hello")])
```

No changes to `LLMClient`, middleware, or any existing backend are needed.

---

## Quick Start

```python
from awe_agent.core.llm import LLMClient, LLMConfig, Message

# Load from YAML
from awe_agent.core.config.loader import load_yaml
config = LLMConfig(**load_yaml("configs/llm/anthropic.yaml"))

# Or construct directly
config = LLMConfig(
    backend="anthropic",
    model="claude-sonnet-4-6",
    thinking=True,
    thinking_type="adaptive",
    params={"max_tokens": 16384},
)

async with LLMClient(config) as client:
    response = await client.chat([
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello!"),
    ])
    print(response.content)           # Text output
    print(response.reasoning_text)    # Human-readable reasoning (for logs)
    print(response.reasoning_raw)     # Raw provider payload (for round-trip)
```
