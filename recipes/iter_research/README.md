# IterResearch

Long-horizon research agent based on **IterResearch** (arXiv:2511.07327, ICLR
2026): *Rethinking Long-Horizon Agents via Markovian State Reconstruction*.

## What makes it different

Most agents accumulate the full chat transcript. IterResearch does not. Each
turn the model emits:

- a `<report>` — its **rewritten working memory** (everything learned so far), and
- a `<tool_call>` (a JSON object `{"name": ..., "arguments": {...}}`) **or** a
  final `<answer>`.

After a tool runs, the entire window is **discarded** and the next turn's single
user message is reconstructed from `{question, tools, last report, last action,
last observation}`. The report *is* the state. Context stays bounded (≈ one
report + one observation) no matter the horizon, which is what lets the agent
sustain hundreds of tool calls.

This is incompatible with the stock append-only `AgentLoop`, so the scaffold
supplies its own `IterResearchLoop` via `Agent.create_loop`. It still reuses the
framework's primitives (`ctx.llm`, `ctx.get_tool`, `ctx.trajectory`,
`AgentResult`), so it plugs into the runner, the trajectory exporter, and the
BrowseComp LLM judge unchanged.

## Tools

Three tools, exposed to the model under their IterResearch names and mapped to
AweAgent tools:

| Model sees (external) | AweAgent tool (internal) | Notes |
|---|---|---|
| `google_search` | `web_search` | query array passed through |
| `Visit` | `web_fetch` | URL array fans out over single-URL fetches; results concatenated; `goal`→`prompt` |
| `PythonInterpreter` | `python_interpreter` | remote SandboxFusion sandbox (`core/tool/public`) |

## Setup

```bash
# The Python sandbox client (sandbox_fusion + json5) ships with the base install
pip install -e .                      # or: uv pip install sandbox_fusion json5

# LLM — DeepSeek V4 Pro (configs/tasks/iter_research.yaml reads these)
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"   # default
export DEEPSEEK_MODEL="deepseek-v4-pro"                  # default

# Tool backends — env-driven (serpapi/jina, or a custom backend, etc.)
export SEARCH_BACKEND="serpapi"; export SERPAPI_API_KEY="..."
export READER_BACKEND="jina";    export JINA_API_KEY="..."

# Visit page-summary model (DeepSeek); else web_fetch returns raw content
export WEB_FETCH_CONFIG_PATH="configs/llm/web_fetch/deepseek_v4_pro.yaml"

# Python sandbox (only if python_interpreter is enabled)
export SANDBOX_FUSION_ENDPOINTS="http://host-a:8080,http://host-b:8080"  # multi-endpoint pool
```

## Choosing tools (e.g. drop the sandbox)

The enabled tools are just the `agent.tools` list in
`configs/tasks/iter_research.yaml`. To run **search + fetch only** (no Python
sandbox), remove one line:

```yaml
agent:
  tools:
    - web_search
    - web_fetch
    # - python_interpreter   # ← commented out: no sandbox
```

The prompt's tool schema and the dispatch table are built only from this list,
so nothing else needs to change (and SandboxFusion isn't contacted).

## Run

```bash
bash recipes/iter_research/run_iter_research_browsecomp.sh \
    --data-file /path/to/bc_en200.jsonl \
    --max-steps 256 \
    --max-concurrent 80
```

`--dry-run` lists instances without calling the model. `--instance-ids ID ...`
runs a subset.

## Debug in VSCode

`.vscode/launch.json` (at the workspace root, one level above `AweAgent/`) ships
three debugpy configs:

- **IterResearch BrowseComp — debug 1 instance (eval off)** — single-flight
  (`--max-concurrent 1`), one instance, judge skipped. Set breakpoints in
  `aweagent/scaffold/iter_research/loop.py` and inspect `response.content`,
  `text`, `report`, `tool_call`, `observation`, and the reconstructed
  `window[0].content` each turn to see whether the model follows the format.
- **… — dry run (list instances)** — prints instance ids without calling the model.
- **IterResearch — pytest** — debug the scaffold + tool tests.

`justMyCode` is false so you can step into framework code. Edit the
`DEEPSEEK_API_KEY` / `BROWSECOMP_DATA_FILE` / backend env in the config as needed.

## Configuration notes

- **Token budget, not a cap-bug.** The DeepSeek `llm` block sets
  `max_completion_tokens: 65536` — generous so a full `<report>` plus a long
  `<answer>` is never truncated mid-tag (which would fail format validation).
  Avoid the dangerous 4096 default; the agent logs a warning if it sees a cap on
  the LLM config it's handed.
- **`max_turn` is `agent.max_steps`** (or `--max-steps`). BrowseComp uses 256.
- **Thinking mode.** The recipe runs DeepSeek V4 Pro with `extra.thinking.type:
  enabled`. In thinking mode DeepSeek *ignores* temperature/top_p/presence_penalty
  (harmless but inert). To run closer to the original IterResearch (Qwen,
  non-thinking, where those sampling values are load-bearing), turn thinking off;
  the loop's `extra.iter_research.sampling` then takes effect.
- **Sampling** (temperature 0.6, top_p 0.95, presence_penalty 1.5) is supplied by
  the loop and overridable under `extra.iter_research.sampling`.
- **Observation truncation** to 32000 tokens uses tiktoken (`o200k_base`) as an
  estimate; it falls back to char truncation if the vocab is unavailable offline.
  Override via `extra.iter_research.observation_max_tokens` / `tokenizer_encoding`.
- **Page summarizer.** `Visit` (`web_fetch`) summarizes fetched pages with an LLM;
  point `WEB_FETCH_CONFIG_PATH` at `configs/llm/web_fetch/deepseek_v4_pro.yaml`.
  Without a summarizer it returns raw content, which hits the 32k truncation often.
- **Tag location.** The protocol parses `<report>`/`<tool_call>` from the message
  *content*. The loop falls back to the reasoning channel if `content` carries no
  tags, but smoke-test (single-step) that your endpoint emits the tags in content.

Scoring reuses the BrowseComp task + LLM judge: the loop puts the final
`<answer>` in `AgentResult.metadata["final_answer"]` with `finish_reason="finish"`.

Training mode is intentionally not wired.
