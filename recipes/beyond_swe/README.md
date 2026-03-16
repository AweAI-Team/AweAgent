# BeyondSWE + SearchSWE

Run the BeyondSWE benchmark with the SearchSWE agent scaffold. Supports four task types (Doc2Repo, CrossRepo, DepMigrate, DomainFix) in both search-enabled and non-search modes.

## Dataset

Download the BeyondSWE dataset from [Hugging Face](https://huggingface.co/datasets/AweAI-Team/BeyondSWE):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AweAI-Team/BeyondSWE",
    repo_type="dataset",
    local_dir="<your_path>/BeyondSWE",
)
```

Or download the JSONL file directly from the Hugging Face repo page.

## Prerequisites

1. **Docker** — each instance runs in an isolated container
2. **LLM API** — configure Azure OpenAI or another backend (see `configs/llm/`)
3. **Data file** — BeyondSWE JSONL (see [Dataset](#dataset) above)
4. **Docker images** — must be available locally (use `crane pull` + `docker load` if network is restricted)

### Environment Variables

```bash
# Required — LLM
export AZURE_OPENAI_ENDPOINT="https://your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"

# Required for Doc2Repo evaluation
export BEYONDSWE_TEST_SUITE_DIR="/path/to/doc2repo_test_suite"

# Required for search mode (configure at least one search backend)
export SEARCH_BACKEND="serpapi"              # or jina
export READER_BACKEND="jina"                 # web page reader
export SERPAPI_API_KEY="your-serpapi-key"     # if using serpapi
export JINA_API_KEY="your-jina-key"          # if using jina
export LINK_SUMMARY_CONFIG_PATH="/path/to/configs/llm/link_summary/azure.yaml"
```

## Quick Start

### Batch run

```bash
# Search mode (default config)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --mode batch

# Specific instances only
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --mode batch \
    --instance-ids inst_001 inst_002

# Non-search mode (OpenHands style)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --mode batch \
    --no-search
```

### Shell script (batch only)

```bash
bash recipes/beyond_swe/run_beyond_swe_search.sh \
    --data-file /path/to/beyondswe.jsonl \
    --model gpt-4o \
    --max-steps 100
```

## Config Files

| File | Description |
|------|-------------|
| `configs/tasks/beyondswe_searchswe.yaml` | Main task config (search mode) |
| `configs/llm/azure.yaml` | Azure OpenAI LLM config |
| `configs/llm/link_summary/azure.yaml` | LLM config for search link summarization |

Key settings in `beyondswe_searchswe.yaml`:

```yaml
agent:
  type: search_swe
  max_steps: 200          # max agent steps per instance
  enable_search: true     # toggle search tools
  bash_timeout: 1200      # bash command timeout (seconds)

execution:
  max_concurrent: 50      # parallel instances
  max_retries: 3          # retry on failure
```

### Switching LLM Backends

The task config uses `LLM_CONFIG` to select the LLM backend at runtime — no need to create separate task YAML files per model.

There are two equivalent ways to specify the LLM config: `--llm-config` (CLI) or `LLM_CONFIG` (env var). CLI takes precedence.

```bash
# Default: OpenAI (configs/llm/openai.yaml)
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch

# Claude 4.6 — via CLI flag
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/anthropic.yaml

# Kimi K2.5 — via env var
LLM_CONFIG=../llm/examples/kimi.yaml \
    python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch

# GLM-5
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/examples/glm5.yaml

# Qwen 3.5
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/examples/qwen.yaml

# DeepSeek Reasoner
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/examples/deepseek.yaml

# MiniMax M2.5 (OpenAI-compatible)
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/examples/minimax.yaml

# MiniMax M2.5 (Anthropic-compatible)
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/examples/minimax_anthropic.yaml
```

**Path resolution:** `--llm-config` accepts either an absolute path or a path relative to the working directory. When using the `LLM_CONFIG` env var, the path is relative to `configs/tasks/` (since the task YAML contains the `!include`).

**Override model name within a backend** — combine `--llm-config` with `--model`:

```bash
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/anthropic.yaml --model claude-opus-4-6
```

**Logging:** The resolved LLM config path is printed at startup, so you can verify which backend is being used. The full LLM configuration (backend, model, params, etc.) is saved in the batch output `run_config.json`.

Each LLM preset reads its API key from an environment variable (e.g., `ANTHROPIC_API_KEY`, `MOONSHOT_API_KEY`, `DASHSCOPE_API_KEY`). See the corresponding YAML file for required variables.

Available presets: `configs/llm/` (production) and `configs/llm/examples/` (model-specific). For full details on reasoning modes and configuration, see [`doc/llm_client/README.md`](../../doc/llm_client/README.md).

## Modes

| Mode | Description | Docker Required |
|------|-------------|:---------------:|
| `dry-run` | List all instances | No |
| `prompt` | Print prompt and task_info for one instance | No |
| `debug` | Full single-instance run with step-by-step trace | Yes |
| `batch` | Concurrent batch run, outputs JSONL results | Yes |

## CLI Arguments

```
--data-file PATH          JSONL data file (required)
--config / -c PATH        Config file (default: configs/tasks/beyondswe_searchswe.yaml)
--llm-config PATH         LLM backend config YAML (overrides LLM_CONFIG env var)
--mode MODE               prompt | debug | batch | dry-run
--instance-id ID          Single instance ID (prompt/debug)
--instance-ids ID ...     Multiple instance IDs (batch, optional)
--model MODEL             Override LLM model
--max-steps N             Override max steps
--max-concurrent N        Override concurrency (batch)
--enable-search           Enable search tools
--no-search               Disable search tools
--output DIR              Output directory
--skip-eval               Skip evaluation
--verbose                 DEBUG logging
```

## Output

Batch results are saved to the `--output` directory (default `results/beyondswe_searchswe/`):

```
results/beyondswe_searchswe/
  <run_id>/
    results.jsonl           # one line per instance result
    run_config.json         # config snapshot
    trajectories.jsonl      # per-instance agent trajectories
```

Each line in `results.jsonl`:

```json
{
  "instance_id": "...",
  "dataset_id": "BeyondSWE",
  "task": "crossrepo",
  "success": true,
  "score": 1.0,
  "error": null,
  "finish_reason": "finish"
}
```

## Analyze Results

```bash
python recipes/beyond_swe/analyze_results.py \
    --result-dir <output_run_dir>
```

Statistics:
- **Doc2Repo** (domain): avg pass rate, almost correct count (>=90%), correct count (100%)
- **CrossRepo / DomainFix / DepMigrate**: solved % (score == 1.0)

Output is saved to `<result-dir>/analysis.json`.

## Troubleshooting

**LLM timeout** — add `timeout: 600` to `configs/llm/azure.yaml` (default is 120s, may be too short for large contexts).

**Search not working** — verify `SEARCH_BACKEND` and the corresponding API key are set.
