# BeyondSWE + SearchSWE

Run the BeyondSWE benchmark with the SearchSWE agent scaffold. Supports four task types (Doc2Repo, CrossRepo, DepMigrate, DomainFix) in both search-enabled and non-search modes.

## Prerequisites

1. **Docker** — each instance runs in an isolated container
2. **LLM API** — configure Azure OpenAI or another backend (see `configs/llm/`)
3. **Data file** — BeyondSWE JSONL format
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

### 1. List instances (no Docker needed)

```bash
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --mode dry-run
```

### 2. Inspect prompt (no Docker needed)

```bash
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --instance-id pylons_plaster_pastedeploy_pr14 \
    --mode prompt
```

### 3. Debug a single instance

```bash
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --instance-id pylons_plaster_pastedeploy_pr14 \
    --mode debug \
    --model gpt-4o \
    --max-steps 20 \
    --enable-search \
    --verbose
```

### 4. Batch run

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

### 5. Shell script (batch only)

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
  max_steps: 200          # max steps per instance
  enable_search: true     # toggle search tools
  bash_timeout: 1200      # bash command timeout (seconds)

execution:
  max_concurrent: 50      # parallel instances
  max_retries: 3          # retry on failure
```

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
    config.json             # config snapshot
    trajectories/           # per-instance agent trajectories
      inst_001.json
      inst_002.json
```

## Troubleshooting

**Cannot pull Docker image** — if network is restricted, download via crane and load locally:
```bash
HTTPS_PROXY=http://127.0.0.1:11110 crane pull <image:tag> /tmp/img.tar
docker load -i /tmp/img.tar
```

**LLM timeout** — add `timeout: 600` to `configs/llm/azure.yaml` (default is 120s, may be too short for large contexts).

**Search not working** — verify `SEARCH_BACKEND` and the corresponding API key are set.
