# ScaleSWE

Run the ScaleSWE benchmark — a large-scale SWE-bench style coding evaluation. Uses the SearchSWE agent scaffold in non-search mode with CodeAct XML tool calling.

## Prerequisites

1. **Docker** — each instance runs in an isolated container
2. **LLM API** — configure in `configs/llm/` (default: Azure OpenAI)
3. **Data file** — ScaleSWE JSONL format

### Environment Variables

```bash
export AZURE_OPENAI_ENDPOINT="https://your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"
```

## Quick Start

```bash
# 1. List instances (no Docker)
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl --mode dry-run

# 2. Inspect prompt for one instance (no Docker)
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl \
    --instance-id <ID> --mode prompt

# 3. Debug single instance
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl \
    --instance-id <ID> --mode debug --verbose

# 4. Batch run (all instances)
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl --mode batch

# 5. Batch run (specific instances)
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl --mode batch \
    --instance-ids inst_001 inst_002
```

Or use the shell wrapper:

```bash
bash recipes/scale_swe/run_scale_swe.sh \
    --data-file /path/to/scaleswe.jsonl --model gpt-4o
```

## Modes

| Mode | Description | Docker |
|------|-------------|:------:|
| `dry-run` | List all instances | No |
| `prompt` | Print prompt and task_info | No |
| `debug` | Single-instance run with step trace | Yes |
| `batch` | Concurrent batch run, JSONL output | Yes |

## CLI Arguments

```
--data-file PATH          JSONL data file (required)
--config / -c PATH        Config file (default: configs/tasks/scale_swe.yaml)
--mode MODE               prompt | debug | batch | dry-run
--instance-id ID          Single instance (prompt/debug)
--instance-ids ID ...     Multiple instances (batch)
--model MODEL             Override LLM model
--max-steps N             Override max steps
--max-concurrent N        Override concurrency
--output DIR              Output directory
--skip-eval               Skip evaluation
--no-trajectories         Don't save per-instance trajectories
--verbose                 DEBUG logging
```

## Config

Default config: `configs/tasks/scale_swe.yaml`

```yaml
agent:
  type: search_swe
  max_steps: 200
  enable_search: false        # no web search
  bash_timeout: 1200
  tool_call_format: codeact_xml

execution:
  max_concurrent: 50
  max_retries: 3
```

## Output

```
results/scale_swe/<run_id>/
  results.jsonl           # per-instance results
  config.json             # config snapshot
  trajectories/           # per-instance agent traces
```
