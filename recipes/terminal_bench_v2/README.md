# Terminal Bench 2.0

Run the [Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2.git) benchmark with the Terminus-2 agent scaffold. Evaluation runs in the same container as the agent (no patch, no isolation), aligned with the official Harbor framework.

## Dataset

Download the Terminal Bench 2.0 dataset from the [official repository](https://github.com/laude-institute/terminal-bench-2.git), and switch to commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`. The dataset consists of task folders; each folder contains `instruction.md`, `task.toml`, `environment/`, and `tests/`.

You need:
1. **Task data directory** — root directory containing all task folders
2. **Instance IDs file** — JSON array of instance IDs to run (e.g. `["sanitize-git-repo", "vulnerable-secret", ...]`)

## Prerequisites

1. **Docker** — each instance runs in an isolated container
2. **LLM API** — configure in `configs/llm/` (OpenAI-compatible)
3. **Task data** — Terminal Bench 2.0 task folders and instance IDs JSON

### Environment Variables

```bash
# Required — LLM (use your provider's base URL and API key)
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="your-key"

# Optional — PyPI index (default: https://pypi.org/simple)
# Set for restricted networks, e.g. internal mirror
export TERMINAL_BENCH_V2_PYPI_INDEX="..."

# Optional — Proxy for container network access
export HTTP_PROXY="..."
export HTTPS_PROXY="..."
```

## Quick Start

### List instances (dry-run)

```bash
python recipes/terminal_bench_v2/run.py \
    --task-data-dir /path/to/terminal-bench-2 \
    --data-file /path/to/instance_ids.json \
    --mode dry-run
```

### Inspect prompt (no Docker)

```bash
python recipes/terminal_bench_v2/run.py \
    --task-data-dir /path/to/terminal-bench-2 \
    --data-file /path/to/instance_ids.json \
    --instance-id configure-git-webserver \
    --mode prompt
```

### Debug single instance

```bash
python recipes/terminal_bench_v2/run.py \
    --task-data-dir /path/to/terminal-bench-2 \
    --data-file /path/to/instance_ids.json \
    --instance-id configure-git-webserver \
    --mode debug
```

### Batch run

```bash
python recipes/terminal_bench_v2/run.py \
    --task-data-dir /path/to/terminal-bench-2 \
    --data-file /path/to/instance_ids.json \
    --mode batch

# Specific instances only
python recipes/terminal_bench_v2/run.py \
    --task-data-dir /path/to/terminal-bench-2 \
    --data-file /path/to/instance_ids.json \
    --mode batch \
    --instance-ids task_001 task_002
```

### Shell script

```bash
bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
    --task-data-dir /path/to/terminal-bench-2 \
    --data-file /path/to/instance_ids.json \
    --model glm-5 \
    --mode batch
```

## Reproducibility Results

Results on Terminal Bench 2.0 with AweAgent release aligned with Harbor Leaderboard evaluation settings::

| Model          | Harbor Leaderboard | AweAgent Release |
| :------------- | :----------------- | :--------------- |
| GLM 4.7        | 33.4% ± 2.8        | 33.78%           |
| MiniMax M2.1   | 29.2% ± 2.9        | 30.33%           |
| GLM 5          | 52.4% ± 2.6        | 49.43%           |

## Modes

| Mode     | Description                          | Docker |
| :------- | :----------------------------------- | :----: |
| `dry-run` | List all instances                   | No     |
| `prompt`  | Print prompt and task_info           | No     |
| `debug`   | Single-instance run with step trace  | Yes    |
| `batch`   | Concurrent batch run, JSONL output   | Yes    |

## CLI Arguments

```
--task-data-dir DIR     Root directory of task folders (or TASK_DATA_DIR)
--data-file PATH        JSON file with instance ID array (or DATA_FILE)
--config / -c PATH      Config file (default: configs/tasks/terminal_bench_v2.yaml)
--mode MODE             prompt | debug | batch | dry-run
--instance-id ID        Single instance ID (prompt/debug)
--instance-ids ID ...   Instance IDs for batch (optional filter)
--model MODEL           Override LLM model
--max-steps N           Override max agent steps
--max-concurrent N      Override concurrency (batch)
--output DIR            Output directory (batch)
--skip-eval             Skip evaluation
--no-trajectories       Don't save trajectories
--verbose               DEBUG logging
```

## Config

Task config supports `override_agent_timeout` to override per-task timeouts (e.g. set to 7200 for 2h).

## Output

Batch results are saved to the `--output` directory (default `results/terminal_bench_v2/`):

```
results/terminal_bench_v2/
  <model>_<timestamp>/
    results.jsonl        # one line per instance
    trajectories.jsonl   # per-instance agent trajectories
    run_config.json      # config snapshot
```

Each line in `results.jsonl`:

```json
{
  "instance_id": "...",
  "dataset_id": "terminal_bench_v2",
  "success": true,
  "score": 1.0,
  "error": null,
  "finish_reason": "finish",
  "duration": 157.36
}
```

## Troubleshooting

**Docker permission** — ensure the user can run `docker` without sudo.

**LLM timeout** — add `timeout: 600` to the LLM config for large contexts.

**Restricted networks** — set `TERMINAL_BENCH_V2_PYPI_INDEX` to an accessible PyPI mirror; set `HTTP_PROXY` / `HTTPS_PROXY` to inject proxy for container network access.
