# DeNovoSWE

Run the DeNovoSWE benchmark with the SearchSWE agent scaffold.  The agent
builds a package from scratch against a spec inlined in its prompt; the
docker image's original source has been wiped by `clean.sh` before the
agent runs.  Evaluation re-runs `clean.sh`, applies the agent patch +
golden `test_patch`, and runs `pytest` per file.

See [`aweagent/tasks/denovo_swe/README.md`](../../aweagent/tasks/denovo_swe/README.md)
for task-level documentation (workflow, data format, anti-hack details).

## Prerequisites

1. **Docker / Portal runtime** — each instance runs in an isolated container.
2. **LLM API** — configure your backend (see `configs/llm/`).
3. **Input data** — JSONL produced by Step 1 below.

```bash
# LLM (pick a backend, see configs/llm/)
export AZURE_OPENAI_ENDPOINT="https://your-endpoint"
export AZURE_OPENAI_API_KEY="your-key"

# Search mode (optional — DeNovoSWE defaults to non-search)
export SEARCH_BACKEND="serpapi"
export READER_BACKEND="jina"
export SERPAPI_API_KEY="your-serpapi-key"
export JINA_API_KEY="your-jina-key"
export WEB_FETCH_CONFIG_PATH="/path/to/configs/llm/web_fetch/azure.yaml"
```

## Step 1 — Extract test patches

Reads the raw JSONL, launches one sandbox per instance, finds every
unit-test file referenced by `passed_ptp`, and writes a unified diff plus
a base64-encoded tar.gz of binary fixtures back into the row.  Each run
auto-creates a timestamped subdirectory under `--output`.

```bash
python recipes/denovo_swe/extract_patch.py \
    --input /path/to/ready_denovoswe.jsonl \
    --output ./results/denovoswe \
    --config configs/tasks/denovoswe.yaml \
    --max-concurrent 10
```

Output layout:

```
results/denovoswe/extract_patch_20260601_120000/
    results.jsonl       # full results (incremental writes)
    status.jsonl        # per-instance success/error/no_test_files
    run_config.json     # config snapshot
    extract.log         # full log
```

Useful flags:

```bash
# Dry run — list instances without launching sandboxes
--dry-run

# Restrict to specific instances
--instance-ids PyCQA_pep8_pr970 ...

# Delete each docker image after extraction
--del-done-images

# Skip in-container pypi_name extraction (faster if input is pre-populated)
--no-extract-package-info
```

## Step 2 — Run agent + eval

The TaskRunner schedules the agent in a docker session, then opens a new
session for evaluation.  All four modes auto-create a timestamped
subdirectory under `execution.output_path`.

```bash
# Inspect generated prompt (no Docker)
python recipes/denovo_swe/run.py \
    --data-file /path/to/denovoswe_with_patches.jsonl \
    --config configs/tasks/denovoswe.yaml \
    --instance-id PyCQA_pep8_pr970 --mode prompt

# Single-instance debug (full agent + eval trace)
python recipes/denovo_swe/run.py \
    --data-file /path/to/denovoswe_with_patches.jsonl \
    --config configs/tasks/denovoswe.yaml \
    --instance-id PyCQA_pep8_pr970 --mode debug --verbose

# Batch
python recipes/denovo_swe/run.py \
    --data-file /path/to/denovoswe_with_patches.jsonl \
    --config configs/tasks/denovoswe.yaml \
    --mode batch --max-concurrent 50

# List instances
python recipes/denovo_swe/run.py \
    --data-file /path/to/denovoswe_with_patches.jsonl \
    --config configs/tasks/denovoswe.yaml \
    --mode dry-run
```

### validate-run

Skips the agent and runs evaluation against the original source — handy
for sanity-checking that `test_patch` + `pytest` work end-to-end on a
fresh image.

```bash
python recipes/denovo_swe/run.py \
    --data-file /path/to/denovoswe_with_patches.jsonl \
    --config configs/tasks/denovoswe.yaml \
    --instance-id PyCQA_pep8_pr970 \
    --mode debug --validate-run --verbose
```

## Modes

| Mode | Description | Docker required |
|------|-------------|:----:|
| `dry-run` | List loaded instances | No |
| `prompt` | Print rendered prompt + task_info for one instance | No |
| `debug` | Full single-instance run with step-by-step trace | Yes |
| `batch` | Concurrent batch run, JSONL outputs | Yes |

## CLI Arguments

```
--data-file PATH           JSONL data file (required)
--config / -c PATH         Config file (default: configs/tasks/denovoswe.yaml)
--llm-config PATH          LLM backend config YAML (overrides LLM_CONFIG)
--mode MODE                prompt | debug | batch | dry-run
--instance-id ID           Single instance ID (prompt/debug)
--instance-ids ID ...      Multiple instance IDs (batch, optional)
--model MODEL              Override LLM model
--max-steps N              Override max agent steps
--max-concurrent N         Override concurrency (batch)
--enable-search            Enable search tools
--no-search                Disable search tools
--output DIR               Output directory
--skip-eval                Skip evaluation
--validate-run             Skip agent, run eval only
--del-done-images          docker rmi after each instance
--dump-clean-snapshot PATH Dump per-instance post-clean workspace
                           snapshots to a JSONL file
--prompt-version v1|v2     Prompt version (default: v2, with finish gate)
--verbose                  DEBUG-level logging
```

## Output

Batch results land in `execution.output_path/<model>_<timestamp>/`:

```
results/denovoswe/
  <model>_<timestamp>/
    results.jsonl          # one line per instance result (incremental)
    trajectories.jsonl     # per-instance agent trajectories
    run_config.json        # config snapshot
```

`results.jsonl` schema:

```json
{
  "instance_id": "PyCQA_pep8_pr970",
  "dataset_id": "denovo_swe",
  "success": true,
  "score": 0.96,
  "finish_reason": "finish",
  "eval_result": {
    "accepted": false,
    "score": 0.96,
    "details": {
      "passed": 92, "failed": 4, "errors": 0,
      "pass_rate": 0.958,
      "num_test_files": 6,
      "per_file": { "...": {...} }
    }
  }
}
```

Debug mode writes to `execution.output_path/debug_<instance_id>_<timestamp>/`:

```
debug_PyCQA_pep8_pr970_20260601_140000/
    results.jsonl
    trajectory.json
    agent.patch
    post_clean_snapshot.jsonl
    run_config.json
    debug.log
```

## Analyze Results

```bash
python recipes/denovo_swe/analyze_results.py \
    results/denovoswe/<run_dir>/results.jsonl
```

Computes `avg_pass_rate`, `correct_rate` (score == 1.0), `almost_correct_rate`
(score ≥ 0.8), and the error count.  Output is saved alongside as
`analysis.json`.

## Configuration

`configs/tasks/denovoswe.yaml` selects the LLM backend via `LLM_CONFIG`
(defaults to `configs/llm/openai.yaml`).  Switch backends without editing
the task YAML:

```bash
# CLI flag (takes precedence)
python recipes/denovo_swe/run.py --data-file data.jsonl --mode batch \
    --llm-config configs/llm/anthropic.yaml

# Env var
LLM_CONFIG=../llm/examples/glm5.yaml \
    python recipes/denovo_swe/run.py --data-file data.jsonl --mode batch
```

Key settings:

```yaml
agent:
  type: search_swe
  max_steps: 500          # spec-to-code builds can take many steps
  enable_search: false    # disable search by default — anti-hack
  bash_timeout: 1200

execution:
  max_concurrent: 50
  max_retries: 3
```

## Troubleshooting

**`clean.sh` failed** — almost always a missing python interpreter in the
image (the multi-Python uninstall pass needs at least one working one).
Run with `--verbose` to see the script output.

**`test_patch` rejected during eval** — usually means the agent created a
file at the same path `test_patch` is about to add.  The evaluator
pre-cleans add-only paths to avoid this; if you still see rejects, check
the `rejected_files` field in `details`.

**Score is 0 with `no_test_patch` error** — Step 1 (`extract_patch.py`)
failed for this instance.  Inspect `status.jsonl` in the extract run dir.
