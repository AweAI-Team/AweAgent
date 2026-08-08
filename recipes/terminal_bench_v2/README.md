# Terminal-Bench 2.0

Run [Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2) with either the **CalibForge** or **Terminus-2** scaffold. The default configuration selects CalibForge.

Evaluation runs `/tests/test.sh` in the same container after the agent finishes, matching the benchmark's stateful evaluation model. No patch export or fresh evaluation container is involved.

## Scaffolds

### CalibForge

CalibForge is a code-agent scaffold with bash, file-editing, and finish tools for terminal tasks.

The released scaffold is an independent implementation based on the code-agent setup described in the [DeepSeek-V4 paper](https://arxiv.org/abs/2606.19348), not an official DeepSeek implementation.

> **Release scope:** this repository contains the CalibForge evaluation scaffold, the Terminal-Bench 2.0 task and evaluator, and the scripts and configuration needed to run them.

CalibForge exposes exactly three tools:

| Tool | Purpose |
|:--|:--|
| `execute_bash` | Run commands in the task container |
| `str_replace_editor` | Inspect, create, and edit files |
| `finish` | End the rollout when the task is complete |

CalibForge requires `agent.tool_call_format: openai_function`.

### Terminus-2

Terminus-2 is a tmux-backed terminal agent driven by raw JSON keystrokes on the shared `AgentLoop`. Select it with `agent.type: terminus_2`; the Terminal-Bench task will then install the required `tmux` and `asciinema` dependencies automatically.

For the Harbor-compatible JSON reproduction, configure the endpoint and model, then use the dedicated public config:

```bash
export TERMINUS2_BASE_URL="http://localhost:8000/v1"
export TERMINUS2_API_KEY="dummy"
export TERMINUS2_MODEL="your-model"

python recipes/terminal_bench_v2/run.py \
    --config configs/tasks/terminal_bench_v2_official.yaml \
    --instance-ids sanitize-git-repo
```

The reproduction config is
[`configs/tasks/terminal_bench_v2_official.yaml`](../../configs/tasks/terminal_bench_v2_official.yaml).
Set `TERMINUS2_TOKENIZER_PATH` to a local tokenizer directory when the served model name is not available in LiteLLM's token-counting map.

## Prerequisites

1. Install AweAgent from the repository root; see the [main README](../../README.md#rocket-installation). If you are not installing the development dependencies, include the Terminus extra: `pip install -e ".[terminus2]"`.
2. Make sure the current user can run Docker.
3. Configure an LLM backend. The default task config includes `configs/llm/openai.yaml`.
4. Download the pinned Terminal-Bench 2.0 task checkout.

```bash
# Required for the default OpenAI-compatible backend
export OPENAI_API_KEY="your-key"

# Optional: custom OpenAI-compatible endpoint and model
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="your-model"

# Download the dataset at the commit used by this recipe
bash datasets/download.sh terminal_bench_v2
```

The downloader checks out commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`, detects the directory containing the task folders, and writes the full instance list. The default config then uses:

```text
datasets/terminal_bench_v2/tasks
datasets/terminal_bench_v2/instance_ids.json
```

Each task folder must contain `instruction.md`, `task.toml`, `environment/`, and `tests/`.

### Container network settings

The Terminal-Bench task forwards the following public network settings into the task container:

```bash
# Optional PyPI index; defaults to https://pypi.org/simple
export TERMINAL_BENCH_V2_PYPI_INDEX="https://pypi.org/simple"

# Optional proxy settings
export HTTP_PROXY="..."
export HTTPS_PROXY="..."
export ALL_PROXY="..."
export NO_PROXY="..."
```

Do not put credentials directly in the task YAML or shell scripts.

## Quick Start

From the AweAgent repository root, start with one instance:

```bash
python recipes/terminal_bench_v2/run.py \
    --instance-ids sanitize-git-repo \
    --max-concurrent 1
```

Then run every instance listed in the downloaded `instance_ids.json`:

```bash
python recipes/terminal_bench_v2/run.py
```

Use task data from another checkout:

```bash
python recipes/terminal_bench_v2/run.py \
    --task-data-dir /path/to/terminal-bench-2/tasks \
    --data-file /path/to/instance_ids.json
```

The shell wrapper accepts the same recipe-specific overrides:

```bash
bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
    --instance-ids sanitize-git-repo \
    --model your-model \
    --max-concurrent 1
```

Use the Python recipe or its shell wrapper for Terminal-Bench 2.0 evaluation. The TB2-specific timeout and resource overrides documented below are intentionally not added to the generic `awe-agent run` command.

## CalibForge Evaluation Setting

For CalibForge reproduction, use one-hour agent timeout, 16 CPUs, and 32 GiB of memory per sandbox. The default config already sets agent timeout to 3600 seconds; the following command makes the agent timeout explicit and overrides the default resource limits:

```bash
python recipes/terminal_bench_v2/run.py \
    --max-steps 200 \
    --agent-timeout 3600 \
    --cpu-milli 16000 \
    --memory-mb 32768
```

## Reproducibility Results

### CalibForge

Results on Terminal-Bench 2.0 using the CalibForge scaffold:

| Model | Reasoning effort | Result source | TB2 accuracy (%) |
|:--|:--:|:--|--:|
| Qwen3-30B-A3B-Instruct (base) | — | Reference baseline | 7.87 ± 0.00 |
| Qwen3.5-35B-A3B (base) | — | Reference baseline | 39.10 ± 1.09 |
| DeepSeek-V4-Pro | high | AweAgent reproduction: 58/89, 59/89, 60/89 | **66.29 ± 0.65** |

The DeepSeek-V4-Pro result is reported as mean ± SEM over three runs; the per-run accuracies are 65.17%, 66.29%, and 67.42%.

### Terminus-2

Results on Terminal-Bench 2.0 with the AweAgent release aligned with Harbor Leaderboard evaluation settings:

| Model | Harbor Leaderboard | AweAgent Release |
|:--|--:|--:|
| GLM 5 | 52.4% ± 2.6% | 51.35% ± 0.97% |
| Qwen3.6-35B-A3B | 51.5% | 46.44% ± 1.50% |
| Kimi K2 Thinking | 35.7% ± 2.8% | 37.09% |
| GLM 4.7 | 33.4% ± 2.8% | 34.68% ± 1.19% |
| MiniMax M2.1 | 29.2% ± 2.9% | 30.33% |
| Kimi K2 | 27.8% ± 2.5% | 24.71% |

## Configuration

The default config is [`configs/tasks/terminal_bench_v2.yaml`](../../configs/tasks/terminal_bench_v2.yaml) and selects CalibForge:

```yaml
agent:
  type: calibforge
  max_steps: 500

task:
  type: terminal_bench_v2
  override_agent_timeout: 3600

eval:
  enabled: true
```

To use Terminus-2 instead, change only the scaffold selection:

```yaml
agent:
  type: terminus_2
```

The task derives its environment setup from `agent.type`: both scaffolds receive the configured PyPI and proxy environment, while only `terminus_2` installs the `tmux` and `asciinema` dependencies. No separate setup-mode setting is required.

### Timeout precedence

| Timeout | Highest priority | Then | Fallback |
|:--|:--|:--|:--|
| Agent rollout | `--agent-timeout` | `task.override_agent_timeout` | `[agent].timeout_sec` in each task's `task.toml` |
| Verifier | `--verifier-timeout` | `eval.verifier_timeout` | `[verifier].timeout_sec` in each task's `task.toml` |

`eval.timeout` remains available for compatibility with the common evaluation config, but it is not treated as the Terminal-Bench verifier timeout.

### Resource precedence

Without resource CLI flags, each task's `task.toml` may override the baseline `runtime.resource_limits` from YAML. Supplying either `--cpu-milli` or `--memory-mb` makes the global runtime limits authoritative for every task; normally pass both together:

```bash
python recipes/terminal_bench_v2/run.py \
    --cpu-milli 16000 \
    --memory-mb 32768
```

CPU values are expressed in millicores (`16000` means 16 CPUs), and memory is expressed in MiB (`32768` means 32 GiB).

## CLI Arguments

```text
--task-data-dir DIR     Root directory of task folders (or TASK_DATA_DIR env)
--data-file PATH        JSON file containing an instance ID array (or DATA_FILE env)
--config, -c PATH       Task config (default: configs/tasks/terminal_bench_v2.yaml)
--instance-ids ID ...   Run only the selected instance IDs
--model MODEL           Override the configured LLM model
--max-steps N           Override the maximum number of agent steps
--agent-timeout SEC     Override the wall-clock agent timeout for every task
--verifier-timeout SEC  Override the /tests/test.sh timeout for every task
--max-concurrent N      Override evaluation concurrency
--cpu-milli N           Force a global CPU limit in millicores
--memory-mb N           Force a global memory limit in MiB
--output DIR            Override the output directory
--skip-eval             Run the agent without the verifier
--no-trajectories       Do not save per-instance trajectories
--verbose               Enable DEBUG logging
```

## Output

Results are written under `results/terminal_bench_v2/` by default:

```text
results/terminal_bench_v2/
  <model>_<timestamp>/
    results.jsonl
    trajectories.jsonl
    run_config.json
```

`results.jsonl` contains one record per instance, including its score, finish reason, duration, error (if any), and evaluator details. `trajectories.jsonl` stores the agent interaction history, and `run_config.json` records the resolved configuration used for the run.

## Troubleshooting

**Docker permission errors** — verify that `docker info` works without `sudo`.

**Dataset paths are missing** — run `bash datasets/download.sh terminal_bench_v2`, or pass both `--task-data-dir` and `--data-file` for a custom checkout.

**The LLM request times out** — increase `timeout` in the selected LLM config. This is separate from the per-instance `--agent-timeout`.

**Packages cannot be installed in the task container** — set `TERMINAL_BENCH_V2_PYPI_INDEX` to an accessible public mirror and configure the proxy environment variables if required by your network.

**A task ignores the YAML CPU or memory value** — per-task `task.toml` values take precedence by default. Pass `--cpu-milli` and `--memory-mb` to force the global limits.
