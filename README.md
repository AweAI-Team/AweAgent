# AweAgent

A high-quality, extensible agent scaffold for code and search tasks. Built for the [BeyondSWE](https://xxx.com) benchmark family — supports doc2repo, cross-repo, refactor, and domain task types with pluggable LLM backends, isolated Docker evaluation, and web-search-augmented agents.

## Architecture

```
awe_agent/
  core/           # Framework internals
    agent/        #   Agent loop, context, trajectory
    condenser/    #   Context window management
    config/       #   YAML config loading & schema
    eval/         #   Evaluation framework (PatchTestEvaluator, isolation)
    llm/          #   LLM backends (OpenAI, Azure, Ark, SGLang)
    runtime/      #   Container runtimes (Docker)
    task/         #   Task protocol, runner, types
    tool/         #   Tool registry (bash, editor, search, think, finish)
  scaffold/       # Agent implementations
    search_swe/   #   SearchSWE agent with optional web search
  tasks/          # Benchmark-specific logic
    beyond_swe/   #   BeyondSWE task, evaluator, prompts
  integrations/   # External framework adapters (Slime RL)
  plugins/        # Plugin registry system

configs/          # YAML configurations (LLM, task, runtime)
recipes/          # Reproducible experiment entry points
  beyond_swe/     #   Unified BeyondSWE runner (prompt/debug/batch/dry-run)
tests/            # Test suite
```

## Installation

Requires **Python >= 3.11** and **Docker** (for container-based agent execution and evaluation).

### Option 1: uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that handles virtualenvs automatically.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> AweAgent && cd AweAgent
uv venv --python 3.11
uv pip install -e ".[dev]"

# Verify
uv run awe-agent info
```

### Option 2: conda + pip

```bash
# Create environment
conda create -n aweagent python=3.11 -y
conda activate aweagent

# Install
cd AweAgent
pip install -e ".[dev]"

# Verify
awe-agent info
```

### Optional extras

```bash
pip install -e ".[ark]"    # Volcengine Ark backend
pip install -e ".[k8s]"    # Kubernetes runtime
pip install -e ".[all]"    # Everything
```

> **Why `pip install -e .` matters**: AweAgent uses `pyproject.toml` entry-points for plugin discovery (LLM backends, runtimes, agents, evaluators, tools). Without installation, the plugin registry cannot discover components and will raise `KeyError`. Editable mode (`-e`) ensures code changes take effect immediately without reinstalling.

## Quick Start

### 1. Configure LLM

Set your API credentials:

```bash
export OPENAI_API_KEY="sk-..."
# or for Azure:
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com"
```

### 2. Run BeyondSWE

The `recipes/beyond_swe/run.py` is the recommended entry point, supporting 4 modes:

```bash
# Inspect prompt for a single instance (no Docker needed)
python recipes/beyond_swe/run.py \
    --data-file data.jsonl --instance-id inst_001 --mode prompt

# Debug single instance with full agent + eval trace
python recipes/beyond_swe/run.py \
    --data-file data.jsonl --instance-id inst_001 --mode debug \
    --model gpt-4o --max-steps 30 --verbose

# Batch run all instances
python recipes/beyond_swe/run.py \
    --data-file data.jsonl --mode batch

# Batch run without search tools (OpenHands style)
python recipes/beyond_swe/run.py \
    --data-file data.jsonl --mode batch --no-search

# List all instances without executing
python recipes/beyond_swe/run.py \
    --data-file data.jsonl --mode dry-run
```

Or use the CLI directly:

```bash
export DATA_FILE=/path/to/data.jsonl
awe-agent run -c configs/tasks/beyondswe_searchswe.yaml --dry-run
awe-agent run -c configs/tasks/beyondswe_searchswe.yaml
```

## Configuration

Configs are YAML files with environment variable substitution (`${VAR}`) and `!include` support:

```yaml
# configs/tasks/beyondswe_searchswe.yaml
llm: "!include ../llm/azure.yaml"

agent:
  type: search_swe
  max_steps: 200
  enable_search: true
  bash_timeout: 1200

task:
  type: beyond_swe
  dataset_id: beyond_swe
  data_file: ${DATA_FILE}

eval:
  enabled: true
  timeout: 3600

execution:
  max_concurrent: 50
  output_path: ./results/beyondswe_searchswe
```

### LLM Backends

| Backend | Config | Required Env Vars |
|---------|--------|-------------------|
| OpenAI | `configs/llm/openai.yaml` | `OPENAI_API_KEY` |
| Azure OpenAI | `configs/llm/azure.yaml` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Volcengine Ark | `configs/llm/ark.yaml` | `ARK_API_KEY` |
| SGLang | `configs/llm/sglang.yaml` | (self-hosted) |

## Development

```bash
# Run tests
pytest

# Lint
ruff check awe_agent/
ruff format awe_agent/

# Type check
mypy awe_agent/
```

## License

Apache-2.0
