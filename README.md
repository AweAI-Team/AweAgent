# AweAgent

An extensible agent framework for software engineering benchmarks. Supports [BeyondSWE](https://github.com/AweAI-Team/BeyondSWE) (doc2repo, cross-repo, dep-migrate, domain-fix) and [ScaleSWE](https://github.com/AweAI-Team/ScaleSWE) with pluggable LLM backends, Docker-isolated execution, and optional web-search-augmented agents.

## Features

- **Multi-benchmark support** — BeyondSWE (4 task types) and ScaleSWE out of the box
- **Pluggable LLM backends** — OpenAI, Azure OpenAI, Volcengine Ark, ...
- **Docker isolation** — each instance runs in its own container with resource limits
- **Search-augmented agents** — optional web search + link summary tools
- **Batch execution** — concurrent evaluation with configurable parallelism
- **Plugin architecture** — LLM backends, runtimes, agents, evaluators, tools are all extensible via entry-points

## Architecture

```
awe_agent/
  core/              # Framework internals
    agent/           #   Agent loop, context, trajectory
    condenser/       #   Context window management
    config/          #   YAML config loading & schema
    eval/            #   Evaluation (PatchTestEvaluator, isolation)
    llm/             #   LLM backends + tool-call formatting
    runtime/         #   Container runtimes (Docker)
    task/            #   Task protocol, batch runner
    tool/            #   Tool registry (bash, editor, search, think, finish)
  scaffold/          # Agent implementations
    search_swe/      #   SearchSWE agent with optional web search
  tasks/             # Benchmark-specific task & evaluator
    beyond_swe/      #   BeyondSWE
    scale_swe/       #   ScaleSWE

configs/             # YAML configurations (LLM, task, runtime)
recipes/             # Reproducible entry points
  beyond_swe/        #   BeyondSWE runner (prompt / debug / batch / dry-run)
  scale_swe/         #   ScaleSWE runner
```

## Prerequisites

- **Python >= 3.11**
- **Docker** — each benchmark instance runs in an isolated container
- **LLM API access** — OpenAI, Azure OpenAI, or another supported backend

## Installation

### uv (Recommended)

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/AweAI-Team/AweAgent.git && cd AweAgent
uv venv --python 3.11
uv pip install -e ".[dev]"

# Verify
uv run awe-agent info
```

### pip

```bash
git clone https://github.com/AweAI-Team/AweAgent.git && cd AweAgent
pip install -e ".[dev]"

# Verify
awe-agent info
```

> **Why editable install?** AweAgent uses entry-points for plugin discovery. Without `pip install -e .`, the plugin registry cannot find LLM backends, agents, or tools and will raise `KeyError`.

## Quick Start

### 1. Configure LLM

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Or Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

### 2. Run BeyondSWE

```bash
# List all instances (no Docker needed)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl --mode dry-run

# Inspect prompt for one instance (no Docker needed)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --instance-id <INSTANCE_ID> --mode prompt

# Debug a single instance (full agent + eval trace)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl \
    --instance-id <INSTANCE_ID> --mode debug \
    --model gpt-4o --max-steps 30 --verbose

# Batch run (search-enabled, default)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl --mode batch

# Batch run (search-disabled, OpenHands style)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl --mode batch --no-search
```

### 3. Run ScaleSWE

```bash
# List all instances
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl --mode dry-run

# Debug a single instance
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl \
    --instance-id <INSTANCE_ID> --mode debug --verbose

# Batch run
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl --mode batch
```

### 4. Using the CLI Directly

```bash
export DATA_FILE=/path/to/data.jsonl

# BeyondSWE
awe-agent run -c configs/tasks/beyondswe_searchswe.yaml
awe-agent run -c configs/tasks/beyondswe_searchswe.yaml --dry-run

# ScaleSWE
awe-agent run -c configs/tasks/scale_swe.yaml
```

## Data Format

### BeyondSWE JSONL

Each line in the JSONL file represents one task instance. BeyondSWE supports 4 task types: `doc2repo`, `crossrepo`, `depmigrate`, `domainfix`.

```jsonc
{
  "instance_id": "pylons_plaster_pastedeploy_pr14",
  "task": "crossrepo",                          // doc2repo | crossrepo | depmigrate | domainfix
  "repo": "pylons/plaster_pastedeploy",
  "image": "beyondswe/crossrepo:pylons_plaster", // Docker image (pre-built)
  "workdir": "/workspace",
  "base_commit": "abc1234...",
  "problem_statement": "Description of the issue to solve...",
  "pre_commands": "cd /workspace && pip install -e .",

  // Evaluation fields
  "f2p_patch": "diff --git ...",                 // Patch introducing failing tests
  "f2p_script": "import pytest\n...",            // Test script content
  "FAIL_TO_PASS": "[\"test_foo\", \"test_bar\"]", // Tests that should pass after fix
  "PASS_TO_PASS": "[\"test_existing\"]",          // Tests that must not regress

  // Doc2Repo specific (only for task=doc2repo)
  "REPO_DOCUMENT_CONTENT": "# Specification\n...", // Repo specification document
  "test_suite": "test_suite_name.zip",             // Test suite archive
  "test_suite_num": 42                             // Expected number of tests
}
```

**Task type details:**

| Task Type | Description | Key Fields |
|-----------|-------------|------------|
| `doc2repo` | Build a repository from a specification document | `REPO_DOCUMENT_CONTENT`, `test_suite`, `test_suite_num` |
| `crossrepo` | Fix issues spanning multiple files/modules | `problem_statement`, `f2p_patch`, `f2p_script` |
| `depmigrate` | Dependency migration tasks | `problem_statement`, `f2p_patch`, `f2p_script` |
| `domainfix` | Domain-specific technical problems | `problem_statement`, `f2p_patch`, `f2p_script` |

> **Doc2Repo evaluation** requires a local directory containing test suite ZIP files. Set `BEYONDSWE_TEST_SUITE_DIR` or configure `test_suite_dir` in the YAML config.

### ScaleSWE JSONL

ScaleSWE follows a SWE-bench-style format with a single task type (issue-resolving).

```jsonc
{
  "instance_id": "django__django-12345",
  "user": "django",                             // GitHub user/org
  "repo": "django",                             // Repository name
  "image_url": "scaleswe/django:12345",          // Docker image (pre-built)
  "workdir": "/testbed",
  "parent_commit": "def5678...",                  // Base commit to work from
  "problem_statement": "Description of the issue...",
  "pre_commands": "cd /testbed && git checkout def5678...",

  // Evaluation fields
  "f2p_patch": "diff --git ...",
  "f2p_script": "import pytest\n...",
  "FAIL_TO_PASS": "[\"test_foo\"]",
  "PASS_TO_PASS": "[\"test_bar\"]"
}
```

**Key differences from BeyondSWE:**

| Field | BeyondSWE | ScaleSWE |
|-------|-----------|----------|
| Base commit | `base_commit` | `parent_commit` |
| Docker image | `image` | `image_url` |
| Repository | `repo` (full name) | `user` + `repo` (combined) |
| Default workdir | `/workspace` | `/testbed` |
| Task types | 4 types | Single (issue-resolving) |
| Search mode | Configurable | Always disabled |

### Docker Images

Both benchmarks require pre-built Docker images for each instance. The image name is specified in the JSONL data (`image` or `image_url` field). Images must be available locally before running.

```bash
# If images are in a remote registry
docker pull <image:tag>

# If network-restricted, download + load manually
crane pull <image:tag> /tmp/img.tar
docker load -i /tmp/img.tar
```

## Configuration

Configs are YAML files with environment variable substitution (`${VAR}`, `${VAR:-default}`) and `!include` support.

### BeyondSWE Config

```yaml
# configs/tasks/beyondswe_searchswe.yaml
llm: "!include ../llm/openai.yaml"

runtime:
  backend: docker
  timeout: 14400
  resource_limits:
    cpu: "4"
    memory: "8Gi"

agent:
  type: search_swe
  max_steps: 200
  enable_search: true         # Set false for OpenHands-style (no search)
  bash_timeout: 1200

task:
  type: beyond_swe
  dataset_id: beyond_swe
  data_file: ${DATA_FILE}
  # test_suite_dir: /path/to/doc2repo_test_suite  # or set BEYONDSWE_TEST_SUITE_DIR

eval:
  enabled: true
  timeout: 3600

execution:
  max_concurrent: 50
  output_path: ./results/beyondswe_searchswe
```

### ScaleSWE Config

```yaml
# configs/tasks/scale_swe.yaml
llm: "!include ../llm/openai.yaml"

agent:
  type: search_swe
  max_steps: 200
  enable_search: false
  bash_timeout: 1200
  tool_call_format: codeact_xml    # Uses CodeAct XML format

task:
  type: scale_swe
  dataset_id: scale_swe
  data_file: ${DATA_FILE}

execution:
  max_concurrent: 50
  output_path: ./results/scale_swe
```

### LLM Backends

| Backend | Config File | Required Env Vars |
|---------|-------------|-------------------|
| OpenAI | `configs/llm/openai.yaml` | `OPENAI_API_KEY` |
| Azure OpenAI | `configs/llm/azure.yaml` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Volcengine Ark | `configs/llm/ark.yaml` | `ARK_API_KEY`, `ARK_MODEL_ID` |
| SGLang | `configs/llm/sglang.yaml` | (self-hosted endpoint) |

### Search Mode (BeyondSWE only)

When `enable_search: true`, the agent gains access to web search tools. Requires additional environment variables:

```bash
# Search backend (SerpAPI)
export SERPAPI_API_KEY="your-serpapi-key"

# Reader backend (Jina)
export JINA_API_KEY="your-jina-key"

# Link summary LLM (optional)
export LINK_SUMMARY_CONFIG_PATH="configs/llm/link_summary/azure.yaml"
export LINK_SUMMARY_MODEL="gpt-4o-mini"
```

## Modes

Both `recipes/beyond_swe/run.py` and `recipes/scale_swe/run.py` support the same 4 modes:

| Mode | Description | Docker Required |
|------|-------------|:-:|
| `dry-run` | List all instances from the JSONL file | No |
| `prompt` | Print the agent prompt and task_info for one instance | No |
| `debug` | Full single-instance agent run with step-by-step trace | Yes |
| `batch` | Concurrent batch execution, outputs JSONL results | Yes |

### CLI Arguments (recipes)

```
--data-file PATH            JSONL data file (required)
--config / -c PATH          Config file (see defaults per recipe)
--mode MODE                 prompt | debug | batch | dry-run (default: prompt)
--instance-id ID            Single instance (prompt / debug)
--instance-ids ID [ID ...]  Multiple instances (batch, optional filter)
--model MODEL               Override LLM model
--max-steps N               Override max agent steps
--max-concurrent N          Override concurrency (batch)
--output DIR                Override output directory
--enable-search             Force-enable search tools (BeyondSWE only)
--no-search                 Force-disable search tools (BeyondSWE only)
--skip-eval                 Skip evaluation after agent run
--no-trajectories           Don't save per-instance trajectory files
--verbose                   DEBUG level logging
```

## Output

Batch results are saved to the output directory:

```
results/<run_id>/
  results.jsonl              # One line per instance
  config.json                # Config snapshot for reproducibility
  trajectories/              # Per-instance agent traces (optional)
    <instance_id>.json
```

### results.jsonl format

Each line contains:

```jsonc
{
  "instance_id": "django__django-12345",
  "success": true,
  "score": 1.0,               // 1.0 = all tests pass, 0.0 = fail
  "finish_reason": "success",  // success | max_steps | error | context_overflow
  "error": null,
  "duration": 123.4,
  "patch": "diff --git ..."    // Agent-generated patch
}
```

### Trajectory files

Each trajectory JSON contains the full step-by-step agent trace:

```jsonc
{
  "instance_id": "...",
  "trajectory": [
    {
      "step": 1,
      "action": {
        "type": "tool_call",
        "content": "Let me explore the repository structure...",
        "tool_calls": [{"name": "execute_bash", "arguments": {"command": "ls -la"}}]
      },
      "observations": ["total 64\ndrwxr-xr-x ..."]
    }
    // ...
  ],
  "eval_result": {
    "accepted": true,
    "score": 1.0,
    "details": { "f2p_count": 3, "p2p_count": 10 }
  }
}
```

## Development

```bash
# Run tests
pytest

# Lint & format
ruff check awe_agent/
ruff format awe_agent/

# Type check
mypy awe_agent/
```

## License

Apache-2.0
