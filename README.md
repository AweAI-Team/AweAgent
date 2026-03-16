<div align="center">

# AweAgent

**A general-purpose agent framework with pluggable scaffolds and reproducible evaluation.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
<!-- [![GitHub stars](https://img.shields.io/github/stars/AweAI-Team/AweAgent.svg?style=social)](https://github.com/AweAI-Team/AweAgent) -->

</div>

AweAgent provides two core capabilities:

- **Pluggable Agent Scaffolds** — Modular agent loop with extensible tools (bash, editor, search, think), pluggable LLM backends (OpenAI, Azure, Ark, SGLang), and configurable context management.
- **Reproducible Evaluation** — Docker-isolated execution, built-in evaluators, batch runner with concurrent execution, and structured result / trajectory output.

AweAgent currently ships with [ScaleSWE](https://github.com/AweAI-Team/ScaleSWE) (training data), [BeyondSWE](https://github.com/AweAI-Team/BeyondSWE), and [Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2) benchmarks support.

## :newspaper: News

- `[2026-03-15]` 🎉 Terminus-2 agent scaffold with [Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2) benchmark support
- `[2026-03-11]` 🎉 Sync codebase with internal version for BeyondSWE + SearchSWE
- `[2026-03-01]` 🎉 Initial release — SearchSWE agent scaffold with [BeyondSWE](https://github.com/AweAI-Team/BeyondSWE) & [ScaleSWE](https://github.com/AweAI-Team/ScaleSWE) support

## :building_construction: Architecture

```
awe_agent/
  core/              # Framework internals
    agent/           #   Agent loop, context, trajectory, protocol
    condenser/       #   Context window management
    config/          #   YAML config loading & schema
    eval/            #   Evaluation (PatchTestEvaluator, isolation)
    llm/             #   LLM backends + tool-call formatting
      format/        #     Three modes: openai_function, codeact_xml, terminus_json
    runtime/         #   Container runtimes (Docker)
    task/            #   Task protocol, TaskRunner (unified batch engine)
    tool/            #   Tool registry (bash, editor, search, think, finish)
  scaffold/          # Agent implementations
    search_swe/      #   SearchSWE agent with optional web search
    terminus_2/      #   Terminus-2 agent (tmux + JSON keystrokes)
  tasks/             # Benchmark-specific task & evaluator
    beyond_swe/      #   BeyondSWE
    scale_swe/       #   ScaleSWE
    terminal_bench_v2/  #   Terminal Bench 2.0

configs/             # YAML configurations (LLM, task, runtime)
recipes/             # Reproducible entry points
  beyond_swe/        #   BeyondSWE runner
  scale_swe/         #   ScaleSWE runner
  terminal_bench_v2/ #   Terminal Bench 2.0 runner
```

## :jigsaw: Supported Scaffolds

### SearchSWE

The built-in **SearchSWE** agent scaffold (`awe_agent/scaffold/search_swe/`) is a modular agent loop that can operate in two modes — switch between them with a single config flag:

| Mode | `enable_search` | `tool_call_format` | Description |
|------|:---:|---|---|
| **SearchSWE** | `true` | `openai_function` | Full tool set including web search & link summary |
| **OpenHands-style** | `false` | `codeact_xml` | CodeAct XML format, compatible with OpenHands agent behavior |

**Tool Blocks.** The agent composes its tool set from independent, self-contained tool blocks. Each block implements a unified `Tool` protocol (name, JSON Schema parameters, async execute) and is registered via a plugin registry with entry-point discovery:

| Tool | Name | Description |
|------|------|-------------|
| Bash | `execute_bash` | Persistent shell session inside Docker with output truncation, timeout control, and regex-based command blocklist |
| Editor | `str_replace_editor` | File viewer/editor with `view`, `create`, `str_replace`, and `insert` sub-commands |
| Search | `search` | Web search with anti-leak filtering (auto-blocks target repo URLs). Only active when `enable_search: true` |
| Link Summary | `link_summary` | Fetch a URL and summarize content via a dedicated LLM. Only active when `enable_search: true` |
| Think | `think` | Reasoning scratchpad — no environment side-effects, helps the agent plan before acting |
| Finish | `finish` | Signals task completion and triggers evaluation |

Adding a custom tool is as simple as implementing the `Tool` protocol and registering it via a Python entry-point — no changes to the agent loop required.

### Terminus-2

The **Terminus-2** agent scaffold (`awe_agent/scaffold/terminus_2/`) is a terminal agent that uses tmux for persistent shell sessions. The LLM outputs raw JSON with keystrokes; the framework translates this into actions via `TerminusJSONFormat` (the third `ToolCallFormat` alongside `openai_function` and `codeact_xml`). This allows Terminus-2 to run inside the standard `AgentLoop`, inheriting RL training, context condensing, stats tracking, and step callbacks — without changing the LLM-facing prompt.

Designed for [Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2) evaluation, aligned with the official Harbor framework. Evaluation runs in the **same container** as the agent (no patch — the agent modifies container state directly).

### Evaluation

**SWE-bench style (BeyondSWE / ScaleSWE).** After the agent finishes, evaluation runs in a **separate Docker container** to ensure a clean, tamper-proof environment:

1. Check out the base commit in a fresh container
2. Apply the agent-generated patch (6 auto-fallback strategies)
3. Restore original test files (prevents the agent from gaming tests)
4. Run fail-to-pass & pass-to-pass test suites via an injected pytest runner
5. Report structured results (score, pass/fail details, trajectory)

**Terminal Bench 2.0.** Evaluation runs in the **same container** — the evaluator uploads test scripts, executes `bash /tests/test.sh`, and reads the reward file. This is controlled by the `Evaluator.requires_same_session` protocol property.

## :rocket: Installation

### uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/AweAI-Team/AweAgent.git && cd AweAgent
uv venv --python 3.11
uv pip install -e ".[dev]"
```

### pip

```bash
git clone https://github.com/AweAI-Team/AweAgent.git && cd AweAgent
pip install -e ".[dev]"
```

> **Why editable install?** AweAgent uses entry-points for plugin discovery. Without `-e`, the plugin registry cannot find LLM backends, agents, or tools.

## :clipboard: Supported Benchmarks

| Benchmark | Description | Agent | Dataset | Guide |
|-----------|-------------|-------|---------|-------|
| BeyondSWE | Doc2Repo, CrossRepo, DepMigrate, DomainFix | SearchSWE (with web search) | [Hugging Face](https://huggingface.co/datasets/AweAI-Team/BeyondSWE) | [README](recipes/beyond_swe/) |
| ScaleSWE | Large-scale SWE-bench style training datasets (20k instances) | SearchSWE (CodeAct XML) | [Hugging Face](https://huggingface.co/datasets/AweAI-Team/Scale-SWE) | [README](recipes/scale_swe/) |
| Terminal Bench 2.0 | Terminal tasks in containerized environments | Terminus-2 | [GitHub](https://github.com/laude-institute/terminal-bench-2) (commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`) | [README](recipes/terminal_bench_v2/README.md) |

### Download Data

```bash
# BeyondSWE
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AweAI-Team/BeyondSWE",
    repo_type="dataset",
    local_dir="<your_path>/BeyondSWE",
)

# ScaleSWE — see the Hugging Face collection for available splits
# https://huggingface.co/datasets/AweAI-Team/Scale-SWE

# Terminal Bench 2.0 — clone and checkout commit for reproducibility
git clone https://github.com/laude-institute/terminal-bench-2.git
cd terminal-bench-2
git checkout 69671fbaac6d67a7ef0dfec016cc38a64ef7a77c
# Each task folder contains instruction.md, task.toml, environment/, tests/
```

### Quick Example

```bash
# Configure LLM
export OPENAI_API_KEY="sk-..."

# List instances (no Docker needed)
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl --mode dry-run

# Batch run
python recipes/beyond_swe/run.py \
    --data-file /path/to/beyondswe.jsonl --mode batch
```

See each benchmark's guide above for full setup, CLI arguments, and output format.

## :gear: Configuration

Configs are YAML files with environment variable substitution (`${VAR}`, `${VAR:-default}`) and `!include` support.

### LLM Backends

| Backend | Config File | Required Env Vars |
|---------|-------------|-------------------|
| OpenAI | `configs/llm/openai.yaml` | `OPENAI_API_KEY` |
| Azure OpenAI | `configs/llm/azure.yaml` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Volcengine Ark | `configs/llm/ark.yaml` | `ARK_API_KEY`, `ARK_MODEL_ID` |
| SGLang | `configs/llm/sglang.yaml` | (self-hosted endpoint) |

### Environment Variables

Copy [`.env.example`](.env.example) to `.env` and fill in your values:

```bash
cp .env.example .env
```

The `.env.example` is organized into three sections:

1. **LLM Backend** (pick one) — set the API key and endpoint for your chosen backend:
   ```bash
   # OpenAI
   OPENAI_API_KEY=sk-...

   # Or Azure OpenAI
   AZURE_OPENAI_API_KEY=your-key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   ```

2. **Task Data** — path to your benchmark JSONL file:
   ```bash
   DATA_FILE=/path/to/data.jsonl
   ```

3. **Search Tools** (optional, BeyondSWE search mode only) — required when running with `enable_search: true`:
   ```bash
   SERPAPI_API_KEY=your-serpapi-key
   JINA_API_KEY=your-jina-key          # optional, 20 RPM free without key
   ```

See each [benchmark guide](#clipboard-supported-benchmarks) for the full list of variables.

## :world_map: Roadmap

Our long-term goal is to build practical, general-purpose agents and optimize them with reinforcement learning.

**Agent Scaffolds & Capabilities**
- [x] SearchSWE — coding agent with optional web search augmentation
- [x] Terminus-2 — terminal agent for Terminal Bench 2.0
- [ ] deep research Agent

**Evaluation & Optimization**
- [x] BeyondSWE & ScaleSWE benchmark support
- [x] Terminal Bench 2.0 benchmark support
- [ ] More benchmarks — wider task coverage and domain diversity
- [ ] Agentic RL — scalable reinforcement learning infrastructure for agent optimization


## 📄 License
This project is released under the [Apache-2.0 License](LICENSE).

## 📨 Contact

For any questions or feedback, please reach out to us at `gx.chen.chn@gmail.com`.