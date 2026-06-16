# DeepSearch

DeepSearch is AweAgent's search-oriented QA scaffold. It lets an agent search,
read web pages, think, and submit a short final answer with `finish(answer=...)`.
It supports configurable context folding, fresh rollout retries, and forced
answer extraction when the agent reaches the step limit without submitting.

Current result:

| Model | BrowseComp |
|-------|------------|
| `deepseek-v4-pro` | **64.5** |

<sub>Official reported score: 83.4. Since the harness and search/fetch backends
may differ, treat it as a reference point. We are continuing to improve this
setup.</sub>

## Setup

Install AweAgent in editable mode:

```bash
cd /path/to/AweAgent
uv pip install -e ".[dev]"
```

Configure an OpenAI-compatible LLM:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o"
```

Configure search/read backends. The public built-ins are `serpapi` for search
and `jina` for page reading:

```bash
export SEARCH_BACKEND="serpapi"
export READER_BACKEND="jina"
export SERPAPI_API_KEY="your-serpapi-key"
export JINA_API_KEY="your-jina-key"  # optional, higher rate limits
```

## Data

BrowseComp-style data can be JSON, JSONL, CSV, or Parquet. A common JSON shape:

```json
[
  {
    "id": "test_001",
    "problem": "Question text",
    "answer": "Reference answer"
  }
]
```

The loader also accepts `question`, `correct_answer`, and SFT-style fields such
as `prompt`, `reward_model.ground_truth`, and `extra_info.question/answer/id`.

## Run

Dry run:

```bash
bash recipes/deepsearch/run_deepsearch_browsecomp.sh \
  --data-file /path/to/browsecomp.json \
  --dry-run
```

Run one instance:

```bash
bash recipes/deepsearch/run_deepsearch_browsecomp.sh \
  --data-file /path/to/browsecomp.json \
  --instance-ids test_001 \
  --model gpt-4o \
  --max-steps 100 \
  --max-concurrent 1
```

Batch run:

```bash
bash recipes/deepsearch/run_deepsearch_browsecomp.sh \
  --data-file /path/to/browsecomp.json \
  --model gpt-4o \
  --max-steps 300 \
  --rollout-retries 0 \
  --max-concurrent 4 \
  --output-dir results/deepsearch_browsecomp
```

Direct CLI (Recommended):

```bash
export BROWSECOMP_DATA_FILE=/path/to/browsecomp.json

python -m aweagent.cli run \
  --config configs/tasks/browsecomp.yaml \
  --instance-ids test_001 \
  --max-steps 300 \
  --max-concurrent 1 \
  --output results/deepsearch_browsecomp
```

## Config

Default config: [configs/tasks/browsecomp.yaml](../../configs/tasks/browsecomp.yaml).

Key fields:

DeepSearch runs no shell commands, so the task opens no container runtime.

```yaml
agent:
  type: deepsearch
  max_steps: 100
  rollout_retries: 0
  force_final_answer: true
  tools: [web_search, web_fetch, finish]
  tool_options:
    web_search:
      backend: serpapi
      engine: google
      max_attempts: 3
    web_fetch:
      reader_backend: jina
      max_attempts: 3
      reader_max_attempts: 3
  condenser:
    type: tool_result_omission
    keep_recent_tool_results: 5

task:
  type: browsecomp
  data_file: ${BROWSECOMP_DATA_FILE}

execution:
  max_concurrent: 50
  max_retries: 3
  output_path: ./results/browsecomp
```

Useful knobs:

| Field | Meaning |
|-------|---------|
| `agent.max_steps` | Max steps in one rollout |
| `agent.rollout_retries` | Extra fresh rollouts after the first one |
| `agent.force_final_answer` | Extract a best-effort answer if no `finish` is called |
| `agent.condenser.keep_recent_tool_results` | Keep recent tool outputs; `-1` means full context |
| `execution.max_concurrent` | Parallel instances |

For full context:

```bash
bash recipes/deepsearch/run_deepsearch_browsecomp.sh \
  --data-file /path/to/browsecomp.json \
  --full-context
```

## Evaluation

BrowseComp evaluation reads `metadata["final_answer"]` and uses a judge model to
compare it with the reference answer. By default, the judge reuses the main LLM.
You can set a separate judge model:

```yaml
eval:
  enabled: true
  judge_llm:
    backend: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    model: gpt-4o
    params:
      temperature: 0.0
      max_tokens: 1024
```

## Output

Each run writes:

```text
results/deepsearch_browsecomp/
  <model>_<timestamp>/
    results.jsonl
    trajectories.jsonl
    run_config.json
```

`results.jsonl` has one summary row per instance. `trajectories.jsonl` stores
messages, tool calls, evaluation details, and DeepSearch metadata such as:

```json
{
  "final_answer": "Example Answer",
  "agent_submitted_final_answer": true,
  "forced_final_answer": false
}
```

## Tips

- Start with `--dry-run`, then test one `--instance-ids` example.
- Use `--max-concurrent 1` while debugging.
- Use `keep_recent_tool_results: 5` for lower token cost.
- Use `--full-context` only when you need complete history for analysis.
