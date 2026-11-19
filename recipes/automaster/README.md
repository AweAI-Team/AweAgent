# AutoMaster checkpoint inference

These portable examples run a Qwen3.5-based checkpoint from a Hugging Face repo
or local directory, without the autonomous training controller or generated skills.
They use public AweAgent components only: CalibForge (terminal), FoldSearch
(BrowseComp), and SearchSWE with search disabled (Doc2Repo).

## Install and serve

Install AweAgent from the specific commit supplied with your model release:

```bash
python -m pip install -e '.[search]'
```

Run SGLang in a **separate serving environment** with a Qwen3.5-compatible version
of SGLang, PyTorch and CUDA. Use the serving version provided by your model release;
this recipe does not install or validate a GPU stack. Keep the checkpoint's
`tokenizer_config.json` and `chat_template.jinja` alongside its weights.

```bash
export MODEL_PATH='your-org/your-model'  # or /path/to/downloaded/checkpoint
export SERVED_MODEL_NAME=automaster
bash recipes/automaster/serve_qwen35.sh
```

For multiple GPUs, set `TP` to the desired tensor-parallel size. The default bind
address is loopback; set `HOST` explicitly if serving a remote client. Pass additional
SGLang options as script arguments. HF authentication, if needed, uses the user's
standard Hugging Face credentials; no credentials belong in this repository.

In the evaluation environment, set `MODEL_BASE_URL` if the server is not at
`http://127.0.0.1:30000/v1`. The configs request `enable_history_reasoning=true`:
confirm that the model's template supports this setting so a follow-up user
reminder does not silently remove earlier reasoning.

## Run a task, then a benchmark

| Domain | Required assets | Example config |
|---|---|---|
| Terminal | `TB_TASK_DIR`: extracted task directories; `TB_TASK_IDS`: JSON array of selected task IDs | `terminal.yaml` |
| BrowseComp | `BROWSECOMP_DATA_FILE`: CSV, JSON or JSONL questions; `MODEL_PATH`: matching tokenizer | `browsecomp.yaml` |
| Doc2Repo | `DOC2REPO_DATA_FILE`: Doc2Repo-only BeyondSWE JSONL; `DOC2REPO_TEST_SUITE_DIR`: test-suite ZIP directory | `doc2repo.yaml` |

See the [terminal](../terminal_bench_v2/README.md),
[BrowseComp](../deepsearch/README.md), and [BeyondSWE](../beyond_swe/README.md)
recipes for dataset preparation. Code tasks require Docker and accessible task
images. BrowseComp does not create a sandbox container.

For example, after setting the domain's required variables:

```bash
# Inspect the selected task before sending any model requests.
awe-agent run --config recipes/automaster/doc2repo.yaml --max-instances 1 --dry-run
# Run one real task and its evaluator.
awe-agent run --config recipes/automaster/doc2repo.yaml --max-instances 1
# Run the supplied set three times. Concurrency is a resource setting.
awe-agent run --config recipes/automaster/doc2repo.yaml \
  --num-rollouts 3 --max-concurrent 4 --output ./results/doc2repo
```

Replace the config filename to run another domain. Results and trajectories are
written under the output directory. Preserve all attempts and failures when
computing scores; do not select a repeat based on the resulting score.

### BrowseComp services

Set `SERPAPI_API_KEY` and `JINA_API_KEY` for the public search and reader backends.
Set `SUMMARY_MODEL`, `SUMMARY_BASE_URL`, `SUMMARY_API_KEY` and the analogous
`JUDGE_*` variables for OpenAI-compatible summary and judging services. These are
separate from the evaluated model. For inference without grading, set
`AWE_AGENT__EVAL__ENABLED=false`; summary and search services are still required.

FoldSearch exposes `search_api`, `link_summary_tool` and `finish` to the model,
while dispatching to AweAgent's public search/read tools. It folds the oldest tool
observations in the **model input**, retains the full saved interaction history,
and reserves room for a final answer. No generated AutoMaster skills are injected.

## Scope of these examples

These are portable starting configurations, **not frozen paper reproduction
manifests**. They start with one concurrent task and no whole-task retry. Terminal
uses a 3-hour agent budget; Doc2Repo uses 4 hours, separate from verifier time;
BrowseComp uses context/step budgets. The examples deliberately do not depend on
private sandboxes, search services, account routing or a remote scheduler.

Replacing search providers, dataset revisions, images, sampling parameters,
retry policies or verifier implementations can change scores. Use model-release
manifests to reproduce a specific result, including the exact task IDs, tokenizer,
scaffold commit, model revision and three evaluation records. A successful config
load or unit test is not an end-to-end GPU/benchmark reproduction.
