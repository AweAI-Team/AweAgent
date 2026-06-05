# SWE-bench Verified Reproduction with Scale-SWE-Agent

This note documents a local run that reproduces the published **Scale-SWE-Agent** result on SWE-bench Verified ([paper](https://arxiv.org/abs/2602.09892)) with AweAgent inference and the public SWE-bench harness.

The paper reports **64%** and is stably reproducible in our full setup. The materials here reach **61.4%** (`307 / 500`) — slightly below, most likely because a few setup details are not yet fully aligned in this public pipeline — and are provided as a runnable reference.

> The `61.4%` depends on **evaluation-side compatibility patches** to the public `swebench==4.1.0` harness (documented below) and on **runtime/prompt settings not included in this PR**, so it is not the score produced by unmodified upstream `main` or the stock official harness alone.

This document and the helper scripts are intentionally scoped as reproduction materials. They do not change AweAgent's default runtime behavior. Runtime,
prompt, and tool-call behavior changes used during the run should be reviewed separately from these reproduction notes.

This directory lives under `recipes/scale_swe/` because the run uses the existing AweAgent ScaleSWE recipe with a converted SWE-bench Verified dataset.

## Scope

Included here:

- dataset conversion from SWE-bench Verified rows to the AweAgent ScaleSWE JSONL shape;
- conversion from AweAgent trajectories to SWE-bench predictions;
- prediction cleanup for non-submission artifacts generated during agent exploration;
- two evaluation-side compatibility patches for the installed `swebench==4.1.0` package;
- a compact result summary in `recipes/scale_swe/swebench_verified/artifacts/scale_swe_agent_61_4_summary.json`.

Not included here:

- full trajectories, predictions, Docker logs, or SWE-bench run logs;
- changes to AweAgent's default prompt, runtime, LLM backend, tools, or config;
- benchmark result JSON files containing the full resolved and unresolved instance lists.

## Result

The reproduced run used `AweAI-Team/Scale-SWE-Agent` with temperature `1.0`, max turns `200`, and public SWE-bench harness evaluation after applying the
compatibility patches documented below.

```text
dataset:             SWE-bench/SWE-bench_Verified
split:               test
total_instances:     500
submitted_instances: 500
completed_instances: 498
resolved_instances:  307
unresolved_instances:191
empty_patch_instances: 2
error_instances:     0

score: 307 / 500 = 61.4%
```

The model was sampled at temperature `1.0`, so rerunning inference from scratch may produce a slightly different score. To audit the exact score above,
use the saved trajectories and cleaned predictions from the corresponding run, then rerun the same SWE-bench harness command with the same evaluation-side
patches.

## Environment

The run used Python 3.11 with the project `uv` environment. Relevant package
versions recorded during the run:

```text
awe-agent==0.1.0
swebench==4.1.0
docker==7.1.0
openai==1.109.1
datasets==4.8.5
pydantic==2.12.5
tenacity==9.1.4
httpx==0.28.1
```

> **`swebench` is not an AweAgent dependency** (we don't add it to the default install). Install the exact harness version these scripts target into the same environment:
>
> ```bash
> uv pip install swebench==4.1.0
> ```
>
> The compatibility patches below are written against `swebench==4.1.0`; on other versions their patch anchors won't match and the patch scripts exit with an error.

The vLLM service environment was recorded as:

```text
vllm==0.19.0
transformers==5.2.0
```

## Serve the Model

Example vLLM launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve AweAI-Team/Scale-SWE-Agent \
  --served-model-name Scale-SWE-Agent \
  --host 127.0.0.1 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 65536 \
  --enable-prefix-caching
```

AweAgent then talks to the service through the OpenAI-compatible API:

```text
OPENAI_BASE_URL=http://127.0.0.1:8001/v1
OPENAI_API_KEY=local-key
AWE_AGENT_MODEL=Scale-SWE-Agent
```

## Convert SWE-bench Verified

Convert the SWE-bench Verified test split into the JSONL shape consumed by the AweAgent ScaleSWE recipe:

```bash
HF_HOME=/tmp/hf-home \
HF_DATASETS_CACHE=/tmp/hf-datasets \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python recipes/scale_swe/swebench_verified/convert_swebench_verified_to_scaleswe.py \
  --dataset-name SWE-bench/SWE-bench_Verified \
  --split test \
  --output data/local/swebench_verified_scaleswe_full500.jsonl \
  --limit 500 \
  --no-require-local-image
```

The converter preserves fields needed by AweAgent inference and later SWE-bench harness evaluation, including `instance_id`, `repo`, `base_commit`,
`problem_statement`, `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS`, and the SWE-bench Docker image key.

## Run AweAgent Inference

Example command used for the reproduction run:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8001/v1 \
OPENAI_API_KEY=local-key \
AWE_AGENT_MODEL=Scale-SWE-Agent \
AWE_AGENT__LLM__MODEL=Scale-SWE-Agent \
AWE_AGENT__LLM__PARAMS__TEMPERATURE=1.0 \
AWE_AGENT__LLM__PARAMS__MAX_TOKENS=16384 \
AWE_AGENT__AGENT__MAX_CONTEXT_LENGTH=220000 \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python recipes/scale_swe/run.py \
  --data-file data/local/swebench_verified_scaleswe_full500.jsonl \
  --mode batch \
  --model Scale-SWE-Agent \
  --max-steps 200 \
  --max-concurrent 24 \
  --output results/scaleswe_verified_full500_61p4 \
  --skip-eval
```

The expected inference output is:

```text
results/scaleswe_verified_full500_61p4/Scale-SWE-Agent_<timestamp>/trajectories.jsonl
```

The final score should be computed by converting trajectories to SWE-bench predictions and running the SWE-bench harness.

Note: the reproduction run used local runtime changes that made `--skip-eval` skip the repo-local evaluator during batch inference. This PR does not include
that runtime change. On unmodified upstream `main`, verify the current `--skip-eval` behavior before using this command for a full inference-only batch run.

## Convert and Clean Predictions

Convert AweAgent trajectories to SWE-bench predictions:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python recipes/scale_swe/swebench_verified/aweagent_to_swebench_predictions.py \
  --trajectories results/scaleswe_verified_full500_61p4/Scale-SWE-Agent_<timestamp>/trajectories.jsonl \
  --output results/scaleswe_verified_full500_61p4/predictions.jsonl \
  --model-name Scale-SWE-Agent
```

Clean non-submission artifacts from the generated patches:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run python recipes/scale_swe/swebench_verified/clean_swebench_predictions.py \
  --input results/scaleswe_verified_full500_61p4/predictions.jsonl \
  --output results/scaleswe_verified_full500_61p4/predictions_benchmark_v2_rerun.jsonl \
  --preserve-nonempty-fallback
```

The cleaner removes obvious exploration artifacts from predictions, such as auto-generated `.gitignore` blocks, bytecode files, and newly created temporary
`debug`, `repro`, `verify`, or test helper scripts. It does not alter the underlying task data or SWE-bench grading logic.

The recorded cleanup summary for the `61.4%` run was:

```json
{
  "changed_rows": 491,
  "dropped_blocks": {
    "agent_gitignore": 498,
    "new_temp_script": 2283,
    "preserved_nonempty_fallback": 7
  },
  "empty_after": 2,
  "empty_before": 2,
  "rows": 500
}
```

## Patch the Installed SWE-bench Harness

Before evaluation, apply the two compatibility patches to the installed `swebench==4.1.0` package in the active environment:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python recipes/scale_swe/swebench_verified/patch_swebench_pytest_parser.py
UV_CACHE_DIR=/tmp/uv-cache uv run python recipes/scale_swe/swebench_verified/patch_swebench_container_resources.py
```

These patches are evaluation-side compatibility patches. They do not change model predictions, SWE-bench task data, image contents, test commands, or ground truth patches.

`patch_swebench_pytest_parser.py` handles narrow log parsing cases observed with the public harness:

- pytest logs that contain dot output plus a passed summary but no explicit test-name status map;
- expected empty parametrization IDs such as `test_x[]` when pytest emits concrete IDs such as `test_x[param]`;
- Sympy logs where the output marker split the expected test name and trailing `ok` line.

`patch_swebench_container_resources.py` forwards existing SWE-bench Docker resource specs, such as `nano_cpus`, into `docker.containers.create()`. This was
needed for Pylint cases whose tests depend on the container seeing at least two available CPUs.

Recreate or reinstall the environment before comparing against an unpatched public harness, because these scripts modify the installed package files in the
current environment.

## Run SWE-bench Harness Evaluation

Evaluate the cleaned predictions with the public SWE-bench harness after applying the compatibility patches above:

```bash
HF_HOME=/tmp/hf-home \
HF_DATASETS_CACHE=/tmp/hf-datasets \
UV_CACHE_DIR=/tmp/uv-cache \
uv run python -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --split test \
  --predictions_path results/scaleswe_verified_full500_61p4/predictions_benchmark_v2_rerun.jsonl \
  --max_workers 96 \
  --open_file_limit 65536 \
  --run_id scaleswe_verified_full500_61p4 \
  --cache_level instance \
  --clean False
```

The recorded `61.4%` result corresponds to:

```text
run_id: scaleswe_C_userobs_sortedview_official_verified_20260518
score:  307 / 500 = 61.4%
```

## Known Caveats

- Inference used `temperature=1.0`; exact reruns are expected to have sampling variance.
- The `61.4%` score depends on the documented evaluation-side SWE-bench compatibility patches.
- Sphinx `linkcheck` cases can be sensitive to external network availability; this reproduction did not add a code patch for those cases.
- This PR does not include the runtime, prompt, and tool-call behavior changes used during the run. Those changes should be reviewed in a separate PR if
  maintainers want to incorporate them into AweAgent itself.
