# NL2Repo Task

Faithful AweAgent port of [NL2RepoBench](https://github.com/multimodal-art-projection/NL2RepoBench).

The agent receives a natural-language repository specification (`start.md`) and
must implement the **entire** Python project from scratch inside a per-project
docker sandbox that already has the project's dependencies pre-installed.

## Sandbox scheduling

Every NL2Repo instance is launched in its own per-project docker container,
read from each row's `evaluation_image` field (the same image
`post_processor.create_dockerfile` uses as `FROM`). The AweAgent `TaskRunner`
schedules one container per instance through `task.get_image(instance)`,
mirroring the original `openhands_app.start_app`'s `ThreadPoolExecutor` /
`max_pool_size` loop. Concurrency is controlled by `execution.max_concurrent`
in the YAML config.

## Quick start

```bash
# Via Python entry point — list instances
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --mode dry-run

# Inspect prompt without spinning up Docker
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --instance-id aiofiles --mode prompt

# Full debug run on one instance
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --instance-id aiofiles --mode debug

# Batch run
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --mode batch
```

## Data source

A single JSONL or CSV file is the authoritative source. Each row is one
instance and provides every field the benchmark needs:

| Field               | Internal mapping            | Purpose                                            |
|---------------------|-----------------------------|----------------------------------------------------|
| `evaluation_image`  | `metadata.eval_image`       | Per-instance image used by `NL2RepoEvaluator`.     |
| `instance_id`       | `Instance.id`               | Unique id / project name.                          |
| `package_name`      | `metadata.pro_name`         | Python package name (defaults to `instance_id`).   |
| `start_instruction` | `metadata.start_md`         | Uploaded to `/workspace/start.md`.                 |
| `test_cases_num`    | `metadata.test_case_count`  | Expected pytest case count (scoring denom).        |
| `verify_cmd`        | `metadata.test_shell`       | List — shell commands run for evaluation.          |
| `verify_files`      | `metadata.py_test_file_list`| List — golden test paths restored pre-eval.        |

Pass the data via `--data-file` or the `DATA_FILE` environment variable.
For callers that build rows programmatically, `instances=[{...}]` is also
accepted.

## Image selection

Two knobs:

- **`DATA_FILE`** (env) / **`data_file=`** (constructor) — points at the
  JSONL/CSV. Every row's `evaluation_image` becomes that instance's evaluation
  image.
- **`AGENT_RUN_DOCKER`** (env) / **`agent_run_docker=`** (constructor) —
  optional single image for the agent side. Empty string = agent shares the
  per-instance evaluation image.

## Evaluation

`NL2RepoEvaluator` consumes the whole agent workspace as a tarball, not a
git patch. It mirrors `openhands.post_processor.post_process_task`
step-by-step:

1. Spin up a fresh container of the per-instance `evaluation_image`.
2. Upload the agent workspace tarball and extract it into `/tmp/workspace`.
3. Strip package-management files (`setup.py`, `pyproject.toml`,
   `requirements*.txt`, ...) recursively from the staged workspace, matching
   `remove_package_files`.
4. Strip staged copies of golden test paths listed in `py_test_file_list`,
   matching `remove_test_files`; the base image's golden tests then survive
   the overlay.
5. Overlay `/tmp/workspace` onto `/workspace`, matching the official
   `COPY workspace /workspace` behavior.
6. Run the project's `test_shell` commands sequentially in `/workspace` with
   `PYTHONPATH=/workspace:$PYTHONPATH`.
7. Compute `pass_rate = min(passed / test_case_count, 1)` with the same loose
   pytest-output regex parser as `analyze_pytest_results`.

`count_mismatch` is reported in result details when pytest's parsed
`passed + failed + errors` differs from `test_case_count`, but it is only a
diagnostic. The official benchmark still scores with the count as the
denominator, so the evaluator does not surface `test_count_mismatch` as an
evaluation error.

For runner compatibility, `accepted` is `True` when at least the expected
number of tests pass and pytest reports no failures or errors.

## Full parameters

```python
NL2RepoTask(
    dataset_id="nl2repo",
    data_file=None,          # JSONL/CSV path; falls back to DATA_FILE env
    agent_run_docker=None,   # override image; falls back to AGENT_RUN_DOCKER
    instances=None,          # raw dicts alternative (programmatic)
    workdir="/workspace",    # matches NL2RepoBench config.template.toml
)
```
