# DeNovoSWE Task

From-scratch repository construction benchmark.  Similar to BeyondSWE
`doc2repo`, but the docker images still contain the original source — the
runtime cleans it with `clean.sh` before the agent gets a chance to look,
then asks the agent to re-implement the package from the spec in `README.md`.

## Differences vs. BeyondSWE doc2repo

| | BeyondSWE doc2repo | DeNovoSWE |
|---|---|---|
| Image | Source pre-cleaned | Source still present; `clean.sh` cleans before agent runs |
| Spec file | `repo_document.md` | `README.md` |
| Tests | Test-suite zip uploaded | `test_patch` extracted from image, applied during eval |
| Evaluation | `realswe_eval_script.py` | Delete agent tests → apply `test_patch` → per-file pytest |
| Known-broken tests | n/a | AST removes `failed_ptp` so one broken collection doesn't poison the file |

## Workflow

**Agent session** (`prepare_session` → agent loop):

1. `git checkout -f parent_commit`.
2. `clean.sh` — uninstall the target package across every interpreter,
   purge site-packages residue + pip caches, delete `.py/.pyx/.pxd/.pyi`
   sources, wipe test dirs/files, recreate `.git`, commit baseline.
3. Snapshot remaining files (optionally dumped to a JSONL via
   `--dump-clean-snapshot`).
4. Upload `README.md` (the spec) and fold it into the baseline commit so
   agent patches don't include it.
5. Fold the runtime `.gitignore` block into the baseline commit too.
6. `pip freeze` → `installed_packages` slot in the user prompt.
7. Agent builds the package from the spec.

**Evaluation session** (new container):

1. `git checkout -f parent_commit` and re-run `clean.sh`.
2. Re-inject `README.md` (some `setup.py` files read it at install time).
3. Apply the agent patch.
4. Nuke every test directory + agent-written verification scripts.
5. `mkdir -p` for directories `test_patch` will create, `rm -f` files it
   will newly add (so additive hunks don't reject against agent leftovers).
6. Apply `test_patch` (golden unit tests).
7. AST-delete every `failed_ptp` function/method from the test files.
8. Extract + apply the binary-fixture archive when present.
9. Uninstall the (possibly-cached) package, run `pip install -e .`.
10. Per-file pytest run; `pass_rate = passed / |passed_ptp|`.

`validate_run=True` skips agent steps 3-7 and step 3 in eval, so you can
verify the test patches work against the original source.

## Data format

Input JSONL (output of `recipes/denovo_swe/extract_patch.py`):

```jsonc
{
  "instance_id": "PyCQA_pep8_pr970",
  "workdir": "/workspace/pep8",
  "image": "...",
  "repo": "pep8",
  "parent_commit": "07b113bdb...",
  "document": "## 1. Overview ...",     // becomes README.md
  "passed_ptp": ["testsuite/test_all.py::TestCase::test_method", ...],
  "failed_ptp": ["testsuite/test_api.py::TestCase::test_nullbytes", ...],
  "test_patch": "diff --git a/testsuite/test_all.py ...",
  "test_files": ["testsuite/test_all.py", "testsuite/support.py", ...],
  "test_binary_archive_b64": "...",      // base64 tar.gz, "" if none
  "test_binary_files": [...],

  // Optional — anti-hack search constraints
  "pypi_name": "pep8",
  "pypi_name_candidates": [...],
  "import_names": ["pep8"]
}
```

## Quick Start

```bash
# Step 1: extract test patches from images (one-off preprocessing)
python recipes/denovo_swe/extract_patch.py \
    --input /path/to/ready_denovoswe.jsonl \
    --output ./results/denovoswe \
    --config configs/tasks/denovoswe.yaml

# Step 2: run agent + eval on a single instance (debug mode)
python recipes/denovo_swe/run.py \
    --data-file ./results/denovoswe/extract_patch_*/results.jsonl \
    --instance-id PyCQA_pep8_pr970 --mode debug --verbose

# Step 3: batch run
python recipes/denovo_swe/run.py \
    --data-file ./results/denovoswe/extract_patch_*/results.jsonl \
    --mode batch --max-concurrent 50
```

## Full Parameters

```python
DeNovoSWETask(
    dataset_id="denovo_swe",        # Dataset identifier
    data_file=None,                 # Path to JSONL data file
    instances=None,                 # Raw instance dicts (alternative)
    search_mode=False,              # Enable search tool prompts
    validate_run=False,             # Skip agent, run eval only
    del_done_images=False,          # docker rmi after each instance
    clean_snapshot_file=None,       # Dump post-clean workspace snapshots
    prompt_version="v2",            # "v1" or "v2" (default — finish gate)
)
```

## File Layout

```
aweagent/tasks/denovo_swe/
    __init__.py
    task.py                 # DeNovoSWETask
    evaluator.py            # DeNovoSWEEvaluator (multi-iter aggregation)
    clean.sh                # Source-cleaning shell script
    clean_predict.py        # Predict clean.sh deletions from a path string
    package_extractor.py    # Container-side pypi_name extractor (used by
                            # extract_patch.py)
    README.md               # This file
    prompt/
        __init__.py
        system.py           # System prompt registry (reuses beyondswe)
        user.py             # v1 + v2 user prompts (search and non-search)
```

## Anti-hack constraints

The agent prompt has an `<ANTI_CHEAT_CONSTRAINT>` block that forbids
fetching the target package from PyPI/conda/GitHub.  The runtime
backs that up with a generated bash blocklist:
`DeNovoSWETask.get_search_constraints` returns a
`PackageSearchConstraints` (in `aweagent.core.tool.search.package_constraints`)
seeded from `pypi_name`, `pypi_name_candidates`, `import_names`, and the
repo slug.  The blocklist covers every common package manager
(`pip`, `pipx`, `uv`, `poetry`, `conda`, `pdm`, `hatch`, `easy_install`,
`mamba`) plus name variants (dash/underscore/case/CamelCase splits) and
both VCS and direct-download channels.
