# BeyondSWE Task

Task implementation for the BeyondSWE benchmark. Supports four task types:

| Type | Description |
|------|-------------|
| `doc2repo` | Build a repository from a specification document |
| `crossrepo` | Fix issues spanning multiple files/modules |
| `depmigrate` | Dependency migration |
| `domainfix` | Domain-specific technical problems |

## Quick Start

```bash
# Via shell script (search mode)
bash recipes/beyond_swe/run_beyondswe_searchswe.sh --data-file data.jsonl

# Via Python entry point
python recipes/beyond_swe/run.py --data-file data.jsonl --mode batch
```

## Doc2Repo Evaluation: Configuring test_suite_dir

Doc2Repo evaluation requires local test suite zip files. The JSONL data only contains `test_suite` (the zip filename) without a local path, so you need to tell the evaluator where the zip files are via `test_suite_dir`.

Resolution priority for `test_suite_path`:

1. Constructor argument `test_suite_dir` (or `task.test_suite_dir` in YAML config)
2. Environment variable `BEYONDSWE_TEST_SUITE_DIR`

### Option 1: Environment Variable (Recommended)

```bash
export BEYONDSWE_TEST_SUITE_DIR=/path/to/doc2repo_test_suite
bash recipes/beyond_swe/run_beyondswe_searchswe.sh --data-file data.jsonl
```

### Option 2: YAML Config

```yaml
# configs/tasks/beyondswe_searchswe.yaml
task:
  type: beyond_swe
  dataset_id: beyond_swe
  data_file: ${DATA_FILE}
  test_suite_dir: /path/to/doc2repo_test_suite
```

### Option 3: Constructor Argument

```python
from awe_agent.tasks.beyond_swe.task import BeyondSWETask

task = BeyondSWETask(
    data_file="data.jsonl",
    test_suite_dir="/path/to/doc2repo_test_suite",
)
```

## Full Parameters

```python
BeyondSWETask(
    dataset_id="beyond_swe",       # Dataset identifier
    data_file=None,                # Path to JSONL data file
    instances=None,                # Raw instance dicts (alternative to data_file)
    search_mode=False,             # Enable search tool prompts
    test_suite_dir=None,           # Directory containing doc2repo test suite zips
)
```
