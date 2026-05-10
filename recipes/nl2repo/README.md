# NL2Repo Recipe

Faithful AweAgent port of [NL2RepoBench](https://github.com/multimodal-art-projection/NL2RepoBench).

## Quick start

```bash
# Inspect prompt for one instance (no Docker needed)
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --instance-id aiofiles --mode prompt

# Full debug run on one instance
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --instance-id aiofiles --mode debug --verbose

# Batch run (uses the YAML config and TaskRunner)
python recipes/nl2repo/run.py \
    --data-file /path/to/data/nl2repo/nl2repo.jsonl \
    --mode batch

# Convenience wrapper
bash recipes/nl2repo/run_nl2repo.sh
```

## Data file

The default `--data-file` points to a pre-converted JSONL at
`/path/to/data/nl2repo/nl2repo.jsonl`.
The original NL2RepoBench CSV layout (`nl2repo_data_info.csv`) is also
accepted directly.

Each row has these fields (matches NL2RepoBench's CSV):

```json
{
  "evaluation_image": "your-registry.example.com/nl2repo/aiofiles:latest",
  "instance_id": "aiofiles",
  "package_name": "aiofiles",
  "start_instruction": "## Aiofiles project intro ...",
  "test_cases_num": 10,
  "verify_cmd": ["pytest"],
  "verify_files": ["tests"]
}
```

## Image selection

* `--data-file` → each row's `evaluation_image` becomes the per-instance
  evaluation image.
* `--agent-run-docker IMG` (or `AGENT_RUN_DOCKER` env) → optionally
  override the agent-side image. Empty string means agent shares the
  evaluation image (the default).
