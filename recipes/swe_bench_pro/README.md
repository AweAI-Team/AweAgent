# SWE-bench-Pro Recipe

Faithful AweAgent port of SWE-bench-Pro.

## Quick start

```bash
# Inspect prompt for one instance (no Docker needed)
python recipes/swe_bench_pro/run.py \
    --data-file /path/to/data/swe_bench_pro/swe_bench_pro.jsonl \
    --instance-id <iid> --mode prompt

# List instances
python recipes/swe_bench_pro/run.py \
    --data-file /path/to/data/swe_bench_pro/swe_bench_pro.jsonl \
    --mode dry-run

# Batch run
python recipes/swe_bench_pro/run.py \
    --data-file /path/to/data/swe_bench_pro/swe_bench_pro.jsonl \
    --mode batch

# Convenience wrapper
bash recipes/swe_bench_pro/run_swe_bench_pro.sh
```

## Data sources

* `--data-file` accepts JSONL/JSON/parquet/yaml/dataset-dir/CSV.
* `--images-jsonl` (or `SWEBENCH_PRO_IMAGES_JSONL` env) supplies the
  per-instance ``icm_image`` mapping.
* `--official-repo-root` (or `SWEBENCH_PRO_OFFICIAL_REPO_ROOT` env)
  points at the official SWE-bench-Pro repo and is required when the
  data rows do not ship prebuilt eval scripts (`entryscript_sh`,
  `run_script_sh`, `parser_py`).

The pre-converted defaults are at
`/path/to/data/swe_bench_pro/`.

## Notes

* `requires_git_snapshot=False` mirrors the official SWE-bench-Pro flow:
  the patch is computed directly against `base_commit` rather than an
  AweAgent-injected pre-agent commit.
* `inject_gitignore_in_patch=False` strips the AweAgent auto-generated
  `.gitignore` block from the agent's patch so it matches the official
  `git diff <base_commit>` output byte-for-byte.
