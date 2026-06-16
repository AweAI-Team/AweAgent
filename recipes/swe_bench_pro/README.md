# SWE-bench-Pro Recipe

Faithful AweAgent port of SWE-bench-Pro.

## Download the data

`--data-file` is the only data path the recipe needs. Download the
AweAgent-processed dataset from:

> https://huggingface.co/datasets/AweAI-Team/AweAgent-Meta-SWE-Bench-Pro

```bash
# Downloads to datasets/swe_bench_pro/swe_bench_pro.jsonl (a symlink into
# the HuggingFace snapshot). HF_TOKEN / HF_ENDPOINT are honored.
bash datasets/swe_bench_pro/download.sh
```

Each row in this JSONL already ships everything the recipe needs:
`source_image` (runtime image) plus the prebuilt eval assets
`entryscript_sh`, `run_script_sh`, `parser_py`. No additional paths
(images.jsonl, official repo checkout, …) are required.

## Quick start

```bash
# Inspect prompt for one instance (no Docker needed)
python recipes/swe_bench_pro/run.py \
    --data-file datasets/swe_bench_pro/swe_bench_pro.jsonl \
    --instance-id <iid> --mode prompt

# List instances
python recipes/swe_bench_pro/run.py \
    --data-file datasets/swe_bench_pro/swe_bench_pro.jsonl \
    --mode dry-run

# Batch run
python recipes/swe_bench_pro/run.py \
    --data-file datasets/swe_bench_pro/swe_bench_pro.jsonl \
    --mode batch

# Convenience wrapper
bash recipes/swe_bench_pro/run_swe_bench_pro.sh \
    --data-file datasets/swe_bench_pro/swe_bench_pro.jsonl
```

`--data-file` also accepts JSON / parquet / yaml / dataset-dir / CSV
forms of the same dataset.

## Notes

* `requires_git_snapshot=False` mirrors the official SWE-bench-Pro flow:
  the patch is computed directly against `base_commit` rather than an
  AweAgent-injected pre-agent commit.
* `inject_gitignore_in_patch=False` strips the AweAgent auto-generated
  `.gitignore` block from the agent's patch so it matches the official
  `git diff <base_commit>` output byte-for-byte.

## Reproducibility Results
| Model | Average score | go | js | python | ts | Official result |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.6-35A3B | 0.416 | 0.354 | 0.345 | 0.504 | 0.700 | 0.495 |
