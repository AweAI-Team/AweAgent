# Benchmark data

One-stop download for the datasets the framework's tasks run on. Each task id
here matches the `dataset_id` in `configs/tasks/<task>.yaml`.

```bash
# from the AweAgent repo root
bash datasets/download.sh browsecomp        # one task
bash datasets/download.sh all               # everything wired up
FORCE=true bash datasets/download.sh browsecomp   # re-download
```

Data lands under `datasets/<task>/`, and each task config **defaults** to that
path (`data_file: ${VAR:-datasets/<task>/...}`). So after downloading you can
run the task with **no env var set**. Override only when you want a different
file:

```bash
export BROWSECOMP_DATA_FILE=/path/to/your/own.jsonl   # task-specific name
```

Env var names are kept **distinct per task** (not one shared `DATA_FILE`) so you
can set several at once and run multiple tasks without them colliding.

## Status

All three are wired.

| Task id | Source | Produces | Config env (override) |
|---|---|---|---|
| `browsecomp` | OpenAI simple-evals CSV (canary-encrypted) | file | `BROWSECOMP_DATA_FILE` |
| `terminal_bench_v2` | pinned git checkout of terminal-bench-2 | dir + id list | `TASK_DATA_DIR`, `DATA_FILE` |
| `beyond_swe` | HuggingFace `AweAI-Team/BeyondSWE` | jsonl | `DATA_FILE` |

`beyond_swe` needs the HuggingFace client, which `pip install -e .` now bundles
(or `pip install huggingface_hub` standalone). Set `HF_TOKEN` if the repo is gated;
`export HF_ENDPOINT=http://<mirror>` pulls through an internal HF mirror (the
script prints the effective endpoint so you can confirm it took effect). The
"unauthenticated requests to the HF Hub" line is only a token rate-limit
warning — harmless, and unrelated to whether the mirror is used.

## Notes

- **You normally set no env var.** Each task config defaults to the path its
  downloader writes (`data_file: ${VAR:-datasets/<task>/...}`), so running
  different tasks never collides on a shared variable. The override names above
  follow each task's existing convention (`DATA_FILE` is shared by the SWE-family
  tasks — only matters if you *explicitly* export it while running several).
- **BrowseComp stays encrypted at rest.** The downloader only fetches the
  canary-encrypted CSV — it does **not** decrypt. The browsecomp task decrypts
  each record in memory at load time, so plaintext eval answers never hit disk
  (keeps the benchmark uncontaminated).
- **Terminal-Bench** is a pinned, blobless, shallow git checkout. The downloader
  symlinks `datasets/terminal_bench_v2/tasks` to the detected task folder and
  writes `instance_ids.json` (the full run set — edit it to run a subset).
- The downloaded payloads are git-ignored; the scripts and this README are
  tracked.
- Run the scripts from the repo root so the relative default paths resolve.
