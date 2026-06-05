#!/usr/bin/env bash
# Download Terminal-Bench 2.0 task data (pinned commit) for the terminal_bench_v2 task.
#
#   bash datasets/terminal_bench_v2/download.sh
#   FORCE=true bash datasets/terminal_bench_v2/download.sh
#   TERMINAL_BENCH_COMMIT=<sha> bash datasets/terminal_bench_v2/download.sh   # pin another commit
#
# Produces under datasets/terminal_bench_v2/:
#   terminal-bench-2/    pinned git checkout (blobless + shallow; small)
#   tasks  ->  <dir>     symlink to the folder of task subdirs — the stable path
#                        the config defaults to (TASK_DATA_DIR), independent of
#                        the repo's internal layout
#   instance_ids.json    JSON array of every task id (the run set; edit to subset)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

REPO_URL="${TERMINAL_BENCH_REPO:-https://github.com/laude-institute/terminal-bench-2.git}"
COMMIT="${TERMINAL_BENCH_COMMIT:-69671fbaac6d67a7ef0dfec016cc38a64ef7a77c}"
DEST="${AWE_DATASETS_DIR}/terminal_bench_v2"
REPO_DIR="${DEST}/terminal-bench-2"
TASKS_LINK="${DEST}/tasks"
IDS_JSON="${DEST}/instance_ids.json"

require_cmd git

if already_have "$IDS_JSON" && [[ -e "$TASKS_LINK" ]]; then
    log "terminal_bench_v2 already present (FORCE=true to refresh): ${DEST}"
    log "  task dir  -> ${TASKS_LINK}   (config default: TASK_DATA_DIR)"
    log "  instances -> ${IDS_JSON}     (config default: DATA_FILE)"
    exit 0
fi

[[ "${FORCE:-false}" == "true" ]] && rm -rf "$REPO_DIR" "$TASKS_LINK" "$IDS_JSON"

# Pinned, blobless, shallow checkout: only the tree at COMMIT, no full history/blobs.
if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "cloning ${REPO_URL} (blobless, shallow, no-checkout) ..."
    git clone --filter=blob:none --depth 1 --no-checkout "$REPO_URL" "$REPO_DIR"
fi
log "fetching pinned commit ${COMMIT} ..."
git -C "$REPO_DIR" fetch --depth 1 origin "$COMMIT"
git -C "$REPO_DIR" checkout --quiet "$COMMIT"

# Detect the folder holding task subdirs (each has task.toml + instruction.md),
# symlink it to the stable path, and list every task id.
"${PYTHON_BIN:-python3}" - "$REPO_DIR" "$TASKS_LINK" "$IDS_JSON" <<'PY'
import json, os, shutil, sys
from pathlib import Path

repo, link, ids_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])

def task_ids(d):
    if not d.is_dir():
        return []
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and (p / "task.toml").exists() and (p / "instruction.md").exists()
    )

# Handle repo-root or a nested tasks/ layout: pick the dir with the most tasks.
best, best_ids = None, []
for c in [repo, repo / "tasks", repo / "dataset", repo / "datasets"]:
    ids = task_ids(c)
    if len(ids) > len(best_ids):
        best, best_ids = c, ids

if not best_ids:
    sys.exit(f"ERROR: found no task folders (task.toml + instruction.md) under {repo}")

if link.is_symlink() or link.exists():
    if link.is_dir() and not link.is_symlink():
        shutil.rmtree(link)
    else:
        link.unlink()
os.symlink(best.resolve(), link)
ids_json.write_text(json.dumps(best_ids, indent=2) + "\n")
print(f"detected {len(best_ids)} tasks under {best.relative_to(repo.parent)}")
PY

log "ready:"
log "  task dir  -> ${TASKS_LINK}   (config default: TASK_DATA_DIR)"
log "  instances -> ${IDS_JSON}     (config default: DATA_FILE; edit to run a subset)"
log "the config defaults to these paths; no env var needed to run."
