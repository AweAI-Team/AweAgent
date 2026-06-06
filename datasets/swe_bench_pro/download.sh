#!/usr/bin/env bash
# Download the SWE-bench-Pro dataset (AweAgent-processed) from HuggingFace.
#
#   pip install -e .                  # bundles huggingface_hub (or: pip install huggingface_hub)
#   bash datasets/swe_bench_pro/download.sh
#   FORCE=true bash datasets/swe_bench_pro/download.sh
#   HF_TOKEN=hf_xxx bash datasets/swe_bench_pro/download.sh    # if the repo is gated
#   HF_ENDPOINT=http://<mirror> bash datasets/swe_bench_pro/download.sh   # internal HF mirror
#
# Produces under datasets/swe_bench_pro/:
#   AweAgent-Meta-SWE-Bench-Pro/   the HuggingFace dataset snapshot
#   swe_bench_pro.jsonl       ->   symlink to the data JSONL inside the snapshot
#                                  (configs/tasks/swe_bench_pro.yaml defaults to it; DATA_FILE overrides).
#                                  Each row carries its own ``source_image``, so no images.jsonl is needed.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

REPO_ID="${SWE_BENCH_PRO_REPO_ID:-AweAI-Team/AweAgent-Meta-SWE-Bench-Pro}"
REVISION="${SWE_BENCH_PRO_REVISION:-main}"
DEST="${AWE_DATASETS_DIR}/swe_bench_pro"
SNAP_DIR="${DEST}/AweAgent-Meta-SWE-Bench-Pro"
DATA_LINK="${DEST}/swe_bench_pro.jsonl"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
    die "huggingface_hub not installed. Run:  pip install -e .  (or: pip install huggingface_hub)"
fi

if already_have "$DATA_LINK"; then
    log "swe_bench_pro already present (FORCE=true to refresh): ${DATA_LINK}"
    print_use DATA_FILE "$DATA_LINK"
    exit 0
fi

HF_ENDPOINT_EFF="$("$PYTHON_BIN" -c 'import huggingface_hub.constants as c; print(c.ENDPOINT)' 2>/dev/null || echo '?')"
log "HF endpoint: ${HF_ENDPOINT_EFF}   (override: export HF_ENDPOINT=<your mirror>)"
log "downloading HuggingFace dataset ${REPO_ID}@${REVISION} ..."
"$PYTHON_BIN" - "$REPO_ID" "$REVISION" "$SNAP_DIR" "$DATA_LINK" <<'PY'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, revision, snap_dir, data_link = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
local = Path(snapshot_download(
    repo_id=repo_id, repo_type="dataset", revision=revision, local_dir=str(snap_dir),
    token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
))

jsonls = sorted(local.rglob("*.jsonl"))
if not jsonls:
    types = sorted({p.suffix for p in local.rglob("*") if p.is_file()})
    sys.exit(f"No .jsonl found in {local} (file types present: {types}). "
             f"Set DATA_FILE to the correct file inside the snapshot.")

chosen = max(jsonls, key=lambda p: p.stat().st_size)
if len(jsonls) > 1:
    print("multiple .jsonl found; picking the largest:", chosen.name)
    print("  others:", ", ".join(p.name for p in jsonls if p != chosen))

if data_link.is_symlink() or data_link.exists():
    data_link.unlink()
data_link.symlink_to(chosen.resolve())
print(f"data file -> {chosen.relative_to(local)}  ({chosen.stat().st_size} bytes)")
PY

print_use DATA_FILE "$DATA_LINK"
