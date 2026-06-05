#!/usr/bin/env bash
# Download the BeyondSWE dataset from HuggingFace for the beyond_swe task.
#
#   pip install -e .                  # bundles huggingface_hub (or: pip install huggingface_hub)
#   bash datasets/beyond_swe/download.sh
#   FORCE=true bash datasets/beyond_swe/download.sh
#   HF_TOKEN=hf_xxx bash datasets/beyond_swe/download.sh    # if the repo is gated
#   HF_ENDPOINT=http://<mirror> bash datasets/beyond_swe/download.sh   # internal HF mirror
#
# Notes:
#   * HF_ENDPOINT (read by huggingface_hub at import) routes downloads through an
#     internal mirror. The script prints the effective endpoint so you can confirm.
#   * The "unauthenticated requests to the HF Hub" line is only a token rate-limit
#     warning (set HF_TOKEN to silence it); it prints regardless of HF_ENDPOINT.
#   * If a dataset uses Xet and your mirror does not proxy it, add HF_HUB_DISABLE_XET=1
#     (BeyondSWE has no Xet files, so this is not needed for it).
#
# Produces under datasets/beyond_swe/:
#   BeyondSWE/             the HuggingFace dataset snapshot
#   beyond_swe.jsonl  ->   symlink to the data JSONL inside the snapshot (the
#                          path the config defaults to; DATA_FILE overrides it)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

REPO_ID="${BEYOND_SWE_REPO_ID:-AweAI-Team/BeyondSWE}"
REVISION="${BEYOND_SWE_REVISION:-main}"
DEST="${AWE_DATASETS_DIR}/beyond_swe"
SNAP_DIR="${DEST}/BeyondSWE"
DATA_LINK="${DEST}/beyond_swe.jsonl"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
    die "huggingface_hub not installed. Run:  pip install -e .  (or: pip install huggingface_hub)"
fi

if already_have "$DATA_LINK"; then
    log "beyond_swe already present (FORCE=true to refresh): ${DATA_LINK}"
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
