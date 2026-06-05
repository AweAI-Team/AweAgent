#!/usr/bin/env bash
# Download the official BrowseComp eval set (OpenAI simple-evals).
#
#   bash datasets/browsecomp/download.sh
#   FORCE=true bash datasets/browsecomp/download.sh   # re-download
#
# Why there is NO decrypt step here:
# the file ships canary-ENCRYPTED, and we keep it encrypted at rest on purpose.
# The AweAgent browsecomp task decrypts each record in memory at load time
# (awe_agent/tasks/browsecomp/task.py -> _maybe_decrypt, using the `canary`
# column; same sha256-keyed XOR as the official decrypt). Plaintext eval
# answers therefore never touch disk, which keeps the benchmark uncontaminated.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

CSV_URL="https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
OUT="${AWE_DATASETS_DIR}/browsecomp/browse_comp_test_set.csv"

if already_have "$OUT"; then
    log "browsecomp already present (FORCE=true to re-download): ${OUT}"
    print_use BROWSECOMP_DATA_FILE "$OUT"
    exit 0
fi

log "downloading BrowseComp test set (canary-encrypted) ..."
download_url "$CSV_URL" "$OUT"
log "downloaded $(($(wc -l < "$OUT") - 1)) records -> ${OUT}"
print_use BROWSECOMP_DATA_FILE "$OUT"
