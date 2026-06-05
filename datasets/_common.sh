#!/usr/bin/env bash
# Shared helpers for datasets/<task>/download.sh scripts.
#
# Source this file; do not execute it directly. Each task downloader sets its
# own `set -euo pipefail`, then `source` this for logging + idempotent download.
#
# Conventions every downloader follows:
#   * data lands under datasets/<task>/  (a stable, predictable path)
#   * the matching task config defaults to that path via ${VAR:-<path>}, so a
#     plain `awe_agent run` works with no env set
#   * re-download with FORCE=true

# datasets/ root = the directory holding this file.
AWE_DATASETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# AweAgent project root = one level up (used for nicer relative log paths).
AWE_PROJECT_ROOT="$(cd "${AWE_DATASETS_DIR}/.." && pwd)"
export AWE_DATASETS_DIR AWE_PROJECT_ROOT

log()  { printf '[datasets] %s\n' "$*" >&2; }
warn() { printf '[datasets] WARNING: %s\n' "$*" >&2; }
die()  { printf '[datasets] ERROR: %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

# already_have <path> : succeeds if <path> exists (non-empty, for files),
# unless FORCE=true. Works for both files and directories.
already_have() {
    local target="$1"
    [[ "${FORCE:-false}" == "true" ]] && return 1
    [[ -e "$target" && ( ! -f "$target" || -s "$target" ) ]]
}

# download_url <url> <out> : fetch a single file with retries, atomically.
# Downloads to <out>.part and renames on success, so an interrupted/partial
# transfer never leaves a half file at <out> (which already_have would wrongly
# treat as complete). Prefers curl, falls back to wget, then python urllib.
download_url() {
    local url="$1" out="$2" tmp
    mkdir -p "$(dirname "$out")"
    tmp="${out}.part"
    rm -f "$tmp"   # drop any stale partial from a previous interrupted run
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --retry-delay 2 -o "$tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$tmp" "$url"
    else
        "${PYTHON_BIN:-python3}" - "$url" "$tmp" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
    fi
    mv -f "$tmp" "$out"
}

# print_use <ENV_VAR> <path> : tell the user the canonical path and how to
# point a task at a different file (the config already defaults to <path>).
print_use() {
    local var="$1" path="$2"
    log "ready -> ${path}"
    log "the task config defaults to this path; no env var needed to run."
    log "to use a different file:  export ${var}=/your/own/path"
}
