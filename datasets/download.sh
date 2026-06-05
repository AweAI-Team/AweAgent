#!/usr/bin/env bash
# Download benchmark data for AweAgent tasks.
#
#   bash datasets/download.sh browsecomp            # one task
#   bash datasets/download.sh browsecomp beyond_swe # several
#   bash datasets/download.sh all                   # everything wired up
#   FORCE=true bash datasets/download.sh browsecomp  # re-download
#
# Each <task> id matches the `dataset_id` in configs/tasks/<task>.yaml. Data
# lands under datasets/<task>/ and the matching config defaults to that path,
# so after downloading you can run the task with no extra env.
set -euo pipefail
DATASETS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FORCE="${FORCE:-false}"

# Task ids that have a downloader. Keep in sync with datasets/<task>/download.sh.
KNOWN_TASKS=(browsecomp terminal_bench_v2 beyond_swe)

usage() {
    cat <<EOF
Usage: bash datasets/download.sh <task> [<task> ...] | all

Tasks:
EOF
    local task script status
    for task in "${KNOWN_TASKS[@]}"; do
        script="${DATASETS_DIR}/${task}/download.sh"
        if [[ -f "$script" ]]; then status="ready"; else status="not wired yet"; fi
        printf "  %-20s %s\n" "$task" "$status"
    done
    cat <<EOF

Env:
  FORCE=true   re-download even if the data is already present
EOF
}

run_one() {
    local task="$1"
    local script="${DATASETS_DIR}/${task}/download.sh"
    if [[ ! -f "$script" ]]; then
        die "no downloader for '${task}' yet (${script#"${DATASETS_DIR}/"} missing)"
    fi
    echo "=== ${task} ==="
    bash "$script"
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

[[ $# -ge 1 ]] || { usage; exit 1; }
case "$1" in
    -h|--help) usage; exit 0 ;;
esac

if [[ "$1" == "all" ]]; then
    targets=("${KNOWN_TASKS[@]}")
else
    targets=("$@")
fi

for t in "${targets[@]}"; do
    run_one "$t"
done
