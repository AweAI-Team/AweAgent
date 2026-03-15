#!/usr/bin/env bash
# Run Terminal Bench 2.0 benchmark.
#
# Usage:
#   bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
#       --task-data-dir /path/to/terminal-bench-2 --data-file /path/to/instance_ids.json
#   bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
#       --task-data-dir data/terminal-bench-2 --data-file data/instance_ids.json --model glm-5 --dry-run
#   bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
#       --task-data-dir data/terminal-bench-2 --data-file data/instance_ids.json \
#       --instance-ids task_001 task_002 --mode batch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/tasks/terminal_bench_v2.yaml"

# ── Defaults ──────────────────────────────────────────────────────────
TASK_DATA_DIR=""
DATA_FILE=""
MODE="${MODE:-dry-run}"
MODEL="${MODEL:-}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/terminal_bench_v2}"
INSTANCE_ID=""
INSTANCE_IDS=()
DRY_RUN=false
SKIP_EVAL=false
NO_TRAJECTORIES=false

# ── Parse arguments ───────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run Terminal Bench 2.0 benchmark.

Required:
  --task-data-dir DIR    Root directory of task folders
  --data-file PATH       JSON file with instance ID array

Environment variables (optional):
  TERMINAL_BENCH_V2_PYPI_INDEX   PyPI index URL (default: https://pypi.org/simple)
  HTTP_PROXY / HTTPS_PROXY       Proxy for container network access

Options:
  --config PATH         Task config YAML (default: configs/tasks/terminal_bench_v2.yaml)
  --mode MODE           prompt | debug | batch | dry-run (default: dry-run, env: MODE)
  --instance-id ID      Single instance ID (prompt/debug)
  --instance-ids ID ... Run only specific instance IDs (batch)
  --model MODEL         Override LLM model (env: MODEL)
  --max-steps N         Max agent steps (default: 500, env: MAX_STEPS)
  --max-concurrent N    Max concurrent instances (default: 10, env: MAX_CONCURRENT)
  --output-dir DIR      Output directory (default: results/terminal_bench_v2, env: OUTPUT_DIR)
  --skip-eval           Skip evaluation
  --no-trajectories     Don't save per-instance trajectories
  --dry-run             List instances without running (same as --mode dry-run)
  -h, --help            Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-data-dir)
            TASK_DATA_DIR="$2"
            shift 2
            ;;
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --instance-id)
            INSTANCE_ID="$2"
            shift 2
            ;;
        --instance-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INSTANCE_IDS+=("$1")
                shift
            done
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --no-trajectories)
            NO_TRAJECTORIES=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            MODE="dry-run"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────
if [[ -z "${TASK_DATA_DIR}" ]]; then
    echo "Error: --task-data-dir is required." >&2
    usage
    exit 1
fi

if [[ -z "${DATA_FILE}" ]]; then
    echo "Error: --data-file is required." >&2
    usage
    exit 1
fi

if [[ ! -d "${TASK_DATA_DIR}" ]]; then
    echo "Error: Task data dir not found: ${TASK_DATA_DIR}" >&2
    exit 1
fi

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Error: Data file not found: ${DATA_FILE}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: Config file not found: ${CONFIG}" >&2
    exit 1
fi

# ── Build command ───────────────────────────────────────────────────
CMD=(
    python "${PROJECT_ROOT}/recipes/terminal_bench_v2/run.py"
    -c "${CONFIG}"
    --task-data-dir "${TASK_DATA_DIR}"
    --data-file "${DATA_FILE}"
    --mode "${MODE}"
    --max-steps "${MAX_STEPS}"
    --max-concurrent "${MAX_CONCURRENT}"
    --output "${OUTPUT_DIR}"
)

if [[ -n "${MODEL}" ]]; then
    CMD+=(--model "${MODEL}")
fi

if [[ -n "${INSTANCE_ID}" ]]; then
    CMD+=(--instance-id "${INSTANCE_ID}")
fi

if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
fi

if [[ "${SKIP_EVAL:-false}" == true ]]; then
    CMD+=(--skip-eval)
fi

if [[ "${NO_TRAJECTORIES:-false}" == true ]]; then
    CMD+=(--no-trajectories)
fi

# ── Export env vars for config resolution ─────────────────────────────
export TASK_DATA_DIR
export DATA_FILE

# ── Run ───────────────────────────────────────────────────────────────
echo "=== Terminal Bench 2.0 ==="
echo "Config:         ${CONFIG}"
echo "Task data dir:  ${TASK_DATA_DIR}"
echo "Data file:      ${DATA_FILE}"
echo "Mode:           ${MODE}"
echo "Max steps:      ${MAX_STEPS}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Output dir:     ${OUTPUT_DIR}"
if [[ -n "${MODEL}" ]]; then
    echo "Model:          ${MODEL}"
fi
if [[ -n "${INSTANCE_ID}" ]]; then
    echo "Instance ID:    ${INSTANCE_ID}"
fi
if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "Instance IDs:   ${INSTANCE_IDS[*]}"
fi
echo "=========================="

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
