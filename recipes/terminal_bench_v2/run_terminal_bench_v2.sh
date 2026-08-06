#!/usr/bin/env bash
# Run Terminal Bench 2.0 benchmark.
#
# Usage:
#   bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
#       --task-data-dir /path/to/terminal-bench-2 --data-file /path/to/instance_ids.json
#   bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
#       --task-data-dir data/terminal-bench-2 --data-file data/instance_ids.json --model glm-5
#   bash recipes/terminal_bench_v2/run_terminal_bench_v2.sh \
#       --task-data-dir data/terminal-bench-2 --data-file data/instance_ids.json \
#       --instance-ids task_001 task_002

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/tasks/terminal_bench_v2.yaml"

# ── Defaults ──────────────────────────────────────────────────────────
TASK_DATA_DIR=""
DATA_FILE=""
MODEL="${MODEL:-}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-}"
VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-}"
CPU_MILLI="${CPU_MILLI:-}"
MEMORY_MB="${MEMORY_MB:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/terminal_bench_v2}"
INSTANCE_IDS=()
SKIP_EVAL=false
NO_TRAJECTORIES=false
VERBOSE=false

# ── Parse arguments ───────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run Terminal Bench 2.0 benchmark.

Data (optional — defaults to the downloaded dataset; see datasets/download.sh):
  --task-data-dir DIR    Root directory of task folders
  --data-file PATH       JSON file with instance ID array

Environment variables (optional):
  TERMINAL_BENCH_V2_PYPI_INDEX   PyPI index URL (default: https://pypi.org/simple)
  HTTP_PROXY / HTTPS_PROXY       Proxy for container network access

Options:
  --config, -c PATH     Task config YAML (default: configs/tasks/terminal_bench_v2.yaml)
  --instance-ids ID ... Run only specific instance IDs
  --model MODEL         Override LLM model (env: MODEL)
  --max-steps N         Max agent steps (default: 500, env: MAX_STEPS)
  --agent-timeout SEC   Agent wall-clock timeout (env: AGENT_TIMEOUT)
  --verifier-timeout SEC
                        /tests/test.sh timeout (env: VERIFIER_TIMEOUT)
  --max-concurrent N    Max concurrent instances (default: 10, env: MAX_CONCURRENT)
  --cpu-milli N         Global CPU limit in millicores (env: CPU_MILLI)
  --memory-mb N         Global memory limit in MiB (env: MEMORY_MB)
  --output DIR          Output directory (default: results/terminal_bench_v2, env: OUTPUT_DIR)
  --output-dir DIR      Backward-compatible alias for --output
  --skip-eval           Skip evaluation
  --no-trajectories     Don't save per-instance trajectories
  --verbose             Enable DEBUG logging
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
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --instance-ids)
            shift
            while [[ $# -gt 0 && "$1" != -* ]]; do
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
        --agent-timeout)
            AGENT_TIMEOUT="$2"
            shift 2
            ;;
        --verifier-timeout)
            VERIFIER_TIMEOUT="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --cpu-milli)
            CPU_MILLI="$2"
            shift 2
            ;;
        --memory-mb)
            MEMORY_MB="$2"
            shift 2
            ;;
        --output|--output-dir)
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
        --verbose)
            VERBOSE=true
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

# ── Defaults: fall back to the downloaded dataset ─────────────────────
# (bash datasets/download.sh terminal_bench_v2)
TASK_DATA_DIR="${TASK_DATA_DIR:-${PROJECT_ROOT}/datasets/terminal_bench_v2/tasks}"
DATA_FILE="${DATA_FILE:-${PROJECT_ROOT}/datasets/terminal_bench_v2/instance_ids.json}"

# ── Validate ──────────────────────────────────────────────────────────
if [[ ! -d "${TASK_DATA_DIR}" ]]; then
    echo "Error: Task data dir not found: ${TASK_DATA_DIR}" >&2
    echo "Run:   bash datasets/download.sh terminal_bench_v2" >&2
    exit 1
fi

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Error: Data file not found: ${DATA_FILE}" >&2
    echo "Run:   bash datasets/download.sh terminal_bench_v2" >&2
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
    --max-steps "${MAX_STEPS}"
    --max-concurrent "${MAX_CONCURRENT}"
    --output "${OUTPUT_DIR}"
)

if [[ -n "${MODEL}" ]]; then
    CMD+=(--model "${MODEL}")
fi

if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
fi

if [[ -n "${AGENT_TIMEOUT}" ]]; then
    CMD+=(--agent-timeout "${AGENT_TIMEOUT}")
fi

if [[ -n "${VERIFIER_TIMEOUT}" ]]; then
    CMD+=(--verifier-timeout "${VERIFIER_TIMEOUT}")
fi

if [[ -n "${CPU_MILLI}" ]]; then
    CMD+=(--cpu-milli "${CPU_MILLI}")
fi

if [[ -n "${MEMORY_MB}" ]]; then
    CMD+=(--memory-mb "${MEMORY_MB}")
fi

if [[ "${SKIP_EVAL:-false}" == true ]]; then
    CMD+=(--skip-eval)
fi

if [[ "${NO_TRAJECTORIES:-false}" == true ]]; then
    CMD+=(--no-trajectories)
fi

if [[ "${VERBOSE:-false}" == true ]]; then
    CMD+=(--verbose)
fi

# ── Export env vars for config resolution ─────────────────────────────
export TASK_DATA_DIR
export DATA_FILE

# ── Run ───────────────────────────────────────────────────────────────
echo "=== Terminal Bench 2.0 ==="
echo "Config:         ${CONFIG}"
echo "Task data dir:  ${TASK_DATA_DIR}"
echo "Data file:      ${DATA_FILE}"
echo "Max steps:      ${MAX_STEPS}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Output dir:     ${OUTPUT_DIR}"
if [[ -n "${MODEL}" ]]; then
    echo "Model:          ${MODEL}"
fi
if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "Instance IDs:   ${INSTANCE_IDS[*]}"
fi
if [[ -n "${AGENT_TIMEOUT}" ]]; then
    echo "Agent timeout:  ${AGENT_TIMEOUT}s"
fi
if [[ -n "${VERIFIER_TIMEOUT}" ]]; then
    echo "Verifier timeout: ${VERIFIER_TIMEOUT}s"
fi
if [[ -n "${CPU_MILLI}" ]]; then
    echo "CPU:            ${CPU_MILLI}m"
fi
if [[ -n "${MEMORY_MB}" ]]; then
    echo "Memory:         ${MEMORY_MB}Mi"
fi
echo "=========================="

cd "${PROJECT_ROOT}"
exec "${CMD[@]}"
