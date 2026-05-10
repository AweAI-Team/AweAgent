#!/usr/bin/env bash
# Run NL2Repo benchmark (faithful port of NL2RepoBench).
#
# Usage:
#   bash recipes/nl2repo/run_nl2repo.sh --data-file /path/to/nl2repo.jsonl
#   bash recipes/nl2repo/run_nl2repo.sh --data-file data.jsonl --model gpt-4o --dry-run
#   bash recipes/nl2repo/run_nl2repo.sh --data-file data.jsonl --instance-ids aiofiles cerberus

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/tasks/nl2repo.yaml"

# ── Defaults ──────────────────────────────────────────────────────────
DATA_FILE="${DATA_FILE:-/path/to/data/nl2repo/nl2repo.jsonl}"
AGENT_RUN_DOCKER="${AGENT_RUN_DOCKER:-}"
MODEL="${MODEL:-gpt-4o}"
MAX_STEPS="${MAX_STEPS:-400}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/nl2repo}"
INSTANCE_IDS=()
DRY_RUN=false

# ── Parse arguments ───────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run NL2Repo benchmark (faithful port of NL2RepoBench).

Options:
  --data-file PATH         Path to NL2RepoBench JSONL/CSV (default: \$DATA_FILE)
  --agent-run-docker IMG   Single image for the agent side; empty = use per-instance
                           evaluation_image (default: \$AGENT_RUN_DOCKER)
  --model MODEL            LLM model name (default: gpt-4o, env: MODEL)
  --max-steps N            Max agent steps per instance (default: 400, env: MAX_STEPS)
  --max-concurrent N       Max concurrent instances (default: 50, env: MAX_CONCURRENT)
  --output-dir DIR         Output directory (default: results/nl2repo, env: OUTPUT_DIR)
  --instance-ids ID ...    Run only specific instance IDs
  --dry-run                List instances without running
  -h, --help               Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --agent-run-docker)
            AGENT_RUN_DOCKER="$2"
            shift 2
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
        --instance-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INSTANCE_IDS+=("$1")
                shift
            done
            ;;
        --dry-run)
            DRY_RUN=true
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
if [[ -z "${DATA_FILE}" ]]; then
    echo "Error: --data-file is required (or DATA_FILE env var)." >&2
    usage
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

# ── Build command ─────────────────────────────────────────────────────
CMD=(
    awe-agent run
    -c "${CONFIG}"
    -o "${OUTPUT_DIR}"
    --max-steps "${MAX_STEPS}"
    --max-concurrent "${MAX_CONCURRENT}"
)

if [[ "${DRY_RUN}" == true ]]; then
    CMD+=(--dry-run)
fi

if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
fi

# ── Export env vars for config resolution ─────────────────────────────
export DATA_FILE
export AGENT_RUN_DOCKER
export AWE_AGENT__LLM__MODEL="${MODEL}"

# ── Run ───────────────────────────────────────────────────────────────
echo "=== NL2Repo ==="
echo "Config:           ${CONFIG}"
echo "Data file:        ${DATA_FILE}"
echo "Agent image:      ${AGENT_RUN_DOCKER:-<per-instance evaluation_image>}"
echo "Model:            ${MODEL}"
echo "Max steps:        ${MAX_STEPS}"
echo "Max concurrent:   ${MAX_CONCURRENT}"
echo "Output dir:       ${OUTPUT_DIR}"
if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "Instance IDs:     ${INSTANCE_IDS[*]}"
fi
if [[ "${DRY_RUN}" == true ]]; then
    echo "Mode:             DRY RUN"
fi
echo "================"

exec "${CMD[@]}"
