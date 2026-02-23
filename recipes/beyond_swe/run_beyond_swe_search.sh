#!/usr/bin/env bash
# Run BeyondSWE benchmark in search mode with SearchSWE agent.
#
# Usage:
#   bash recipes/beyond_swe/run_beyond_swe_search.sh --data-file /path/to/data.jsonl
#   bash recipes/beyond_swe/run_beyond_swe_search.sh --data-file data.jsonl --model gpt-4o --dry-run
#   bash recipes/beyond_swe/run_beyond_swe_search.sh --data-file data.jsonl --instance-ids inst_001 inst_002

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/configs/tasks/beyond_swe_search.yaml"

# ── Defaults ──────────────────────────────────────────────────────────
DATA_FILE=""
MODEL="${MODEL:-gpt-4o}"
MAX_STEPS="${MAX_STEPS:-100}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/beyond_swe_search}"
INSTANCE_IDS=()
DRY_RUN=false

# ── Parse arguments ───────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run BeyondSWE benchmark in search mode.

Required:
  --data-file PATH       Path to BeyondSWE JSONL data file

Options:
  --model MODEL          LLM model name (default: gpt-4o, env: MODEL)
  --max-steps N          Max agent steps per instance (default: 100, env: MAX_STEPS)
  --max-concurrent N     Max concurrent instances (default: 50, env: MAX_CONCURRENT)
  --output-dir DIR       Output directory (default: results/beyond_swe_search, env: OUTPUT_DIR)
  --instance-ids ID ...  Run only specific instance IDs
  --dry-run              List instances without running
  -h, --help             Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file)
            DATA_FILE="$2"
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
    echo "Error: --data-file is required." >&2
    usage
    exit 1
fi

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Error: Data file not found: ${DATA_FILE}" >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set." >&2
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
export AWE_AGENT__LLM__MODEL="${MODEL}"

# ── Run ───────────────────────────────────────────────────────────────
echo "=== BeyondSWE Search Mode ==="
echo "Config:         ${CONFIG}"
echo "Data file:      ${DATA_FILE}"
echo "Model:          ${MODEL}"
echo "Max steps:      ${MAX_STEPS}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Output dir:     ${OUTPUT_DIR}"
if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "Instance IDs:   ${INSTANCE_IDS[*]}"
fi
if [[ "${DRY_RUN}" == true ]]; then
    echo "Mode:           DRY RUN"
fi
echo "=============================="

exec "${CMD[@]}"
