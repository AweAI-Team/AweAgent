#!/usr/bin/env bash
# Run BrowseComp-style text QA with the DeepSearch agent.
#
# Usage:
#   bash recipes/deepsearch/run_deepsearch_browsecomp.sh --data-file /path/to/browsecomp.json --dry-run
#   bash recipes/deepsearch/run_deepsearch_browsecomp.sh --data-file data.json --instance-ids test_2
#   bash recipes/deepsearch/run_deepsearch_browsecomp.sh --data-file data.json --max-steps 300 --rollout-retries 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/tasks/browsecomp.yaml"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# Defaults can be overridden by environment variables or CLI flags.
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
DATA_FILE=""
MODEL="${MODEL:-${OPENAI_MODEL:-gpt-5.4}}"
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
MAX_STEPS="${MAX_STEPS:-100}"
ROLLOUT_RETRIES="${ROLLOUT_RETRIES:-0}"
KEEP_RECENT_TOOL_RESULTS="${KEEP_RECENT_TOOL_RESULTS:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/deepsearch_browsecomp}"
TEMPERATURE="${TEMPERATURE:-}"
INSTANCE_IDS=()
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run BrowseComp-style text QA with the DeepSearch agent.

Required:
  --data-file PATH                   BrowseComp JSON/JSONL/CSV/Parquet file

Environment variables:
  OPENAI_API_KEY                     API key for the OpenAI-compatible LLM endpoint
  OPENAI_BASE_URL                    LLM base URL (default: https://api.openai.com/v1)
  OPENAI_MODEL                       Main model name
  SEARCH_BACKEND                     Search backend name, e.g. serpapi
  READER_BACKEND                     Reader backend name, e.g. jina
  SERPAPI_API_KEY                    Required when using the serpapi search backend
  JINA_API_KEY                       Optional key for the jina reader backend

Options:
  --config PATH                      Config file (default: configs/tasks/browsecomp.yaml)
  --model MODEL                      Main LLM model (default: OPENAI_MODEL or gpt-5.4)
  --base-url URL                     OpenAI-compatible base URL
  --max-steps N                      Max agent steps per rollout (default: 100)
  --rollout-retries N                Extra fresh rollout retries (default: 0)
  --keep-recent-tool-results N       Keep recent tool-result turns; -1 keeps all (default: 5)
  --full-context                     Shortcut for --keep-recent-tool-results -1
  --max-concurrent N                 Max concurrent instances (default: 10)
  --output-dir DIR                   Output directory (default: results/deepsearch_browsecomp)
  --temperature FLOAT                Optional LLM temperature override
  --instance-ids ID ...              Run only specific instance IDs
  --dry-run                          Load config and list instances without running
  -h, --help                         Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file)
            DATA_FILE="$2"
            shift 2
            ;;
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --rollout-retries)
            ROLLOUT_RETRIES="$2"
            shift 2
            ;;
        --keep-recent-tool-results)
            KEEP_RECENT_TOOL_RESULTS="$2"
            shift 2
            ;;
        --full-context)
            KEEP_RECENT_TOOL_RESULTS="-1"
            shift
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
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
            echo "Error: unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${DATA_FILE}" ]]; then
    echo "Error: --data-file is required." >&2
    usage
    exit 1
fi

if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Error: data file not found: ${DATA_FILE}" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config file not found: ${CONFIG}" >&2
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "Error: OPENAI_API_KEY environment variable is not set." >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}" -m aweagent.cli run
    --config "${CONFIG}"
    --output "${OUTPUT_DIR}"
    --max-steps "${MAX_STEPS}"
    --max-concurrent "${MAX_CONCURRENT}"
)

if [[ "${DRY_RUN}" == true ]]; then
    CMD+=(--dry-run)
fi

if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    CMD+=(--instance-ids "${INSTANCE_IDS[@]}")
fi

export BROWSECOMP_DATA_FILE="${DATA_FILE}"
export OPENAI_MODEL="${MODEL}"
export OPENAI_BASE_URL="${BASE_URL}"
export AWE_AGENT__AGENT__ROLLOUT_RETRIES="${ROLLOUT_RETRIES}"
export AWE_AGENT__AGENT__CONDENSER__KEEP_RECENT_TOOL_RESULTS="${KEEP_RECENT_TOOL_RESULTS}"
export AWE_AGENT__EXECUTION__SAVE_TRAJECTORIES=true
export PYTHONUNBUFFERED=1

if [[ -n "${TEMPERATURE}" ]]; then
    export AWE_AGENT__LLM__PARAMS__TEMPERATURE="${TEMPERATURE}"
fi

echo "=== DeepSearch BrowseComp ==="
echo "Config:                   ${CONFIG}"
echo "Data file:                ${DATA_FILE}"
echo "Model:                    ${MODEL}"
echo "Base URL:                 ${BASE_URL}"
echo "Max steps:                ${MAX_STEPS}"
echo "Rollout retries:          ${ROLLOUT_RETRIES}"
echo "Keep recent tool results: ${KEEP_RECENT_TOOL_RESULTS}"
echo "Max concurrent:           ${MAX_CONCURRENT}"
echo "Output dir:               ${OUTPUT_DIR}"
if [[ -n "${TEMPERATURE}" ]]; then
    echo "Temperature:              ${TEMPERATURE}"
fi
if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "Instance IDs:             ${INSTANCE_IDS[*]}"
fi
if [[ "${DRY_RUN}" == true ]]; then
    echo "Mode:                     DRY RUN"
fi
echo "============================="

exec "${CMD[@]}"
