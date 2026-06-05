#!/usr/bin/env bash
# Run BrowseComp-style text QA with the IterResearch agent (Markovian context).
#
# Usage:
#   bash recipes/iter_research/run_iter_research_browsecomp.sh --data-file /path/bc_en200.jsonl
#   bash recipes/iter_research/run_iter_research_browsecomp.sh --data-file data.jsonl --dry-run
#   bash recipes/iter_research/run_iter_research_browsecomp.sh --data-file data.jsonl --max-steps 256 --max-concurrent 80

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/tasks/iter_research.yaml"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# Defaults can be overridden by environment variables or CLI flags.
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
DATA_FILE=""
MODEL="${MODEL:-${DEEPSEEK_MODEL:-deepseek-v4-pro}}"
BASE_URL="${BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}}"
MAX_STEPS="${MAX_STEPS:-256}"            # BrowseComp: 256 Markovian turns
MAX_CONCURRENT="${MAX_CONCURRENT:-80}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/iter_research_browsecomp}"
# Tool backends (web search / page reader). Keys read by the backends themselves.
SEARCH_BACKEND="${SEARCH_BACKEND:-serpapi}"
READER_BACKEND="${READER_BACKEND:-jina}"
# Remote Python sandbox: comma-separated endpoints (multi-endpoint pool).
SANDBOX_FUSION_ENDPOINTS="${SANDBOX_FUSION_ENDPOINTS:-}"
# Visit (web_fetch) page summarizer config (DeepSeek). Without it, web_fetch
# returns raw page content instead of a summary.
WEB_FETCH_CONFIG_PATH="${WEB_FETCH_CONFIG_PATH:-${PROJECT_ROOT}/configs/llm/web_fetch/deepseek_v4_pro.yaml}"
INSTANCE_IDS=()
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run BrowseComp-style text QA with the IterResearch agent.

Required:
  --data-file PATH                   BrowseComp JSON/JSONL/CSV/Parquet file

Environment variables:
  DEEPSEEK_API_KEY                   API key for the DeepSeek endpoint
  DEEPSEEK_BASE_URL                  LLM base URL (default: https://api.deepseek.com/v1)
  DEEPSEEK_MODEL                     DeepSeek served model id (default: deepseek-v4-pro)
  SEARCH_BACKEND                     Search backend name (default: serpapi; e.g. bandai)
  READER_BACKEND                     Reader backend name (default: jina; e.g. bandai)
  SERPAPI_API_KEY                    Required when using the serpapi search backend
  JINA_API_KEY                       Optional key for the jina reader backend
  SANDBOX_FUSION_ENDPOINTS           Comma-separated SandboxFusion endpoints for PythonInterpreter
  WEB_FETCH_CONFIG_PATH              Page-summary LLM config for Visit (default: DeepSeek)

Options:
  --config PATH                      Config file (default: configs/tasks/iter_research.yaml)
  --model MODEL                      DeepSeek model id (default: DEEPSEEK_MODEL or deepseek-v4-pro)
  --base-url URL                     OpenAI-compatible base URL
  --max-steps N                      Max Markovian turns per question (default: 256)
  --max-concurrent N                 Max concurrent instances (default: 80)
  --output-dir DIR                   Output directory (default: results/iter_research_browsecomp)
  --sandbox-endpoints CSV            Comma-separated SandboxFusion endpoints
  --instance-ids ID ...              Run only specific instance IDs
  --dry-run                          Load config and list instances without running
  -h, --help                         Show this help message

To run search+fetch only (no Python sandbox), remove the python_interpreter line
under agent.tools in the config — nothing else to change.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file) DATA_FILE="$2"; shift 2 ;;
        --config|-c) CONFIG="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --sandbox-endpoints) SANDBOX_FUSION_ENDPOINTS="$2"; shift 2 ;;
        --instance-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INSTANCE_IDS+=("$1")
                shift
            done
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
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
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "Error: DEEPSEEK_API_KEY environment variable is not set." >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}" -m awe_agent.cli run
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
export DEEPSEEK_MODEL="${MODEL}"
export DEEPSEEK_BASE_URL="${BASE_URL}"
export SEARCH_BACKEND="${SEARCH_BACKEND}"
export READER_BACKEND="${READER_BACKEND}"
export WEB_FETCH_CONFIG_PATH="${WEB_FETCH_CONFIG_PATH}"
export AWE_AGENT__EXECUTION__SAVE_TRAJECTORIES=true
export PYTHONUNBUFFERED=1
if [[ -n "${SANDBOX_FUSION_ENDPOINTS}" ]]; then
    export SANDBOX_FUSION_ENDPOINTS
fi

echo "=== IterResearch BrowseComp (DeepSeek) ==="
echo "Config:                   ${CONFIG}"
echo "Data file:                ${DATA_FILE}"
echo "Model:                    ${MODEL}"
echo "Base URL:                 ${BASE_URL}"
echo "Max steps (turns):        ${MAX_STEPS}"
echo "Max concurrent:           ${MAX_CONCURRENT}"
echo "Search / reader backend:  ${SEARCH_BACKEND} / ${READER_BACKEND}"
echo "Sandbox endpoints:        ${SANDBOX_FUSION_ENDPOINTS:-<built-in defaults>}"
echo "Page summary config:      ${WEB_FETCH_CONFIG_PATH}"
echo "Output dir:               ${OUTPUT_DIR}"
if [[ ${#INSTANCE_IDS[@]} -gt 0 ]]; then
    echo "Instance IDs:             ${INSTANCE_IDS[*]}"
fi
if [[ "${DRY_RUN}" == true ]]; then
    echo "Mode:                     DRY RUN"
fi
echo "=========================================="

exec "${CMD[@]}"
