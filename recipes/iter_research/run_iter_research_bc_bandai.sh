#!/usr/bin/env bash
# Run IterResearch on BrowseComp via the internal Bandai search/reader backend.
#
#   Agent LLM : DeepSeek V4 Pro (reasoning_effort=high)   — env DEEPSEEK_MODEL
#   Judge LLM : DeepSeek V4 Flash                          — env DEEPSEEK_JUDGE_MODEL
#   Search/Read: Bandai                                    — needs awe-agent-internal installed
#   Data      : a subset (default: BrowseComp_Lite_185.json) — --data-file to change
#   Output    : <OUTPUT_DIR>/<model>_<timestamp>/          — model name auto-appended
#
# Usage:
#   bash recipes/iter_research/run_iter_research_bc_bandai.sh                 # full subset
#   bash recipes/iter_research/run_iter_research_bc_bandai.sh --dry-run       # list instances
#   bash recipes/iter_research/run_iter_research_bc_bandai.sh --instance-ids test_2 test_11
#   bash recipes/iter_research/run_iter_research_bc_bandai.sh --max-concurrent 10
#
# Required env:  DEEPSEEK_API_KEY, BANDAI_USER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WKSP_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

# ── Defaults (override via env or flags) ──────────────────────────────────────
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/tasks/iter_research_bc_bandai.yaml}"
DATA_FILE="${BROWSECOMP_DATA_FILE:-${WKSP_ROOT}/BrowseComp_Lite_185.json}"   # the 185-item subset
MODEL="${MODEL:-${DEEPSEEK_MODEL:-deepseek-v4-pro}}"
JUDGE_MODEL="${DEEPSEEK_JUDGE_MODEL:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}}"
MAX_STEPS="${MAX_STEPS:-256}"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
OUTPUT_DIR="${OUTPUT_DIR:-${WKSP_ROOT}/output_dir/aweagent/bc/iterResearch}"
# Visit (web_fetch) page summarizer — DeepSeek Pro by default (faithful to the agent);
# point at a flash summarizer config to cut cost.
WEB_FETCH_CONFIG_PATH="${WEB_FETCH_CONFIG_PATH:-${PROJECT_ROOT}/configs/llm/web_fetch/deepseek_v4_pro.yaml}"
INSTANCE_IDS=()
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run IterResearch on BrowseComp via the internal Bandai search backend.

Required env:
  DEEPSEEK_API_KEY        DeepSeek API key
  BANDAI_USER             Bandai user (e.g. you@bytedance.com)

Options:
  --data-file PATH        BrowseComp subset/full file (default: BrowseComp_Lite_185.json)
  --config PATH           Task config (default: configs/tasks/iter_research_bc_bandai.yaml)
  --model MODEL           Agent model (default: deepseek-v4-pro)
  --judge-model MODEL     Judge model (default: deepseek-v4-flash)
  --max-steps N           Markovian turns per question (default: 256)
  --max-concurrent N      Concurrent instances (default: 20)
  --output-dir DIR        Base output dir; <model>_<timestamp> is appended automatically
  --instance-ids ID ...   Run only specific instance IDs (e.g. test_2 test_11)
  --dry-run               Load config + list instances without running
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-file) DATA_FILE="$2"; shift 2 ;;
        --config|-c) CONFIG="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --max-steps) MAX_STEPS="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --instance-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do INSTANCE_IDS+=("$1"); shift; done
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "Error: DEEPSEEK_API_KEY is not set." >&2; exit 1
fi
if [[ "${DRY_RUN}" != true && -z "${BANDAI_USER:-}" ]]; then
    echo "Error: BANDAI_USER is not set (required for the Bandai search backend)." >&2; exit 1
fi
if [[ ! -f "${DATA_FILE}" ]]; then
    echo "Error: data file not found: ${DATA_FILE}" >&2; exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config not found: ${CONFIG}" >&2; exit 1
fi

# ── Export env the config interpolates ────────────────────────────────────────
export BROWSECOMP_DATA_FILE="${DATA_FILE}"
export DEEPSEEK_MODEL="${MODEL}"
export DEEPSEEK_JUDGE_MODEL="${JUDGE_MODEL}"
export DEEPSEEK_BASE_URL="${BASE_URL}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export SEARCH_BACKEND="bandai"
export READER_BACKEND="bandai"
export WEB_FETCH_CONFIG_PATH="${WEB_FETCH_CONFIG_PATH}"
export AWE_AGENT__EXECUTION__SAVE_TRAJECTORIES=true
export PYTHONUNBUFFERED=1

CMD=(
    "${PYTHON_BIN}" -m awe_agent.cli run
    --config "${CONFIG}"
    --output "${OUTPUT_DIR}"
    --max-steps "${MAX_STEPS}"
    --max-concurrent "${MAX_CONCURRENT}"
)
[[ "${DRY_RUN}" == true ]] && CMD+=(--dry-run)
[[ ${#INSTANCE_IDS[@]} -gt 0 ]] && CMD+=(--instance-ids "${INSTANCE_IDS[@]}")

echo "=== IterResearch · BrowseComp · Bandai ==="
echo "Config:           ${CONFIG}"
echo "Data file:        ${DATA_FILE}"
echo "Agent model:      ${MODEL}  (reasoning_effort=high)"
echo "Judge model:      ${JUDGE_MODEL}"
echo "Search / reader:  bandai / bandai   (BANDAI_USER=${BANDAI_USER:-<unset>})"
echo "Page summarizer:  ${WEB_FETCH_CONFIG_PATH}"
echo "Max steps:        ${MAX_STEPS}    Max concurrent: ${MAX_CONCURRENT}"
echo "Output base:      ${OUTPUT_DIR}   (+ /<model>_<timestamp>/)"
[[ "${DRY_RUN}" == true ]] && echo "Mode:             DRY RUN"
echo "=========================================="

exec "${CMD[@]}"
