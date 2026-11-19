#!/usr/bin/env bash
# MODEL_PATH accepts a downloaded checkpoint directory or a Hugging Face repo ID.
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to your HF model ID or local checkpoint directory}"
exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-automaster}" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-30000}" \
  --tp "${TP:-1}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  "$@"
