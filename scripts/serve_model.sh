#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-Qwen/Qwen3.5-2B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent}"
LLM_PORT="${LLM_PORT:-8000}"
VENV_BIN="$ROOT/.venv/bin"

if [[ ! -x "$VENV_BIN/vllm" ]]; then
  echo "vLLM is not installed. Run: bash scripts/setup.sh" >&2
  exit 1
fi

# vLLM/FlashInfer invokes helper executables such as ninja by name during JIT
# compilation.  Calling vllm by absolute path does not activate its virtualenv,
# so expose the complete environment explicitly.
export PATH="$VENV_BIN:$PATH"
if ! command -v ninja >/dev/null 2>&1; then
  echo "ninja is not installed in the vLLM environment. Run: bash scripts/setup.sh" >&2
  exit 1
fi

exec "$VENV_BIN/vllm" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$LLM_PORT" \
  --max-model-len 24576 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
