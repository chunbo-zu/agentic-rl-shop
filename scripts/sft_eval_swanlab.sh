#!/usr/bin/env bash
# 一键执行 Pure V4 LoRA SFT、合并和 ShopSimulator Final-200 评测。
# 训练过程由 train_lora_sft.py 记录到 SwanLab，评测 summary 另建一个 run。
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEFAULT_BASE_MODEL="${ROOT}/../models/Qwen3.5-2B"
if [[ -d "${ROOT}/models/Qwen3.5-2B" ]]; then
  DEFAULT_BASE_MODEL="${ROOT}/models/Qwen3.5-2B"
fi
BASE_MODEL="${BASE_MODEL:-$DEFAULT_BASE_MODEL}"
PYTHON_BIN="${SFT_PYTHON:-${ROOT}/.venv/bin/python}"
SOURCE="${SFT_SOURCE:-${ROOT}/data/sft_pure_v4/all.jsonl}"
MANIFEST="${SFT_MANIFEST:-${ROOT}/data/sft_curriculum/manifest.json}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-${ROOT}/outputs/models/sft-qwen35-2b-curriculum}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${ROOT}/outputs/evaluation/sft-qwen35-2b}"
BENCHMARK="${EVAL_BENCHMARK:-${ROOT}/data/evaluation/tasks.jsonl}"

SWANLAB_PROJECT="${SWANLAB_PROJECT:-shopping-grpo-sft-qwen35-2b}"
SWANLAB_EVAL_RUN_NAME="${SWANLAB_EVAL_RUN_NAME:-qwen35-2b-sft-final200}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SHOPSIM_BASE_URL="${SHOPSIM_BASE_URL:-http://127.0.0.1:5700}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-shopping-agent-sft}"
LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
START_SERVICES="${START_SERVICES:-1}"
MAX_STEPS="${EVAL_MAX_STEPS:-35}"
MAX_TOKENS="${EVAL_MAX_TOKENS:-512}"

START_STAGE="${SFT_START_STAGE:-a}"
STOP_AFTER_STAGE="${SFT_STOP_AFTER_STAGE:-c}"
DRY_RUN=0
SKIP_EVAL=0
SHOP_PID=""
LLM_PID=""

# serve_model.sh 从环境变量读取 served model 名称。
export SERVED_MODEL_NAME
# 使用自定义 URL 启动子服务时同步端口；显式 SHOPSIM_PORT/LLM_PORT 优先。
if [[ "$SHOPSIM_BASE_URL" =~ :([0-9]+)$ ]]; then
  export SHOPSIM_PORT="${SHOPSIM_PORT:-${BASH_REMATCH[1]}}"
fi
if [[ "$LLM_BASE_URL" =~ :([0-9]+)/v1/?$ ]]; then
  export LLM_PORT="${LLM_PORT:-${BASH_REMATCH[1]}}"
fi

usage() {
  cat <<'USAGE'
用法：bash scripts/sft_eval_swanlab.sh [选项]

  --dry-run             只打印训练命令，不加载模型、不启动服务
  --skip-eval           只完成 SFT + merge，不启动环境或评测
  --no-start-services   不自动启动 ShopSimulator/vLLM（使用已有服务）
  --start-stage STAGE   从 a/b/c 开始（默认 a）
  --stop-after-stage STAGE
                        在 a/b/c 后停止（默认 c）
  -h, --help            显示帮助

常用环境变量：BASE_MODEL、SWANLAB_API_KEY、SWANLAB_MODE、SFT_OUTPUT_ROOT、
EVAL_OUTPUT_ROOT、START_SERVICES=0。BASE_MODEL 默认自动探测
models/Qwen3.5-2B（仓库内或仓库同级目录）。
USAGE
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --no-start-services) START_SERVICES=0; shift ;;
    --start-stage)
      [[ $# -ge 2 ]] || { echo "--start-stage 需要 a、b 或 c" >&2; exit 2; }
      START_STAGE="$2"; shift 2 ;;
    --stop-after-stage)
      [[ $# -ge 2 ]] || { echo "--stop-after-stage 需要 a、b 或 c" >&2; exit 2; }
      STOP_AFTER_STAGE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1（使用 --help 查看用法）" >&2; exit 2 ;;
  esac
done

case "$START_STAGE:$STOP_AFTER_STAGE" in
  a:a|a:b|a:c|b:b|b:c|c:c) ;;
  *) echo "阶段范围无效：start=$START_STAGE stop=$STOP_AFTER_STAGE" >&2; exit 2 ;;
esac

cleanup() {
  local status=$?
  if [[ -n "$LLM_PID" ]]; then
    kill "$LLM_PID" 2>/dev/null || true
    wait "$LLM_PID" 2>/dev/null || true
  fi
  if [[ -n "$SHOP_PID" ]]; then
    kill "$SHOP_PID" 2>/dev/null || true
    wait "$SHOP_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "找不到 Python 环境：$PYTHON_BIN；请先运行 bash scripts/setup.sh，或设置 SFT_PYTHON" >&2
  exit 2
fi
for required_file in "$SOURCE" "$MANIFEST" "$BENCHMARK"; do
  if [[ ! -f "$required_file" ]]; then
    echo "缺少输入文件：$required_file" >&2
    exit 2
  fi
done
TRAIN_CMD=(
  "$PYTHON_BIN" "$ROOT/scripts/run_sft_curriculum.py"
  --base-model "$BASE_MODEL"
  --source "$SOURCE"
  --manifest "$MANIFEST"
  --output-root "$SFT_OUTPUT_ROOT"
  --start-stage "$START_STAGE"
  --stop-after-stage "$STOP_AFTER_STAGE"
  --swanlab
  --swanlab-project "$SWANLAB_PROJECT"
  --swanlab-mode "$SWANLAB_MODE"
)

echo "[SFT] base model: $BASE_MODEL"
echo "[SFT] source:     $SOURCE"
echo "[SFT] output:     $SFT_OUTPUT_ROOT"
printf '[SFT] command:   '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'
if ((DRY_RUN)); then
  echo "[dry-run] 未执行训练、merge、服务启动或评测。"
  exit 0
fi
if [[ "$SWANLAB_MODE" == online && -z "${SWANLAB_API_KEY:-}" ]]; then
  echo "SwanLab online 模式需要 SWANLAB_API_KEY；或设置 SWANLAB_MODE=local" >&2
  exit 2
fi
mkdir -p "$SFT_OUTPUT_ROOT" "$EVAL_OUTPUT_ROOT"

"${TRAIN_CMD[@]}"
MERGED_MODEL="$SFT_OUTPUT_ROOT/stage-$STOP_AFTER_STAGE/merged"
if [[ ! -f "$MERGED_MODEL/config.json" ]]; then
  echo "SFT 完成但找不到合并模型：$MERGED_MODEL/config.json" >&2
  exit 1
fi
echo "[SFT] merged model: $MERGED_MODEL"
if ((SKIP_EVAL)); then
  echo "[eval] 已跳过。"
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "评测需要 curl 来探测服务状态；请先安装 curl。" >&2
  exit 2
fi

wait_http() {
  local url="$1" label="$2" attempts="${3:-180}" i
  for ((i = 1; i <= attempts; i++)); do
    if curl --silent --show-error --fail --max-time 3 "$url" >/dev/null 2>&1; then
      echo "[$label] ready: $url"
      return 0
    fi
    sleep 2
  done
  echo "[$label] 在 $((attempts * 2)) 秒内未就绪：$url" >&2
  return 1
}

llm_model_available() {
  curl --silent --show-error --fail --max-time 3 "${LLM_BASE_URL%/}/models" \
    | "$PYTHON_BIN" -c 'import json, sys; data=json.load(sys.stdin); raise SystemExit(0 if any(item.get("id") == sys.argv[1] for item in data.get("data", [])) else 1)' "$SERVED_MODEL_NAME"
}

if [[ "$START_SERVICES" == 1 ]]; then
  if ! curl --silent --show-error --fail --max-time 3 "$SHOPSIM_BASE_URL/" >/dev/null 2>&1; then
    echo "[shopsim] starting at $SHOPSIM_BASE_URL"
    "$ROOT/scripts/start_environment.sh" >"$EVAL_OUTPUT_ROOT/shopsim.log" 2>&1 &
    SHOP_PID=$!
  else
    echo "[shopsim] using existing service: $SHOPSIM_BASE_URL"
  fi
  wait_http "$SHOPSIM_BASE_URL/" shopsim

  if ! curl --silent --show-error --fail --max-time 3 "${LLM_BASE_URL%/}/models" >/dev/null 2>&1; then
    echo "[vllm] starting model $MERGED_MODEL as $SERVED_MODEL_NAME"
    "$ROOT/scripts/serve_model.sh" "$MERGED_MODEL" >"$EVAL_OUTPUT_ROOT/vllm.log" 2>&1 &
    LLM_PID=$!
  elif ! llm_model_available; then
    echo "[vllm] 现有服务没有模型名 $SERVED_MODEL_NAME；请设置 SERVED_MODEL_NAME 或停止占用端口的服务。" >&2
    exit 1
  else
    echo "[vllm] using existing service: $LLM_BASE_URL"
  fi
  wait_http "${LLM_BASE_URL%/}/models" vllm
  if ! llm_model_available; then
    echo "[vllm] 服务已响应，但没有暴露模型名 $SERVED_MODEL_NAME" >&2
    exit 1
  fi
else
  echo "[services] START_SERVICES=0；使用已有服务。"
  wait_http "$SHOPSIM_BASE_URL/" shopsim
  wait_http "${LLM_BASE_URL%/}/models" vllm
  if ! llm_model_available; then
    echo "[vllm] 已有服务没有暴露模型名 $SERVED_MODEL_NAME" >&2
    exit 1
  fi
fi

SUMMARY="$EVAL_OUTPUT_ROOT/summary.json"
TRAJECTORIES="$EVAL_OUTPUT_ROOT/trajectories.jsonl"
echo "[eval] benchmark: $BENCHMARK"
"$PYTHON_BIN" "$ROOT/scripts/evaluate_shop_benchmark.py" \
  --benchmark "$BENCHMARK" \
  --output "$TRAJECTORIES" \
  --summary "$SUMMARY" \
  --base-url "$SHOPSIM_BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --llm-base-url "$LLM_BASE_URL" \
  --api-key "$LLM_API_KEY" \
  --max-steps "$MAX_STEPS" \
  --max-tokens "$MAX_TOKENS"
"$PYTHON_BIN" "$ROOT/scripts/build_eval_report.py" --run-dir "$EVAL_OUTPUT_ROOT"

# 只上传聚合数值，不上传用户 query、raw observation 或完整轨迹。
"$PYTHON_BIN" - "$SUMMARY" "$SWANLAB_PROJECT" "$SWANLAB_EVAL_RUN_NAME" \
  "$EVAL_OUTPUT_ROOT/swanlab" "$BASE_MODEL" "$MERGED_MODEL" <<'PY'
import json
import os
import sys
from pathlib import Path

summary_path, project, run_name, logdir, base_model, merged_model = sys.argv[1:]
try:
    import swanlab
except ImportError as exc:
    raise SystemExit("缺少 SwanLab；请执行 uv sync --extra sft") from exc

Path(logdir).mkdir(parents=True, exist_ok=True)
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
protocol = summary.get("protocol") or {}
projection = summary.get("context_projection") or {}
metrics = {
    "eval/strict_success_rate": summary.get("strict_success_rate", 0.0),
    "eval/gold_purchase_rate": summary.get("gold_purchase_rate", 0.0),
    "eval/reward_valid_rate": summary.get("reward_valid_rate", 0.0),
    "eval/purchase_success_rate": summary.get("purchase_success_rate", 0.0),
    "eval/done_rate": summary.get("done_rate", 0.0),
    "eval/completed_tasks": summary.get("completed_tasks", 0),
    "eval/mean_final_reward": summary.get("mean_final_reward", 0.0),
    "eval/mean_terminal_utility": summary.get("mean_terminal_utility", 0.0),
    "eval/mean_weighted_score": summary.get("mean_weighted_score", 0.0),
    "eval/average_steps": summary.get("average_steps", 0.0),
    "eval/guard_rejections": projection.get("guard_rejections", 0),
    "eval/context_overflow_tasks": projection.get("context_overflow_tasks", 0),
}
metrics = {key: float(value) for key, value in metrics.items()}
swanlab.init(
    project=project,
    name=run_name,
    mode=os.environ.get("SWANLAB_MODE", "online"),
    logdir=logdir,
    config={
        "base_model": base_model,
        "merged_model": merged_model,
        "benchmark": protocol.get("benchmark", ""),
        "reward_contract": summary.get("reward_contract", ""),
        "max_steps": protocol.get("max_steps", 35),
        "max_tokens": protocol.get("max_tokens", 512),
        "temperature": protocol.get("temperature", 0.0),
        "top_p": protocol.get("top_p", 1.0),
    },
)
swanlab.log(metrics)
swanlab.finish()
print(json.dumps(metrics, ensure_ascii=False))
PY

echo "[done] summary: $SUMMARY"
echo "[done] report:  $EVAL_OUTPUT_ROOT/report.html"
echo "[done] SwanLab: project=$SWANLAB_PROJECT run=$SWANLAB_EVAL_RUN_NAME"
