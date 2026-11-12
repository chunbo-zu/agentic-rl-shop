#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

die() {
    echo "错误：$*" >&2
    exit 2
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

is_model_dir() {
    local model_dir="$1" weight
    [[ -f "${model_dir}/config.json" ]] || return 1
    for weight in \
        model.safetensors \
        model.safetensors.index.json \
        pytorch_model.bin \
        pytorch_model.bin.index.json; do
        [[ -f "${model_dir}/${weight}" ]] && return 0
    done
    return 1
}

bool_value() {
    case "$2" in
        0|1) ;;
        *) die "$1 必须为 0 或 1，当前值：$2" ;;
    esac
}

positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 必须为正整数，当前值：$2"
}

DRY_RUN="${DRY_RUN:-0}"
RUN_SFT="${RUN_SFT:-1}"
RUN_GRPO="${RUN_GRPO:-1}"
ENABLE_SWANLAB="${ENABLE_SWANLAB:-1}"
ENABLE_LIGER="${ENABLE_LIGER:-1}"
ENABLE_QLORA="${ENABLE_QLORA:-0}"
SFT_GPUS="${SFT_GPUS:-4}"
GRPO_GPUS="${GRPO_GPUS:-4}"
GRPO_STEPS="${GRPO_STEPS:-100}"
GRPO_SAVE_FREQ="${GRPO_SAVE_FREQ:-25}"
GRPO_TEST_FREQ="${GRPO_TEST_FREQ:-25}"
GRPO_RUN_NAME="${GRPO_RUN_NAME:-grpo-qwen35-2b-stage-b-main}"
SHOPSIM_PORT="${SHOPSIM_PORT:-5700}"
SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS:-8}"
DATA_ROOT="$(realpath -m -- "${DATA_ROOT:-${ROOT}/outputs}")"
DEFAULT_BASE_MODEL="${ROOT}/../models/Qwen3.5-2B"
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"
PYTHON_BIN="${SFT_PYTHON:-${ROOT}/.venv/bin/python}"

bool_value DRY_RUN "${DRY_RUN}"
bool_value RUN_SFT "${RUN_SFT}"
bool_value RUN_GRPO "${RUN_GRPO}"
bool_value ENABLE_SWANLAB "${ENABLE_SWANLAB}"
bool_value ENABLE_LIGER "${ENABLE_LIGER}"
bool_value ENABLE_QLORA "${ENABLE_QLORA}"
positive_integer SFT_GPUS "${SFT_GPUS}"
positive_integer GRPO_GPUS "${GRPO_GPUS}"
positive_integer GRPO_STEPS "${GRPO_STEPS}"
positive_integer GRPO_SAVE_FREQ "${GRPO_SAVE_FREQ}"
positive_integer GRPO_TEST_FREQ "${GRPO_TEST_FREQ}"
positive_integer SHOPSIM_PORT "${SHOPSIM_PORT}"
positive_integer SHOPSIM_ENV_SLOTS "${SHOPSIM_ENV_SLOTS}"

[[ "${GRPO_RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] ||
    die "GRPO_RUN_NAME 只能包含字母、数字、点、下划线和连字符"
[[ "${SFT_GPUS}" =~ ^(1|2|4|8)$ ]] ||
    die "SFT_GPUS 必须是 1、2、4 或 8，以保持全局 batch size 8"
((8 % GRPO_GPUS == 0)) ||
    die "GRPO_GPUS 必须能整除实际 rollout batch 8"
[[ -x "${PYTHON_BIN}" ]] ||
    die "找不到 Python 环境：${PYTHON_BIN}；请先运行 bash scripts/setup.sh"
[[ -e "${BASE_MODEL}" ]] || die "找不到基础模型：${BASE_MODEL}"

MODELS_DIR="${DATA_ROOT}/models"
SFT_OUTPUT_ROOT="${MODELS_DIR}/sft-curriculum"
SFT_MODEL="${SFT_OUTPUT_ROOT}/stage-b/merged"
GRPO_OUTPUT_DIR="${MODELS_DIR}/${GRPO_RUN_NAME}"
GRPO_DATA_DIR="${DATA_ROOT}/datasets/grpo"
LOG_DIR="${DATA_ROOT}/logs/${GRPO_RUN_NAME}"
CACHE_DIR="${DATA_ROOT}/cache"
RUNTIME_DIR="${DATA_ROOT}/runtime/${GRPO_RUN_NAME}"
PIPELINE_LOG="${LOG_DIR}/pipeline.log"
SHOPSIM_LOG="${LOG_DIR}/shopsimulator.stdout.log"
SHOPSIM_URL="http://127.0.0.1:${SHOPSIM_PORT}"
RUN_HASH="$(printf '%s' "${GRPO_RUN_NAME}" | sha256sum | cut -c1-12)"
PYTHON_TMPDIR="${PYTHON_TMPDIR:-/tmp/shopping-grpo-mp-${RUN_HASH}}"
# Ray appends /ray/session_.../sockets/<socket> to this value.  Keeping its
# root short is required by Linux AF_UNIX's 107-byte path limit; the previous
# run-specific path put the plasma_store socket over that limit.
RAY_TMP_ROOT="${RAY_TMP_ROOT:-/tmp/shopping-rl-ray-${RUN_HASH}}"
RAY_SOCKET_PROBE="${RAY_TMP_ROOT}/ray/session_2099-12-31_23-59-59_999999_9999999/sockets/plasma_store"
RAY_SOCKET_BYTES="$(LC_ALL=C printf '%s' "${RAY_SOCKET_PROBE}" | wc -c)"
((RAY_SOCKET_BYTES <= 107)) ||
    die "RAY_TMP_ROOT 太长，Ray socket 将达到 ${RAY_SOCKET_BYTES} 字节（上限 107）：${RAY_TMP_ROOT}"
PYTHON_MP_SOCKET_PROBE="${PYTHON_TMPDIR}/pymp-12345678/listener-12345678"
PYTHON_MP_SOCKET_BYTES="$(LC_ALL=C printf '%s' "${PYTHON_MP_SOCKET_PROBE}" | wc -c)"
((PYTHON_MP_SOCKET_BYTES <= 107)) ||
    die "PYTHON_TMPDIR 太长，Python multiprocessing socket 将达到 ${PYTHON_MP_SOCKET_BYTES} 字节（上限 107）：${PYTHON_TMPDIR}"

# 将持久化缓存放在项目的 outputs 目录下；进程临时目录保持短路径。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${CACHE_DIR}/xdg"
export HF_HOME="${CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export VLLM_CACHE_ROOT="${CACHE_DIR}/vllm"
# The Ray worker setup hook appends a worker-specific directory to these cache
# roots before importing PyTorch/vLLM, avoiding concurrent autotune races.
export TRITON_CACHE_DIR="${CACHE_DIR}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}/torchinductor"
export TORCH_EXTENSIONS_DIR="${CACHE_DIR}/torch-extensions"
export TMPDIR="${PYTHON_TMPDIR}"
export SHOPPING_PYTHON_TMPDIR="${PYTHON_TMPDIR}"
export RAY_TMPDIR="${RAY_TMP_ROOT}"

GRPO_PREPARE_CMD=(
    "${PYTHON_BIN}" "${ROOT}/scripts/prepare_grpo_dataset.py"
    --train-jsonl "${ROOT}/data/grpo/train.jsonl"
    --train-parquet "${ROOT}/data/grpo/train.parquet"
    --validation-jsonl "${ROOT}/data/grpo/validation.jsonl"
    --validation-parquet "${ROOT}/data/grpo/validation.parquet"
    --sft-source "${ROOT}/data/sft_pure_v4/all.jsonl"
    --sft-manifest "${ROOT}/data/sft_curriculum/manifest.json"
    --evaluation "${ROOT}/data/evaluation/tasks.jsonl"
    --metadata "${ROOT}/data/grpo/metadata.json"
    --output-dir "${GRPO_DATA_DIR}"
)

GRPO_LOGGER=console
if [[ "${ENABLE_SWANLAB}" == "1" ]]; then
    [[ -n "${SWANLAB_API_KEY:-}" ]] ||
        die "ENABLE_SWANLAB=1 需要在环境变量中设置 SWANLAB_API_KEY"
    GRPO_LOGGER=swanlab
fi

GRPO_CMD=(
    "${PYTHON_BIN}" "${ROOT}/scripts/train_grpo.py"
    --model "${SFT_MODEL}"
    --train-data "${GRPO_DATA_DIR}/train.parquet"
    --val-data "${GRPO_DATA_DIR}/validation.parquet"
    --env-url "${SHOPSIM_URL}"
    --output "${GRPO_OUTPUT_DIR}"
    --experiment-name "${GRPO_RUN_NAME}"
    --logger "${GRPO_LOGGER}"
)
GRPO_CMD+=(
    --
    "trainer.n_gpus_per_node=${GRPO_GPUS}"
    "trainer.total_training_steps=${GRPO_STEPS}"
    "trainer.save_freq=${GRPO_SAVE_FREQ}"
    "trainer.test_freq=${GRPO_TEST_FREQ}"
)

next_sft_stage=a
if is_model_dir "${SFT_MODEL}"; then
    next_sft_stage=complete
elif is_model_dir "${SFT_OUTPUT_ROOT}/stage-b/merged"; then
    next_sft_stage=c
elif is_model_dir "${SFT_OUTPUT_ROOT}/stage-a/merged"; then
    next_sft_stage=b
fi

echo "Pipeline root : ${DATA_ROOT}"
echo "SFT output    : ${SFT_OUTPUT_ROOT}"
echo "SFT model     : ${SFT_MODEL}"
echo "GRPO data     : ${GRPO_DATA_DIR}"
echo "GRPO output   : ${GRPO_OUTPUT_DIR}"
echo "Logs          : ${LOG_DIR}"
echo "SFT GPUs      : ${SFT_GPUS}"
echo "GRPO GPUs     : ${GRPO_GPUS}"
echo "GRPO steps    : ${GRPO_STEPS}"
echo "Next SFT      : ${next_sft_stage}"
echo "Ray temp      : ${RAY_TMPDIR}"
echo "Python temp   : ${PYTHON_TMPDIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ "${RUN_SFT}" == "1" && "${next_sft_stage}" != complete ]]; then
        echo
        echo "[dry-run] SFT ${next_sft_stage} -> c"
        env \
            DRY_RUN=1 \
            SFT_PYTHON="${PYTHON_BIN}" \
            BASE_MODEL="${BASE_MODEL}" \
            SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT}" \
            SFT_START_STAGE="${next_sft_stage}" \
            SFT_STOP_AFTER_STAGE=c \
            NUM_PROCESSES="${SFT_GPUS}" \
            ENABLE_LIGER="${ENABLE_LIGER}" \
            ENABLE_QLORA="${ENABLE_QLORA}" \
            ENABLE_SWANLAB="${ENABLE_SWANLAB}" \
            SWANLAB_MODE=online \
            bash "${ROOT}/scripts/run_three_stage_sft.sh"
    else
        echo "[dry-run] SFT 已完成或由 RUN_SFT=0 跳过。"
    fi
    if [[ "${RUN_GRPO}" == "1" ]]; then
        echo
        echo "[dry-run] 生成并审计项目内的 GRPO 数据快照：${GRPO_DATA_DIR}"
        print_command "${GRPO_PREPARE_CMD[@]}"
        echo "[dry-run] 启动 ShopSimulator，日志写入 ${SHOPSIM_LOG}"
        echo "[dry-run] 四卡 GRPO"
        print_command "${GRPO_CMD[@]}"
    fi
    exit 0
fi

if ! mkdir -p \
    "${MODELS_DIR}" \
    "${GRPO_DATA_DIR}" \
    "${LOG_DIR}" \
    "${CACHE_DIR}/xdg" \
    "${CACHE_DIR}/huggingface/datasets" \
    "${CACHE_DIR}/vllm" \
    "${CACHE_DIR}/triton" \
    "${CACHE_DIR}/torchinductor" \
    "${CACHE_DIR}/torch-extensions" \
    "${RUNTIME_DIR}/tmp" \
    "${PYTHON_TMPDIR}" \
    "${RAY_TMPDIR}"; then
    die "无法创建流水线输出目录：${DATA_ROOT}"
fi
[[ -w "${DATA_ROOT}" ]] ||
    die "${DATA_ROOT} 不可写；请将该目录所有权授予当前用户 $(id -un)"

exec > >(tee -a "${PIPELINE_LOG}") 2>&1
echo
echo "[$(date --iso-8601=seconds)] pipeline start"

visible_gpus="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
required_gpus=0
if [[ "${RUN_GRPO}" == "1" ]]; then
    required_gpus="${GRPO_GPUS}"
fi
if [[ "${RUN_SFT}" == "1" && "${SFT_GPUS}" -gt "${required_gpus}" ]]; then
    required_gpus="${SFT_GPUS}"
fi
((visible_gpus >= required_gpus)) ||
    die "当前仅有 ${visible_gpus} 张可见 GPU，但流水线需要 ${required_gpus} 张"

if [[ "${RUN_SFT}" == "1" && "${next_sft_stage}" != complete ]]; then
    echo
    echo "[SFT] 从 Stage ${next_sft_stage^^} 训练到 Stage C"
    env \
        DRY_RUN=0 \
        SFT_PYTHON="${PYTHON_BIN}" \
        BASE_MODEL="${BASE_MODEL}" \
        SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT}" \
        SFT_START_STAGE="${next_sft_stage}" \
        SFT_STOP_AFTER_STAGE=c \
        NUM_PROCESSES="${SFT_GPUS}" \
        ENABLE_LIGER="${ENABLE_LIGER}" \
        ENABLE_QLORA="${ENABLE_QLORA}" \
        ENABLE_SWANLAB="${ENABLE_SWANLAB}" \
        SWANLAB_MODE=online \
        bash "${ROOT}/scripts/run_three_stage_sft.sh"
elif [[ "${next_sft_stage}" == complete ]]; then
    echo "[SFT] Stage C merged 已存在，跳过 SFT：${SFT_MODEL}"
else
    echo "[SFT] RUN_SFT=0，跳过 SFT。"
fi

if [[ "${RUN_GRPO}" != "1" ]]; then
    echo "[GRPO] RUN_GRPO=0，流水线在 SFT 后停止。"
    exit 0
fi
is_model_dir "${SFT_MODEL}" ||
    die "GRPO 需要完整的 Stage C merged 模型：${SFT_MODEL}"
if [[ -d "${GRPO_OUTPUT_DIR}" && -n "$(find "${GRPO_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "GRPO 输出目录必须不存在或为空：${GRPO_OUTPUT_DIR}；请更换 GRPO_RUN_NAME"
fi

echo
echo "[GRPO data] 生成并审计训练快照"
print_command "${GRPO_PREPARE_CMD[@]}"
"${GRPO_PREPARE_CMD[@]}"

SHOP_PID=""
cleanup() {
    local status=$?
    if [[ -n "${SHOP_PID}" ]]; then
        kill "${SHOP_PID}" 2>/dev/null || true
        wait "${SHOP_PID}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if curl --silent --output /dev/null --max-time 2 "${SHOPSIM_URL}/api/shop_agent"; then
    echo "[ShopSimulator] 复用已经运行的服务：${SHOPSIM_URL}"
else
    echo "[ShopSimulator] 启动服务：${SHOPSIM_URL}"
    SHOPSIM_PORT="${SHOPSIM_PORT}" \
    SHOPSIM_ENV_SLOTS="${SHOPSIM_ENV_SLOTS}" \
    SHOP_RUNTIME_DIR="${LOG_DIR}" \
        bash "${ROOT}/scripts/start_environment.sh" >"${SHOPSIM_LOG}" 2>&1 &
    SHOP_PID=$!

    shopsim_ready=0
    for _ in $(seq 1 180); do
        if curl --silent --output /dev/null --max-time 2 "${SHOPSIM_URL}/api/shop_agent"; then
            shopsim_ready=1
            break
        fi
        if ! kill -0 "${SHOP_PID}" 2>/dev/null; then
            tail -n 80 "${SHOPSIM_LOG}" >&2 || true
            die "ShopSimulator 在启动期间退出"
        fi
        sleep 2
    done
    [[ "${shopsim_ready}" == "1" ]] || die "ShopSimulator 在 360 秒内未就绪"
fi

curl --silent --show-error --fail \
    --max-time 30 \
    --header 'Content-Type: application/json' \
    --data '{"action":"release_all"}' \
    "${SHOPSIM_URL}/api/shop_agent" >/dev/null

echo
echo "[GRPO] Stage C -> ${GRPO_RUN_NAME}"
print_command "${GRPO_CMD[@]}"
"${GRPO_CMD[@]}"

echo
echo "[$(date --iso-8601=seconds)] pipeline complete"
echo "SFT model   : ${SFT_MODEL}"
echo "GRPO output : ${GRPO_OUTPUT_DIR}"
echo "Diagnostics : ${GRPO_OUTPUT_DIR}/training_diagnostics.jsonl"
