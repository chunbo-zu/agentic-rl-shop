#!/usr/bin/env bash
set -Eeuo pipefail

# 脚本位于 shopping-rl/scripts/ 下，因此仓库根目录是上一级目录。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 默认优先使用 /data 中已经存在的本地模型。
# 也可以在执行前通过 BASE_MODEL 覆盖为 Hugging Face 模型名或其他本地目录。
if [[ -d /data/models/Qwen3.5-2B ]]; then
    DEFAULT_BASE_MODEL=/data/models/Qwen3.5-2B
else
    DEFAULT_BASE_MODEL="${REPO_ROOT}/../models/Qwen3.5-2B"
fi
BASE_MODEL="${BASE_MODEL:-${DEFAULT_BASE_MODEL}}"

# SFT Python 环境及输入、输出位置。
export SFT_PYTHON="${SFT_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SFT_SOURCE="${SFT_SOURCE:-${REPO_ROOT}/data/sft_pure_v4/all.jsonl}"
SFT_MANIFEST="${SFT_MANIFEST:-${REPO_ROOT}/data/sft_curriculum/manifest.json}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-${REPO_ROOT}/outputs/models/sft-curriculum}"

# 可选配置：
#   DRY_RUN=1                 只打印六条 train/merge 命令
#   ENABLE_SWANLAB=1          开启 SwanLab
#   SWANLAB_MODE=local        本地记录，不要求 API Key
#   SWANLAB_MODE=online       在线记录，需要 SWANLAB_API_KEY
#   ENABLE_LIGER=1            融合 LM head + loss，避免 24K 序列物化全词表 logits
#   ENABLE_QLORA=1            以 NF4 量化加载基座；Liger 后仍 OOM 时再启用
#   NUM_PROCESSES=4           用四张 GPU 做 DDP；不会把四张卡合成一块显存
#   SFT_START_STAGE=a         从 a/b/c 开始，便于接续已完成的课程阶段
#   SFT_STOP_AFTER_STAGE=c    在 a/b/c 后停止
DRY_RUN="${DRY_RUN:-0}"
ENABLE_SWANLAB="${ENABLE_SWANLAB:-0}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-shopping-grpo-sft-curriculum}"
ENABLE_LIGER="${ENABLE_LIGER:-1}"
ENABLE_QLORA="${ENABLE_QLORA:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
SFT_START_STAGE="${SFT_START_STAGE:-a}"
SFT_STOP_AFTER_STAGE="${SFT_STOP_AFTER_STAGE:-c}"

# 四张卡的显存不会被单进程 Trainer 自动合并。这里先解决单样本的显存峰值；
# expandable_segments 用于减少动态长度 batch 导致的 CUDA allocator 碎片。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() {
    echo "错误：$*" >&2
    exit 2
}

[[ -x "${SFT_PYTHON}" ]] ||
    die "找不到 Python 环境：${SFT_PYTHON}；请先执行 bash scripts/setup.sh"

[[ -e "${BASE_MODEL}" ]] ||
    die "找不到基础模型：${BASE_MODEL}"

[[ -f "${SFT_SOURCE}" ]] ||
    die "找不到 SFT 数据：${SFT_SOURCE}"

[[ -f "${SFT_MANIFEST}" ]] ||
    die "找不到课程清单：${SFT_MANIFEST}"

[[ "${NUM_PROCESSES}" =~ ^[1248]$ ]] ||
    die "NUM_PROCESSES 必须是 1、2、4 或 8，以保持固定的全局 batch size 8"

case "${SFT_START_STAGE}:${SFT_STOP_AFTER_STAGE}" in
    a:a|a:b|a:c|b:b|b:c|c:c) ;;
    *) die "SFT 阶段范围无效：start=${SFT_START_STAGE} stop=${SFT_STOP_AFTER_STAGE}" ;;
esac

if [[ "${DRY_RUN}" != "1" && "${ENABLE_LIGER}" == "1" ]] &&
    ! "${SFT_PYTHON}" -c 'import liger_kernel' >/dev/null 2>&1; then
    die "ENABLE_LIGER=1 需要 liger-kernel；请执行：uv sync --extra sft --extra sft-accelerated --extra grpo"
fi

if [[ "${DRY_RUN}" != "1" && "${ENABLE_QLORA}" == "1" ]] &&
    ! "${SFT_PYTHON}" -c 'import bitsandbytes' >/dev/null 2>&1; then
    die "ENABLE_QLORA=1 需要 bitsandbytes；请执行：uv sync --extra sft --extra sft-accelerated --extra grpo"
fi

args=(
    --base-model "${BASE_MODEL}"
    --source "${SFT_SOURCE}"
    --manifest "${SFT_MANIFEST}"
    --output-root "${SFT_OUTPUT_ROOT}"
    --start-stage "${SFT_START_STAGE}"
    --stop-after-stage "${SFT_STOP_AFTER_STAGE}"
    --num-processes "${NUM_PROCESSES}"
)

if [[ "${ENABLE_LIGER}" == "1" ]]; then
    args+=(--liger-kernel)
fi

if [[ "${ENABLE_QLORA}" == "1" ]]; then
    args+=(--qlora)
fi

if [[ "${ENABLE_SWANLAB}" == "1" ]]; then
    if [[ "${SWANLAB_MODE}" == "online" && -z "${SWANLAB_API_KEY:-}" ]]; then
        die "SwanLab online 模式需要设置 SWANLAB_API_KEY"
    fi

    args+=(
        --swanlab
        --swanlab-project "${SWANLAB_PROJECT}"
        --swanlab-mode "${SWANLAB_MODE}"
    )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
fi

echo "Repository : ${REPO_ROOT}"
echo "Base model : ${BASE_MODEL}"
echo "SFT source : ${SFT_SOURCE}"
echo "Manifest   : ${SFT_MANIFEST}"
echo "Output     : ${SFT_OUTPUT_ROOT}"
echo "Liger      : ${ENABLE_LIGER}"
echo "QLoRA      : ${ENABLE_QLORA}"
echo "DDP GPUs   : ${NUM_PROCESSES}"
echo "Stages     : ${SFT_START_STAGE} -> ${SFT_STOP_AFTER_STAGE}"
echo "Allocator  : ${PYTORCH_CUDA_ALLOC_CONF}"
echo

exec bash "${REPO_ROOT}/scripts/sft_curriculum.sh" "${args[@]}"
