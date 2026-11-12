# Qwen3.5-2B 三阶段 SFT 训练与评估报告

> 报告日期：2026-09-16  
> 模型：Qwen3.5-2B  
> 训练方法：三阶段累计课程 LoRA SFT  
> 外部开发集：`dev119`  
> 环境：ShopSimulator Environment v2.1  
> Reward：ShopSimulator Reward v3

## 1. 摘要

三个阶段均已完成训练、LoRA 合并和同协议 `dev119` 评估。最主要的结论是：

1. Stage A 到 Stage B 出现明确收益，严格成功率从 `57.98%` 提高到
   `66.39%`，净增 10 个严格成功任务；提升主要来自 constraints 桶。
2. Stage B 到 Stage C 的严格成功率没有继续提高，均为 `79/119`。Stage C
   的平均 Reward 和加权匹配分略高，但终局率、Reward 可验证率和动作合法性略差。
3. Stage C 对 strategy 桶的平均 Reward 有改善，但严格成功仍为 `10/31`。
   其作用更接近提高部分满足程度，而不是增加完整成功。
4. 当前策略的主要瓶颈不再是工具格式，而是单候选购买、规格核验、预算/品类
   硬门槛和失败后的重新搜索。Stage C 的每个任务都只搜索一次、只打开一个商品。
5. Stage C 标称 `1073/119` 个训练/验证样本，实际进入 Trainer 的是
   `1069/118`；4 条训练轨迹和 1 条验证轨迹超过 24,576 token，被完整丢弃。
6. 若只部署 SFT，Stage B 是更稳健的 checkpoint；若继续 GRPO，建议以 Stage C
   为主线、Stage B 为小规模对照，因为 Stage C 覆盖 strategy supervision 且具有
   更高平均 Reward，但尚未证明其严格成功优于 Stage B。

## 2. 训练设计

课程清单来自 [`data/sft_curriculum/manifest.json`](../../data/sft_curriculum/manifest.json)，
训练源为 Pure V4 的 1,192 条唯一任务。SFT 只对 assistant token 计算 loss，system、
user 和 tool observation 均作为上下文并被 mask。

| 阶段 | 累计数据桶 | 清单训练数 | 清单验证数 | Epoch | LR | 初始化模型 |
|---|---|---:|---:|---:|---:|---|
| A | foundation | 256 | 28 | 1 | `1e-4` | Qwen3.5-2B |
| B | foundation + constraints | 799 | 88 | 1 | `7e-5` | Stage A merged |
| C | foundation + constraints + strategy | 1,073 | 119 | 1 | `5e-5` | Stage B merged |

累计课程会隐式增加简单任务的曝光次数：foundation 训练三遍、constraints 两遍、
strategy 一遍。按清单计算，共有 2,128 次样本曝光；strategy 中最困难的长轨迹又有
4 条因上下文长度被过滤。因此，本实验不能把 Stage C 的平台期直接解释为
“strategy supervision 无效”。

## 3. 训练健康度

| 阶段 | 实际训练/验证样本 | Train loss | Eval loss | Trainer 步数 | 总耗时 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| A | 256 / 28 | 0.4221 | 0.3539 | 32 | 12.7 min | 35.79 GiB |
| B | 799 / 88 | 0.3597 | 0.3387 | 100 | 43.8 min | 49.93 GiB |
| C | 1,069 / 118 | 0.3319 | 0.3481 | 134 | 113.2 min | 68.97 GiB |

训练日志中未观察到 NaN、OOM 或梯度爆炸。不同阶段的 Eval loss 不能直接横向作为
checkpoint 排名：A、B、C 的验证集组成不同，后续阶段加入了更难的桶。

Stage C 被长度过滤的样本如下：

| Task ID | Split | 渲染长度 | 处理结果 |
|---:|---|---:|---|
| 983 | train | 31,900 | 超过 24,576，整条丢弃 |
| 1223 | train | 25,741 | 超过 24,576，整条丢弃 |
| 4526 | validation | 25,675 | 超过 24,576，整条丢弃 |
| 13587 | train | 26,948 | 超过 24,576，整条丢弃 |
| 16539 | train | 28,494 | 超过 24,576，整条丢弃 |

整条丢弃优于在未知边界截断，因为后者可能生成不完整的工具调用；后续应采用保留
动作语义的轨迹压缩或分段方案，而不是直接从尾部截断。

## 4. 评估协议

三次外部评估均使用完全相同的 `dev119`：

- 119 个任务全部生成轨迹，没有 missing task；
- `temperature=0.0`、`top_p=1.0`；
- 最大环境步骤 `35`；
- 单次模型生成上限 `512` token；
- 上下文窗口 `24,576` token；
- observation projection 配置一致；
- 严格成功只认 Reward v3 的合法 `gold_purchase`，且要求
  `reward_valid=true`、完整环境终局和 `purchase_success=true`。

Stage A 目录另有一份早期 `Connection refused` 基础设施失败记录。正式
`trajectories.jsonl` 是后续完整重跑的 119 条结果，早期失败不计入模型指标。

## 5. 总体结果

| 指标 | Stage A | Stage B | Stage C |
|---|---:|---:|---:|
| 严格成功 | 69 / 119 | **79 / 119** | **79 / 119** |
| 严格成功率 | 57.98% | **66.39%** | **66.39%** |
| 平均最终 Reward | 0.5059 | 0.5959 | **0.6088** |
| 平均加权匹配分 | 0.7885 | 0.8078 | **0.8121** |
| 环境终局率 | **99.16%** | 98.32% | 97.48% |
| Reward 可验证率 | **99.16%** | 97.48% | 95.80% |
| 平均步骤 | **7.11** | 8.22 | 7.99 |
| Partial alternative purchase | 38 | **28** | **28** |
| Wrong purchase | 10 | 8 | **7** |
| Guard rejection | 10 | **9** | 13 |
| Context overflow task | **0** | 1 | **0** |
| Invalid-action-limit task | **1** | **1** | 3 |

### 5.1 Stage A 到 Stage B

- 严格成功率提高 `8.40` 个百分点；
- 15 题由失败转成功，5 题由成功转失败，净增 10 题；
- 配对 exact McNemar/符号检验双侧 `p≈0.041`；
- Partial purchase 从 38 降至 28；
- Wrong purchase 从 10 降至 8；
- 平均步骤增加约 1.11，主要来自更完整的详情和规格核验。

这是真正产生主要课程收益的阶段。

### 5.2 Stage B 到 Stage C

- 5 题由失败转成功，同时 5 题由成功转失败；
- 净严格成功为 0，配对检验 `p=1.0`；
- 平均 Reward 增加 `0.0129`；
- 平均加权匹配分增加 `0.0043`；
- Wrong purchase 少 1 个；
- Guard rejection 增加 4 次，invalid-action-limit 增加 2 题；
- Reward 可验证率下降 1.68 个百分点。

因此，Stage C 尚未表现出严格成功增益。它改善了部分失败的匹配程度，但也引入了
任务级回退和动作合法性问题。

## 6. 分难度桶结果

| 开发集桶 | Stage A | Stage B | Stage C |
|---|---:|---:|---:|
| Foundation（28） | 20/28，71.4% | **21/28，75.0%** | 20/28，71.4% |
| Constraints（60） | 40/60，66.7% | 48/60，80.0% | **49/60，81.7%** |
| Strategy（31） | 9/31，29.0% | **10/31，32.3%** | **10/31，32.3%** |

分桶平均 Reward：

| 桶 | Stage A | Stage B | Stage C |
|---|---:|---:|---:|
| Foundation | 0.7129 | **0.7381** | 0.7039 |
| Constraints | 0.6217 | 0.7650 | **0.7865** |
| Strategy | 0.0949 | 0.1402 | **0.1790** |

Stage C 在 strategy 上并非完全无效：平均 Reward 连续提高，但尚未跨过严格成功门槛。
这类连续 Reward 差异正是 GRPO 可以利用的信号。

## 7. 失败结构与行为诊断

Stage C 的 119 个结果为：

| 结果 | 数量 | 解释 |
|---|---:|---|
| `gold_purchase` | 79 | 严格成功 |
| `partial_alternative_purchase` | 28 | 品类和预算通过，但其他要求未完全满足 |
| `wrong_purchase` | 7 | 预算或品类硬门槛错误 |
| `reward_unverifiable` | 2 | 最终 variant 价格不能唯一解析 |
| 非终局 invalid-action-limit | 3 | 连续非法点击后终止 |

28 个 partial purchase 的共同特点：

- 28/28 的品类与预算均通过；
- 28/28 至少选错或漏掉一个关键规格；
- 16/28 没有完全满足核心功能；
- 3/28 还存在型号不匹配。

7 个 wrong purchase 中，5 个违反预算，3 个违反品类，其中 1 个同时违反两项。
两个 `reward_unverifiable` 都来自多个有效价格规格轴，模型在无法确认最终价格时仍然
购买。三个 invalid-action-limit task 为 `5044`、`11426` 和 `20473`，均重复触发
`click_not_in_previous_observation`。

### 7.1 单候选策略是当前最明显的瓶颈

Stage C 在全部 119 个任务上都只执行一次搜索，并且每题只打开一个商品，没有打开
第二个候选。116 个正常环境终局全部是购买终局，没有一次合理放弃。当前行为近似：

```text
搜索一次 -> 打开一个商品 -> 查看详情/规格 -> 购买
```

它没有稳定学到：

```text
搜索 -> 核验候选 A -> 不满足则返回或改搜 -> 核验候选 B
     -> 比较后购买，或在充分探索后放弃
```

因此，后续 GRPO 的首要目标应是候选比较、失败恢复、硬门槛遵守、规格完整性和合理
终止，而不是继续强化已经掌握的基础工具格式。

## 8. SFT checkpoint 决策

### 8.1 独立 SFT 部署

推荐 Stage B：

- 与 Stage C 相同的严格成功率；
- 更高的终局率和 Reward 可验证率；
- 更少的 Guard rejection 和非法动作终止；
- foundation 保持更好。

路径：

```text
outputs/models/sft-qwen35-2b-curriculum/stage-b/merged
```

### 8.2 GRPO 初始化

主线推荐 Stage C，Stage B 作为对照：

- Stage C 已接触 strategy supervision；
- constraints 和 strategy 的平均 Reward 更高；
- 即使没有新增 gold purchase，partial reward 的可分性仍可为在线 GRPO 提供优势信号；
- 但 Stage B 的合法性更好，因此应保留短预算对照，防止 Stage C 的动作退化延续到 RL。

主线路径：

```text
outputs/models/sft-qwen35-2b-curriculum/stage-c/merged
```

## 9. GRPO 数据隔离修复

原始 GRPO 文件在 Pure V4 课程晋升前生成。针对当时的文件审计发现，GRPO train
与当前 SFT 任务存在 6 个交集；这会污染 SFT 到 GRPO 的增益归因，因此是正式训练
前的阻断项：

| 交集 | 数量 | Task IDs |
|---|---:|---|
| SFT train 与 GRPO train | 4 | `8993, 11298, 12380, 17219` |
| SFT dev/dev119 与 GRPO train | 2 | `7682, 14953` |
| SFT 与 GRPO validation | 0 | - |
| SFT 与 Final-200 | 0 | - |
| GRPO train/validation 与 Final-200 | 0 | - |

Final-200 本身是干净的，问题只在训练阶段之间的归因和 dev119 checkpoint 选择。
仓库没有可审计的额外 GRPO 候选池，因此采用最低可行方案：从现有 GRPO train
同步过滤上述 6 个 task，保留 validation 不变。

已执行：

```bash
.venv/bin/python scripts/prepare_grpo_dataset.py
```

脚本 `scripts/prepare_grpo_dataset.py` 会读取 Pure V4 source、课程 manifest 和
Final-200 清单，验证输入 JSONL/Parquet 顺序一致后同步过滤两种格式，重新连续编号
Parquet `extra_info.index`，原子写回文件，并记录输入 hash、排除 task ID、来源和
最终审计结果。当前发布结果为：

- train：`994` tasks（原始 `1,000`，移除 6 个）;
- validation：`50` tasks（未移除）;
- JSONL 与 Parquet task ID 顺序完全一致；
- `data/grpo/metadata.json` 中的文件 hash 与实际文件一致；
- 脚本重复执行保持结果不变。

被移除的 task ID 为：`7682, 8993, 11298, 12380, 14953, 17219`。

修复后的验收条件为：

```text
active SFT train/dev x GRPO train/validation = 0
dev119 x GRPO train/validation = 0
Final-200 x every training/development split = 0
JSONL task IDs == Parquet extra_info.task_id
metadata counts and SHA-256 == actual files
```

上述条件现已全部满足，数据阻断项已解除。正式训练仍应先做独立的运行时 smoke；
本报告本轮只完成数据准备和审计，没有启动 GRPO 训练、模型服务或 Final-200 评估。

## 10. 推荐 GRPO 实施方案

### 10.1 保持第一轮为干净的 vanilla GRPO

第一轮不要同时打开 TRACE、Clip-Higher、KL 或长度惩罚。当前基线已经包含：

- `n=4` stochastic rollout；
- `temperature=0.7`、`top_p=0.9`；
- Reward v3 终局奖励；
- token-mean policy loss；
- 最多三批的 bounded dynamic sampling；
- entropy 记录但不加 entropy loss；
- 单卡 96 GB 的 LoRA + offload 配置。

先建立可解释基线，之后只依据训练诊断做单变量消融。

### 10.2 环境安装与服务

在仓库根目录执行。首次或依赖变化后：

```bash
bash scripts/setup.sh
```

该命令会校验固定依赖、准备 ShopSimulator、构建索引并应用带哈希检查的 veRL 0.8.0
动态采样补丁。

在独立终端启动并保持 ShopSimulator：

```bash
bash scripts/start_environment.sh
```

确认服务可用：

```bash
curl --fail --request OPTIONS http://127.0.0.1:5700/api/shop_agent
```

GRPO 使用 veRL 内部 vLLM rollout，不需要另行运行 `scripts/serve_model.sh`。开始训练前
应停止占用 GPU 的外部 vLLM 服务。

### 10.3 解析命令而不启动训练

当前清洗后的 canonical 数据位于 `data/grpo/`。先检查最终解析结果：

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-qwen35-2b-curriculum/stage-c/merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/grpo-qwen35-2b-stage-c-smoke \
  --experiment-name qwen35-2b-stage-c-smoke \
  --dry-run -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10 \
  trainer.test_freq=10
```

`--dry-run` 只验证入口路径并打印命令；真正运行时还会在模型加载前执行版本、CUDA、
环境 manifest、Reward runtime hash、动态采样补丁和显存配置预检。

### 10.4 运行 20-step smoke

建议继续沿用 SwanLab；不要把 API key 写入脚本或报告：

```bash
export SWANLAB_API_KEY='...'

bash scripts/grpo.sh \
  --model outputs/models/sft-qwen35-2b-curriculum/stage-c/merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/grpo-qwen35-2b-stage-c-smoke \
  --experiment-name qwen35-2b-stage-c-smoke \
  --logger swanlab -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10 \
  trainer.test_freq=10
```

输出目录必须不存在或为空。运行会写入：

```text
outputs/models/grpo-qwen35-2b-stage-c-smoke/
  global_step_*/
  training_diagnostics.jsonl
  swanlab/
```

### 10.5 Smoke 验收门槛

20 step 完成后至少检查：

1. 没有 OOM、Ray worker crash、环境租约泄漏或上下文错误；
2. `training_diagnostics.jsonl` 同时包含 `generation_batch`、`optimizer_step`；
3. `group/effective_ratio` 不是持续接近零；
4. `group/resample_batches` 和 skipped update 没有连续触及安全上限；
5. `group/sampling_invalid` 接近零，任何持续出现都应先定位基础设施或 Reward 问题；
6. entropy、PPO KL、clip fraction、grad norm 和 response length 都是有限值；
7. step 10/20 的 deterministic validation 没有相对 step 0 持续退化；
8. rollout 中开始出现不同搜索 query、不同候选或不同终局，而不是同题四条完全相同。

若 all-equal group 占比长期很高，优先研究 rollout 多样性或任务采样，不要直接打开
复杂信用分配。若 upper clip fraction 和 entropy collapse 同时明显，再单独试
`clip_ratio_high=0.28`。

### 10.6 运行 100-step 主实验

Smoke 通过后使用全新输出目录运行 100 step，并每 25 step 保存和验证：

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-qwen35-2b-curriculum/stage-c/merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/grpo-qwen35-2b-stage-c-main \
  --experiment-name qwen35-2b-stage-c-main \
  --logger swanlab -- \
  trainer.total_training_steps=100 \
  trainer.save_freq=25 \
  trainer.test_freq=25
```

不建议第一轮直接运行默认 500 step。历史结果的代表 checkpoint 也在 step 100；先用
100 step 判断 Reward、有效 group 比例、策略熵和外部开发集是否真的改善，再决定是否
扩大预算。

### 10.7 Stage B 对照

在主线 smoke 正常后，可给 Stage B 相同的 20 或 50 step 预算，仅替换模型和输出目录：

```bash
bash scripts/grpo.sh \
  --model outputs/models/sft-qwen35-2b-curriculum/stage-b/merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/grpo-qwen35-2b-stage-b-control \
  --experiment-name qwen35-2b-stage-b-control \
  --logger swanlab -- \
  trainer.total_training_steps=20 \
  trainer.save_freq=10 \
  trainer.test_freq=10
```

这个对照回答的是“Stage C 的 strategy exposure 是否带来更好的 RL 可学习性”，而不是
再次比较两个 SFT 的初始分数。

## 11. Checkpoint 导出与选择

不要默认选择最后一步。先根据 50 题 GRPO validation 排除明显退化的 checkpoint，
再将 step 25/50/75/100 候选导出：

```bash
bash scripts/export_grpo.sh \
  outputs/models/grpo-qwen35-2b-stage-c-main/global_step_50/actor \
  outputs/models/grpo-qwen35-2b-stage-c-step50-merged
```

在一个终端启动导出模型：

```bash
SERVED_MODEL_NAME=grpo-stage-c-step50 \
  bash scripts/serve_model.sh outputs/models/grpo-qwen35-2b-stage-c-step50-merged
```

在另一个终端对清洗后与 GRPO train 零重叠的 dev119 运行确定性评估：

```bash
RUN_DIR=outputs/evaluation/dev119/grpo-stage-c-step50
mkdir -p "$RUN_DIR"

.venv/bin/python scripts/evaluate_shop_benchmark.py \
  --benchmark outputs/evaluation/dev119/tasks.jsonl \
  --output "$RUN_DIR/trajectories.jsonl" \
  --summary "$RUN_DIR/summary.json" \
  --base-url http://127.0.0.1:5700 \
  --model grpo-stage-c-step50 \
  --llm-base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --max-steps 35 \
  --temperature 0 \
  --top-p 1 \
  --max-tokens 512

.venv/bin/python scripts/build_eval_report.py --run-dir "$RUN_DIR"
```

Checkpoint 选择顺序：

1. dev119 严格成功数；
2. 若严格成功相同，比较 mean final Reward 和 weighted score；
3. 再比较 Reward valid rate、wrong purchase、invalid-action-limit、Guard rejection；
4. 对成功翻转任务做配对检查，避免仅看边际比例；
5. 若所有候选都没有超过 Stage C 的 `79/119`，保留 SFT checkpoint，不以训练完成
   代替能力提升。

## 12. Final-200 使用边界

Final-200 不用于 checkpoint 选择、超参数选择或失败驱动迭代。只有在以下内容冻结后才
运行一次：

- 清洗后的 GRPO 数据及其 hash；
- 初始 SFT checkpoint；
- GRPO 配置；
- 选定 global step；
- 导出的 merged actor；
- 确定性评估协议。

冻结后启动模型并执行：

```bash
EVAL_OUTPUT_DIR=outputs/evaluation/grpo-qwen35-2b-selected \
SERVED_MODEL_NAME=grpo-qwen35-2b-selected \
  bash scripts/evaluate.sh grpo-qwen35-2b-selected
```

正式报告应同时保留 SFT Stage B、SFT Stage C 和 selected GRPO 的配对结果，避免只报告
最终单点数字。

## 13. 推荐决策清单

按优先级执行：

1. [已完成] 过滤 GRPO train，消除 6 个重叠 task，并更新 metadata/hash；
2. 用 Stage C 跑 20-step vanilla GRPO smoke；
3. 审计 dynamic sampling、entropy、clip、错误率和 rollout 多样性；
4. smoke 健康后跑独立的 100-step 主实验；
5. 用 GRPO validation 初筛、干净 dev119 配对选择 checkpoint；
6. 用同预算 Stage B control 判断初始化可塑性；
7. 只有诊断支持时才做 Clip-Higher、TRACE 或长度 shaping 单变量消融；
8. 冻结全部决策后，最后运行一次 Final-200。

## 14. 产物索引

- [Stage A summary](../../outputs/evaluation/dev119/sft-stage-a/summary.json)
- [Stage B summary](../../outputs/evaluation/dev119/sft-stage-b/summary.json)
- [Stage C summary](../../outputs/evaluation/dev119/sft-stage-c/summary.json)
- [Stage A HTML report](../../outputs/evaluation/dev119/sft-stage-a/report.html)
- [Stage B HTML report](../../outputs/evaluation/dev119/sft-stage-b/report.html)
- [Stage C HTML report](../../outputs/evaluation/dev119/sft-stage-c/report.html)
- [Stage A train summary](../../outputs/models/sft-qwen35-2b-curriculum/stage-a/adapter/train_summary.json)
- [Stage B train summary](../../outputs/models/sft-qwen35-2b-curriculum/stage-b/adapter/train_summary.json)
- [Stage C train summary](../../outputs/models/sft-qwen35-2b-curriculum/stage-c/adapter/train_summary.json)
- [课程曝光审计](../../data/sft_curriculum/README.md)
- [GRPO 配置说明](../grpo.md)
