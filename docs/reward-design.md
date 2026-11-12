# ShopSimulator Reward v3 设计说明

本文说明 `shopping-rl` 中 Reward v3 的设计目标、任务约束编译、购买评分、终止惩罚、
训练接入和评测口径。Reward v3 是环境提供的确定性终局奖励；它不调用另一个 LLM 判断
商品是否正确。

## 1. Reward 要解决什么问题

购物 Agent 的结果不能只用“是否点击了 Buy Now”判断。一个有效奖励至少要回答：

1. 商品品类是否正确；
2. 选中的具体规格价格是否满足预算；
3. 品牌、型号、功能和选项满足了多少用户要求；
4. Agent 是否有足够证据完成购买或合理放弃；
5. Agent 是否循环、过早终止或耗尽步数。

因此 Reward v3 采用四个原则：

- **确定性**：同一环境、同一 goal 和同一购买状态得到同一结果；
- **硬门槛优先**：品类和预算失败时，不能用其他偏好匹配抵消；
- **允许合格替代品**：不把目标 ASIN 当成唯一正确答案；
- **证据和终止受约束**：无法验证的硬条件不能伪装成成功，重复探索不能无限刷进展。

实现的主要来源是：

- [`reward.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py)
- [`termination.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py)
- [`reward_features.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features.py)
- [`configs/environment.json`](../environments/ShopSimulator/shop_env/configs/environment.json)

## 2. 从任务生成固定 Reward features

在 Actor 开始行动前，环境已经把任务 instruction 和目标商品编译为固定 features。这个
过程不会根据 Actor 的行为改变需求。

典型字段包括：

```text
category
price_upper
expected_brand
expected_model
expected_core_functions
required_options_by_key
unresolved_option_requirements
reward_feature_version
```

这些字段的来源是任务标注和商品 Gold metadata：

- `category`：任务目标商品品类；
- `price_upper`：只从用户指令中的明确预算上限解析，不从 Gold 商品价格反推预算；
- `expected_brand`：指令中明确要求的品牌别名；
- `expected_model`：指令和目标商品元数据之间能确认的型号标识；
- `expected_core_functions`：任务的属性/功能要求；
- `required_options_by_key`：把颜色、尺寸、容量、套餐等要求映射到商品选项轴；
- `unresolved_option_requirements`：无法可靠映射到具体选项轴、但仍需检查的要求。

编译逻辑见 [`compile_reward_features()`](../environments/ShopSimulator/shop_env/web_agent_site/engine/reward_features.py:184)。
Reward features 在 rollout 前冻结，模型不能通过改写搜索词或动作改变它。

## 3. 购买结果的计算顺序

Agent 调用 `buy_now` 后，环境取得：

- 实际购买商品 `product`；
- 用户任务 `goal`；
- 当前选择的商品规格 `selected_options`；
- 所选 variant 的价格解析结果 `price_resolution`。

然后执行以下顺序：

```text
解析 variant 价格
  → 检查品类和预算两个 hard gate
  → 计算品牌/型号/功能/选项匹配度
  → 计算证据覆盖率
  → 判断 exact Gold 或合格替代品
  → 映射为 reward_type 和 terminal_utility
```

入口是 [`evaluate_purchase()`](../environments/ShopSimulator/shop_env/web_agent_site/engine/reward.py:353)。

## 4. 两个 hard gate

### 4.1 品类门槛

购买商品必须通过 `compare_category(goal.category, product)`。品类不匹配时直接判定为
`wrong_purchase`，奖励为 `-0.85`。

### 4.2 预算门槛

如果任务明确声明了预算上限，环境会解析**当前已选择 variant 的实际价格**，并检查：

```text
actual_variant_price <= goal.price_upper
```

价格超过预算时是 `wrong_purchase`，奖励 `-0.85`。如果无法解析当前 variant 的价格，
结果是 `reward_unverifiable`，奖励 `0.0`，同时 `reward_valid=false`。

如果用户没有声明预算上限，预算门槛按“未声明预算”处理并通过；项目不会从目标商品的
Gold 价格偷偷推导一个用户没有说过的预算。

两个 hard gate 都通过后，才会进入偏好匹配计算。品牌、型号、功能和选项即使在任务中
很重要，在 Reward 实现里仍属于加权偏好维度，不能替代品类和预算门槛。

## 5. 四个偏好维度

Reward v3 对以下维度分别计算匹配分和证据覆盖率：

| 维度 | 权重 |
|---|---:|
| `brand` | 0.35 |
| `model` | 0.25 |
| `core_functions` | 0.25 |
| `key_options` | 0.15 |

### 5.1 单维度分数

对某个维度的要求集合，代码计算：

```text
dimension_score    = 通过要求数 / 要求总数
dimension_coverage = 可验证要求数 / 要求总数
```

没有该维度要求时，这个维度不参与归一化；没有任何激活维度时，整体匹配分和覆盖率都
视为 `1.0`。

具体比较器包括：

- 品牌：品牌别名和 `brand/shop_name` 文本匹配；
- 型号：型号标识比较；
- 核心功能：商品属性和任务功能要求比较；
- 关键选项：当前实际选择的颜色、尺寸、容量、套餐等与 required options 比较。

### 5.2 加权匹配分和覆盖率

设激活维度集合为 `A`，对应权重为 `w_i`，则：

```text
S_match = Σ(w_i × score_i) / Σ(w_i),  i ∈ A
S_cover = Σ(w_i × coverage_i) / Σ(w_i),  i ∈ A
```

只有同时满足：

```text
S_match == 1.0
S_cover == 1.0
```

才算“全部偏好满足”。代码使用浮点近似比较 `1.0`。

`S_match` 会作为 `weighted_score` 和 `terminal_utility` 诊断的一部分写入终局信息；
`S_cover` 会写入 `evidence_coverage`。

## 6. Exact Gold 与合格替代品

当两个 hard gate 通过，且所有激活偏好都满足时，代码再比较购买商品的 ASIN：

```python
asin_match = purchased_product.asin == goal.asin
```

- ASIN 与目标一致：`gold_purchase`，奖励 `1.00`；
- ASIN 不同但所有约束满足：`valid_alternative_purchase`，奖励 `0.55`。

这保证了评测不会把“商品目录里的唯一 ID”误当成用户唯一可接受答案。严格成功率仍然
只统计 `gold_purchase`，而购买成功率可以包含 `valid_alternative_purchase`。

## 7. 终局类型和奖励映射

Reward v3 的固定值如下：

| `reward_type` | 条件 | 奖励 | `reward_valid` |
|---|---|---:|---|
| `gold_purchase` | 目标 ASIN、hard gate、全部偏好和证据都满足 | `1.00` | true |
| `valid_alternative_purchase` | 非目标 ASIN，但 hard gate、全部偏好和证据都满足 | `0.55` | true |
| `partial_alternative_purchase` | hard gate 通过，但偏好只部分满足 | `min(0.25, -0.30 + 0.55 × S_match)` | true |
| `graceful_stop` | 搜索和候选核验充分，且没有已知可接受候选 | `-0.15` | true |
| `early_abstain` | 搜索/核验不足就放弃 | `-0.35` | true |
| `max_steps` | 达到最大步数 | `-0.50` | true |
| `repeat_loop` | 重复动作或连续无进展 | `-0.65` | true |
| `wrong_purchase` | 品类错误或超过预算 | `-0.85` | true |
| `reward_unverifiable` | hard gate 无法由环境证据验证 | `0.00` | **false** |

其中：

- `purchase_success=true` 只对应 `gold_purchase` 和 `valid_alternative_purchase`；
- `partial_alternative_purchase` 是有效但不完整的购买结果，不算购买成功；
- `reward_unverifiable` 的 0 分不是“中性成功”，而是不可用于正常学习的无效信号；
- `termination_reason` 必须与 `reward_type` 相同。

## 8. 部分满足购买的公式

部分购买使用整体偏好匹配分 `S_match`：

```text
R_partial = min(0.25, -0.30 + 0.55 × S_match)
```

例如 `S_match = 0.70` 时：

```text
R_partial = min(0.25, -0.30 + 0.55 × 0.70)
          = 0.085
```

这个结果说明：商品通过了品类和预算门槛，但只部分满足品牌、型号、功能或规格要求。

## 9. 放弃购买什么时候是合理的

`finish_without_purchase` 不会自动得到正奖励。环境要求同时满足：

```text
effective_result_sets >= 2
opened_candidates >= 2
known_acceptable_candidates == 0
```

其中“已知可接受候选”需要满足：

```text
hard gates 全部通过
S_match >= 0.70
S_cover >= 0.75
```

满足这些条件才是 `graceful_stop (-0.15)`；否则是 `early_abstain (-0.35)`。

这使 Agent 可以在确实找不到合适商品时停止，但不能只搜索一次就放弃，也不能在已经
存在明显合格候选时逃避购买。

## 10. 循环、无进展和最大步数

环境维护 `EvidenceProgressTracker`，记录动作签名、搜索结果集、打开的商品、详情子页、
规格选择和约束证据。

### 10.1 循环终止

出现以下任一条件时，终局类型为 `repeat_loop`：

- 连续两次完全相同的动作；
- 连续四步没有新增运行时证据。

运行时证据包括新结果集、新 ASIN、新商品详情、新详情子页、新选项或新的约束证据。

### 10.2 最大步数

默认最大环境步数为 35。达到上限后返回 `max_steps (-0.50)`。

证据奖励有上限：结果集、打开商品和详情子页只能在有限次数内计入进展。这样不能通过
反复翻页或重复查看同一页面无限延长轨迹。

实现见 [`termination.py`](../environments/ShopSimulator/shop_env/web_agent_site/engine/termination.py:40)。

## 11. RewardResult 中保存什么

环境终局会返回版本化的 `reward_detail`。核心字段包括：

```json
{
  "reward_version": "shopsimulator-reward-v3",
  "reward": 1.0,
  "terminal_utility": 1.0,
  "reward_type": "gold_purchase",
  "reward_valid": true,
  "termination_reason": "gold_purchase",
  "target_asin_match": true,
  "purchase_success": true,
  "hard_gates": {...},
  "weighted_score": 1.0,
  "evidence_coverage": 1.0,
  "dimension_scores": {
    "brand": 1.0,
    "model": 1.0,
    "core_functions": 1.0,
    "key_options": 1.0
  }
}
```

同时保存比较器版本、variant price 版本、价格解析过程和每个维度的明细，方便审计。

## 12. Reward 如何进入 GRPO

### 12.1 只有终局 Reward 进入训练

ShopSimulator 的中间 step reward 不直接作为 GRPO 的训练信号。AgentLoop 等待环境终局，
校验以下条件后才接受 Reward v3：

- `done=true` 且 `over=true`；
- reward version 正确；
- reward type 属于固定集合；
- `termination_reason == reward_type`；
- hard gate、weighted score、coverage 和各维度分数格式合法；
- `terminal_utility` 是有限值，并且与环境最终 reward 一致。

如果服务器返回结构不合法、环境异常或基础设施失败，轨迹标记为
`infrastructure_invalid`，不会被制造成一个普通负奖励。

校验入口是 [`runtime.py`](../src/shopping_grpo/training/grpo/adapter/runtime.py:113)。

### 12.2 两种 reward mode

AgentLoop 支持：

| 模式 | 训练使用的标量 |
|---|---|
| `native` | 正常终局的环境原生 `final_reward` |
| `constraint_aware` | Reward v3 的 `terminal_utility`，同时记录 hard gate、匹配度和覆盖率 |

当前配置使用 `reward_mode: constraint_aware`，见
[`configs/agent_loop.yaml`](../configs/agent_loop.yaml)。

`constraint_aware` 模式下，无效或不可验证轨迹的训练总分为 0，并设置
`sampling_invalid=true`。这与 `reward_unverifiable` 的语义一致：样本被记录和审计，但不
应被当作可靠的正负学习信号。

### 12.3 可选的长度整形

训练侧可以额外启用 length shaping：

```text
超过 20 步后，每步扣 0.01，最多扣 0.15
```

它只改变训练使用的 `total` 和 `terminal_utility`，不会修改环境原始 Reward v3。达到
最大步数会设置 `overlong=true` 和 `sampling_invalid=true`。

长度整形默认关闭，可通过配置开启。实现见
[`apply_reward_length_shaping()`](../src/shopping_grpo/training/grpo/adapter/runtime.py:310)。

### 12.4 写入 veRL 的方式

AgentLoop 将每条 trajectory 的训练标量写入 `output.reward_score`，并将详细诊断写入
`output.extra_fields["shopping"]`。之后 veRL 将它放入序列奖励结构，通常表现为：

```text
[0, 0, ..., 0, R]
```

工具 Observation 会影响上下文，但不会成为 actor action loss 的目标 token。GRPO 使用这条
trajectory 级标量形成组内相对优势，不需要额外 learned reward model。

## 13. 动态采样如何使用 Reward

每个 prompt 默认生成 4 条 rollout。动态采样按 prompt 的 group 聚合：

1. 读取每条轨迹的 `terminal_utility`、购买成功和 invalid 标记；
2. 如果一个 group 内所有 Reward 几乎相同，标记为 `constant_reward`；
3. 无效、不可验证或 overlong 的 group 记录原因；
4. 只保留有奖励差异且满足有效性条件的完整 group。

它的目的不是挑出最高分的一条轨迹，而是避免没有相对学习信号的 group 进入更新。相关
实现见 [`dynamic_sampling.py`](../src/shopping_grpo/training/grpo/dynamic_sampling.py)。

## 14. Reward 与 Rubric/Judge 的关系

Reward 和离线 Rubric/Judge 是两个独立面板：

| 组件 | 负责回答 | 是否使用 LLM |
|---|---|---|
| Reward v3 | 环境证据下最终商品/规格是否满足结构化任务 | 否 |
| Rubric | Query 明确要求了哪些需求 | Flash 只做候选筛选 |
| V4 Pro Judge | Actor 如何搜索、核验、决策和终止 | 是 |

Reward 判定为成功但 Rubric Judge 认为某条自然语言要求被违反时，系统记录
`reward_rubric_disagreement`，两个结果都保留，不用一个覆盖另一个。

## 15. 常见误读

- **Reward 不是点击 Buy Now 的奖励。** 购买后还要通过品类、预算、规格和证据检查。
- **品牌不是 hard gate。** Reward v3 的 hard gate 只有品类和预算；品牌进入加权偏好。
- **0 分不一定表示失败。** `reward_unverifiable` 的 0 分同时带有
  `reward_valid=false`，表示不能作为可靠训练信号。
- **替代品可以成功。** 只要 hard gate 和所有激活偏好都满足，非目标 ASIN 仍是
  `valid_alternative_purchase`。
- **Reward 只在终局结算。** 中间 Observation 中的临时 reward 不直接训练。
- **长度整形不是环境 Reward v3。** 它是训练侧可选的额外 penalty。
- **评测不把 Reward、Rubric 和 Judge 分数相加。** 三者回答不同问题，应分面板报告。

