# 单任务目标可达性检查

`scripts/check_target_reachability.py` 对冻结任务池中的一个 `task_id` 做程序化检查，
不调用模型。它使用私有目标 ASIN 和规格规划路径，是目标已知的环境审计，不能当成
教师或学生模型的成功率，也不能把报告或路径直接混入训练集。

在仓库根目录运行，Python 3.10+ 即可；离线检查不需要启动 ShopSimulator。

## 1. 检查目标规格、实际价格和 Reward

```bash
python scripts/check_target_reachability.py --task-id 0 --output outputs/reachability/task-0-static.json
```

脚本验证解压后的商品数据 SHA-256，按环境过滤商品、枚举 instruction 的顺序解析任务，
调用 `compile_reward_features`、`candidate_options_for_evaluation`、
`resolve_variant_price` 和 `evaluate_purchase`。可用 `--expected-asin ASIN` 检查任务映射。
它不使用手填价格覆盖真实 variant 价格。报告包含所选规格、价格证据、预算和 Reward 明细。

这里只检查一个确定性选择的 variant，不穷举所有组合；通过表示存在静态评分见证，
失败也不排除其他 variant 成功。检查遵循项目 Reward v3 的需求解析口径，不能证明
Reward 已完整理解自然语言中的每一项要求。

## 2. 检查普通搜索能否召回目标

如果已有环境索引，可独立检查 top-150 排名与每页 20 条对应的页码：

```bash
python scripts/check_target_reachability.py --task-id 0 --query "儿童 天然乳胶 枕头" --index environments/ShopSimulator/shop_env/search_engine/products.sqlite3
```

索引必须与冻结商品 SHA-256、BM25 版本和字段权重匹配。支持重复传入 `--query`。
查询由调用者提供，脚本不自动拼接私有目标标题；禁止 `<a>` 等环境内部查询捷径。
某个 query 没命中只表示该 query 未找到目标。目标进入 top-150 也不保证能在步骤和
重复动作限制内翻到对应页面。

## 3. 通过合法工具路径取得可达性见证

ShopSimulator 已在运行时执行；此模式不需要本地搜索索引：

```bash
python scripts/check_target_reachability.py --task-id 0 --query "儿童 天然乳胶 枕头" --query "乳胶枕 满天星" --replay --base-url http://127.0.0.1:5700 --output outputs/reachability/task-0-replay.json
```

每个 query 使用独立环境租约，最多 35 步，可用 `--max-steps` 降低上限。
路径为 `search_products → next_page（需要时）→ open_product → select_option → buy_now`。
所有动作先经过项目的 observation 渲染器和动作守卫；只能打开当前页的 ASIN、选择当前
页面的规格值。脚本核对服务端 task 的 instruction、目标 ASIN、目标规格和环境版本，
保留每步工具名、参数、动作、结构化 observation 与最终结果，并释放环境租约。

只有 `done=true`、`over=true`，且完整 Reward v3 结果通过验证、`reward_valid=true`、
`purchase_success=true`、`target_asin_match=true`、`reward_type=gold_purchase`，才算成功。
服务器的循环和无进展终止照常生效；这个简单路径不穷举搜索词、绕行路线或规格组合。
报告中的交互步数只计环境动作，不含 reset/release。

| status | 含义 | reachable |
|---|---|---|
| `static_pass` | 所选 variant 可得严格 gold，尚未验证交互路径 | `null` |
| `retrieval_pass` | 静态检查通过，至少一个 query 召回目标 | `null` |
| `reachable` | 至少一条合法路径取得严格 gold 终局 | `true` |
| `not_proven` | 当前 variant、查询或路径没能取得见证 | `null` |
| `error` | 数据版本、协议或环境连接等出错 | `null` |

退出码为 0（所请求的检查通过）、1（尚未证实）、2（检查错误）。自动程序若要确认
端到端可达，必须读取 `reachable == true`，不能仅检查退出码。已有输出默认不覆盖，
显式传 `--force` 才替换。

该回放验证环境与动作守卫下的可执行性，不调用 tokenizer、observation token 投影或
模型上下文压缩；报告明确标记 `model_context_budget_checked=false`。因此它不代表
在某个教师/学生模型的 token 预算内也一定成功。若目标是全新的自然语言需求，需先定义
对应 goal 的约束和评分口径；本脚本只审计冻结任务池中的现有任务。
