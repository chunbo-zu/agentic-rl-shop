# Documentation

如果第一次接触项目，先读[shopping-rl 项目导读](project-guide.md)，再按下面的专项文档
深入具体阶段。

如果重点是理解 agent harness、veRL `ToolAgentLoop` 的继承关系以及从 trajectory 到
GRPO optimizer update 的代码数据流，继续读[Agent Harness 与训练实现](agent-harness-and-training.md)。

Follow the guides in workflow order:

1. [Data collection](data-collection.md) explains how the checked-in SFT data
   was produced and audited.
2. [SFT](sft.md) trains the first useful shopping agent.
   The current Qwen3.5-2B three-stage results and GRPO handoff are recorded in
   the [2026-09-16 SFT curriculum report](research/2026-09-16-qwen35-2b-sft-curriculum-report.md).
3. [GRPO](grpo.md) improves that model with online environment reward.
4. [SAO](sao.md) adds single-rollout correction and the fully asynchronous
   Rollouter/Trainer recipe.
5. [Evaluation](evaluation.md) compares baseline, SFT and GRPO fairly.
6. [Final-200 Clean evaluation dataset](evaluation-dataset.md) defines the current
   curated benchmark and its update record.

[Reward v3](reward-v3.md) is the detailed specification shared by collection,
GRPO and evaluation.
