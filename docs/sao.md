# ShopSimulator SAO arm

This arm follows [Single-Rollout Asynchronous Optimization for Agentic
Reinforcement Learning](https://arxiv.org/abs/2607.07508v1), adapted to the
ShopSimulator action/observation masks.

`configs/sao.yaml` is an independent recipe for SAO with an asynchronous
rollout server. It leaves `configs/grpo.yaml` unchanged and selects:

- one asynchronous rollout (`rollout.n=1`),
- direct double-sided importance sampling (DIS) from the rollout worker's
  `rollout_log_probs`, with a strict `(0.8, 1.2)` token support interval,
- GAE with action-token masks that bridge tool observations, and
- a value model with frozen attention and two PPO epochs for every actor epoch.

This recipe still uses the standard `main_ppo` batch barrier. It is useful for
isolating the SAO objective; it is not the fully asynchronous trainer.

For a genuine Rollouter/Trainer overlap, use
`configs/sao_fully_async.yaml`. It launches veRL's
`fully_async_policy` entrypoint with a separate rollout resource pool, a
streaming MessageQueue, NCCL weight synchronization, partial-rollout resume,
and `staleness_threshold=0.5`. The recipe requires at least two visible GPUs:
one trainer GPU and one rollout GPU. A positive staleness threshold is what
makes this different from a batch barrier; the Trainer consumes samples while
the Rollouter continues producing new samples.

The DIS adapter is installed by the existing Ray worker setup hook. In veRL
bypass mode, `old_log_probs` already points to the asynchronous rollout
probabilities, so the adapter does not maintain a historical policy or run a
second old-policy forward pass. The baseline bypass objective is used whenever
the typed `rollout_correction.loss_type` sentinel is not `sao_dis`.

The same hook freezes full-attention parameters in the critic before FSDP
wrapping; the value optimizer therefore updates the value head, linear mixer,
and MLP projections while preserving the pretrained full-attention
representation. Terminal
shopping rewards are moved from a final tool observation to its preceding
action token before skip-observation GAE, so a purchase or graceful stop is not
silently discarded by the action mask.

The project-owned numerical contracts are in
`shopping_grpo.training.grpo.sao`; `tests/test_sao.py` checks the ratio support
set, masked policy terms, and skip-observation GAE on CPU. veRL 0.8 already
contains the same skip-observation GAE recurrence; the helper makes the
contract explicit and testable without importing veRL.

先用 dry-run 检查命令与配置路径；正式启动时，训练入口会在加载模型前执行
SAO preflight：

不传 `--output` 时，`configs/sao.yaml` 默认输出到 `outputs/models/sao/`，
`configs/sao_fully_async.yaml` 默认输出到 `outputs/models/sao-fully-async/`。
Checkpoint、Hydra 运行日志和启用后的 SwanLab 本地日志均使用所选输出目录。
可通过 `--output` 为每次实验指定新的目录。

```bash
python scripts/train_grpo.py \
  --config configs/sao.yaml \
  --model /path/to/sft-merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/sao \
  --experiment-name shopping-agent-sao \
  --dry-run
```

The runtime preflight rejects group rollouts, disabled rollout log-probability
collection, incompatible advantage estimators, and a missing critic before a
training process is started.

To start the fully asynchronous recipe, keep ShopSimulator running in one
terminal and use the project launcher in another:

```bash
bash scripts/start_environment.sh

export SWANLAB_API_KEY=your_api_key

CUDA_VISIBLE_DEVICES=0,1 bash scripts/grpo.sh \
  --config configs/sao_fully_async.yaml \
  --fully-async \
  --model outputs/models/sft-curriculum/stage-c/merged \
  --train-data data/grpo/train.parquet \
  --val-data data/grpo/validation.parquet \
  --output outputs/models/sao-fully-async \
  --experiment-name shopping-agent-sao-fully-async \
  --logger swanlab
```

日志会写入输出目录下的 `swanlab/`。如果只做本地 smoke test，可将上面的 `--logger swanlab` 改为
`--logger console`。启动入口会自动设置 `SWANLAB_MODE=online`、日志目录并
执行 SwanLab 版本与配置检查。

Run a short smoke first by appending the following Hydra overrides after `--`:

```text
trainer.total_training_steps=20
rollout.total_rollout_steps=40
trainer.save_freq=10
trainer.test_freq=10
trainer.val_before_train=false
```

The launcher selects the project compatibility wrapper
`shopping_grpo.training.grpo.fully_async_main`, which delegates to veRL's
`fully_async_policy` entrypoint and fixes the veRL 0.8 critic-config field
layout before Ray workers are initialized. The existing
`scripts/serve_model.sh` is not needed; veRL starts the rollout server itself.

The compatibility hook intentionally waits for Ray's
`set_visible_accelerator_ids()` call before importing veRL/PyTorch. This is
required for multiple one-GPU asynchronous actors: inferring a device from
`RANK` can make every rollout replica initialize CUDA on physical GPU 0 and
then fail during NCCL checkpoint-group creation with
`NCCL_ERROR_INVALID_USAGE`. A four-GPU startup should therefore use Ray's
resource assignment directly, as in the command above with
`trainer.n_gpus_per_node=2 rollout.n_gpus_per_node=2` overrides when needed.
