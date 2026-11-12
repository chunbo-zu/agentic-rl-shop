"""veRL adapter for the project-owned SAO direct-IS loss.

veRL 0.8's bypass mode already routes ``old_log_probs`` to the probabilities
recorded by the rollout worker.  This adapter keeps that inexpensive path and
replaces only its policy objective when the typed ``loss_type="sao_dis"``
sentinel is present. The default veRL bypass loss remains untouched for the
existing GRPO recipe.
"""

from __future__ import annotations

from typing import Any


def freeze_critic_attention(module) -> int:
    """Freeze attention parameters in a value model before FSDP wrapping."""

    frozen = 0
    # Qwen3.5 has both ``linear_attn`` and ``self_attn`` blocks.  SAO freezes
    # the full-attention blocks and leaves the linear mixer/MLP trainable.
    attention_fragments = ("self_attn", "self_attention", ".attention")
    for name, parameter in module.named_parameters():
        if any(fragment in name.lower() for fragment in attention_fragments):
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    if frozen == 0:
        raise RuntimeError("SAO frozen-attention critic found no attention parameters")
    return frozen


def _sao_dis_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode: str = "token-mean",
    config=None,
    rollout_is_weights=None,
):
    if config is None:
        raise ValueError("SAO DIS requires the veRL actor config")
    correction = config.policy_loss.get("rollout_correction", {})
    low_clip = float(
        correction.get("sao_low_clip", config.get("clip_ratio_low", 0.2))
    )
    high_clip = float(
        correction.get("sao_high_clip", config.get("clip_ratio_high", 0.2))
    )

    from verl.trainer.ppo.core_algos import agg_loss
    from verl.utils import torch_functional as verl_F
    from shopping_grpo.training.grpo.sao import (
        dis_metrics,
        sao_dis_policy_loss,
    )

    loss_terms, keep, ratio = sao_dis_policy_loss(
        current_log_probs=log_prob,
        rollout_log_probs=old_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        low_clip=low_clip,
        high_clip=high_clip,
    )
    pg_loss = agg_loss(
        loss_mat=loss_terms,
        loss_mask=keep,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )
    metrics: dict[str, Any] = dis_metrics(ratio, keep, response_mask)
    metrics["actor/ppo_kl"] = verl_F.masked_mean(
        -(log_prob - old_log_prob), response_mask
    ).detach().item()
    metrics["actor/pg_clipfrac"] = 1.0 - metrics["sao/dis_keep_ratio"]
    metrics["actor/pg_clipfrac_lower"] = 0.0
    return pg_loss, metrics


def install_sao_policy_loss() -> None:
    """Install a guarded bypass-mode dispatch in each veRL worker process."""

    from verl.trainer.ppo import core_algos

    registry = core_algos.POLICY_LOSS_REGISTRY
    original = registry.get("bypass_mode")
    if original is None:
        raise RuntimeError("veRL bypass_mode policy loss is unavailable")
    if getattr(original, "_shopping_sao_wrapper", False):
        return

    def bypass_mode_with_sao(*args, **kwargs):
        config = kwargs.get("config")
        if config is None and len(args) >= 6:
            config = args[5]
        correction = (
            getattr(config, "policy_loss", {}).get("rollout_correction", {})
            if config
            else {}
        )
        if correction.get("sao_dis", False) or correction.get("loss_type") == "sao_dis":
            return _sao_dis_loss(*args, **kwargs)
        return original(*args, **kwargs)

    bypass_mode_with_sao._shopping_sao_wrapper = True
    registry["bypass_mode"] = bypass_mode_with_sao


def install_sao_gae() -> None:
    """Use the project-owned skip-observation GAE path for SAO batches."""

    from verl.trainer.ppo import core_algos, ray_trainer
    # ``SeparateRayPPOTrainer`` imports ``compute_advantage`` into its own
    # module namespace.  Patching only ray_trainer.compute_advantage therefore
    # misses fully_async_policy; patch both call sites while keeping the
    # wrapper idempotent for Ray worker setup hooks.
    modules = [ray_trainer]
    try:
        from verl.experimental.separation import ray_trainer as separation_ray_trainer

        modules.append(separation_ray_trainer)
    except ImportError:
        pass

    def compute_advantage_with_sao(original, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and len(args) >= 7:
            config = args[6]
        correction = config.get("rollout_correction", {}) if config is not None else {}
        estimator = kwargs.get("adv_estimator")
        if estimator is None and len(args) >= 2:
            estimator = args[1]
        is_gae = estimator == core_algos.AdvantageEstimator.GAE or str(estimator).lower() in {
            "gae",
            "advantageestimator.gae",
        }
        sao_enabled = correction.get("sao_dis", False) or correction.get("loss_type") == "sao_dis"
        if not sao_enabled or not is_gae:
            return original(*args, **kwargs)

        data = kwargs.get("data")
        if data is None and args:
            data = args[0]
        gamma = kwargs.get("gamma", args[2] if len(args) >= 3 else 1.0)
        lam = kwargs.get("lam", args[3] if len(args) >= 4 else 1.0)
        from shopping_grpo.training.grpo.sao import (
            align_rewards_to_action_tokens,
            skip_observation_gae,
        )
        from verl.utils import torch_functional as verl_F

        response_mask = data.batch["response_mask"]
        token_rewards = align_rewards_to_action_tokens(
            data.batch["token_level_rewards"], response_mask
        )
        advantages, returns = skip_observation_gae(
            token_rewards,
            data.batch["values"],
            response_mask,
            gamma=float(gamma),
            lam=float(lam),
        )
        data.batch["advantages"] = verl_F.masked_whiten(
            advantages, response_mask
        )
        data.batch["returns"] = returns
        return data

    for module in modules:
        original = module.compute_advantage
        if getattr(original, "_shopping_sao_wrapper", False):
            continue

        def wrapped(*args, _original=original, **kwargs):
            return compute_advantage_with_sao(_original, *args, **kwargs)

        wrapped._shopping_sao_wrapper = True
        module.compute_advantage = wrapped


def install_sao_critic_freezing() -> None:
    """Freeze value-model attention for the explicit SAO recipe only."""

    import os

    if not os.environ.get("GRPO_CONFIG_NAME", "").startswith("sao"):
        return
    from verl.workers.engine.fsdp import transformer_impl

    original = transformer_impl.FSDPEngine._build_module
    if getattr(original, "_shopping_sao_wrapper", False):
        return

    def build_module_with_frozen_attention(self, *args, **kwargs):
        module = original(self, *args, **kwargs)
        if self.model_config.get("model_type") == "value_model":
            frozen = freeze_critic_attention(module)
            if getattr(self, "rank", 0) == 0:
                print(f"SAO frozen-attention critic parameters: {frozen}")
        return module

    build_module_with_frozen_attention._shopping_sao_wrapper = True
    transformer_impl.FSDPEngine._build_module = build_module_with_frozen_attention
