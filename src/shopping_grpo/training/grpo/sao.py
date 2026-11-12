"""Small, dependency-light building blocks for Single-Rollout Asynchronous Optimization.

The training runtime uses these functions in two places:

* :func:`direct_double_sided_mask` defines the token-level DIS support set.  It
  compares the current policy with the log probabilities recorded by the
  asynchronous rollout worker and rejects tokens outside a strict two-sided
  ratio interval.
* :func:`skip_observation_gae` implements the GAE recurrence over action tokens
  while carrying the bootstrap value across tool observations.  In a
  ``ToolAgentLoop`` response mask, zeroes are environment observations and
  padding, so no credit is assigned to those tokens.

The module deliberately contains no veRL imports.  That makes the numerical
contract testable on CPU and keeps the project-owned part of SAO independent of
the installed veRL version.
"""

from __future__ import annotations

from typing import Any


def _validate_matching_shapes(*values: Any) -> None:
    shapes = {tuple(value.shape) for value in values}
    if len(shapes) != 1:
        raise ValueError(f"SAO tensors must have equal shapes, got {sorted(shapes)}")


def direct_double_sided_mask(
    current_log_probs,
    rollout_log_probs,
    response_mask,
    *,
    low_clip: float = 0.2,
    high_clip: float = 0.2,
):
    """Return the DIS ratio and strict token support mask.

    ``rollout_log_probs`` are the behavior-policy probabilities emitted by the
    asynchronous rollout worker.  The strict inequalities are intentional:
    boundary tokens are rejected, matching the paper's ``1-eps < r < 1+eps``
    support set.  ``response_mask`` removes tool observations and padding from
    both the objective and its diagnostics.
    """

    import torch

    _validate_matching_shapes(current_log_probs, rollout_log_probs, response_mask)
    if not 0 <= float(low_clip) < 1:
        raise ValueError("low_clip must be in [0, 1)")
    if not 0 <= float(high_clip):
        raise ValueError("high_clip must be non-negative")
    if not torch.isfinite(current_log_probs).all() or not torch.isfinite(rollout_log_probs).all():
        raise ValueError("SAO log probabilities must be finite")

    # A large log-ratio can overflow exp() after an asynchronous update.  The
    # clamp only protects the diagnostic ratio; out-of-range values are still
    # rejected by the support mask.
    log_ratio = torch.clamp(current_log_probs - rollout_log_probs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    keep = response_mask.to(torch.bool) & (ratio > 1.0 - float(low_clip)) & (
        ratio < 1.0 + float(high_clip)
    )
    return ratio, keep


def sao_dis_policy_loss(
    current_log_probs,
    rollout_log_probs,
    advantages,
    response_mask,
    *,
    low_clip: float = 0.2,
    high_clip: float = 0.2,
):
    """Compute unreduced SAO DIS policy-loss terms.

    The ratio is detached in the multiplier, as in the paper's clipped
    importance-sampling objective.  Keeping this function unreduced lets veRL
    apply its configured global token/sequence aggregation consistently.
    """

    import torch

    _validate_matching_shapes(
        current_log_probs, rollout_log_probs, advantages, response_mask
    )
    ratio, keep = direct_double_sided_mask(
        current_log_probs,
        rollout_log_probs,
        response_mask,
        low_clip=low_clip,
        high_clip=high_clip,
    )
    loss = -ratio.detach() * advantages * current_log_probs
    return loss, keep, ratio


def skip_observation_gae(
    token_level_rewards,
    values,
    response_mask,
    *,
    gamma: float = 1.0,
    lam: float = 1.0,
):
    """Compute GAE while bridging over non-action observation tokens.

    At an observation token (``response_mask == 0``) the running value and GAE
    state are left unchanged.  The next action therefore bootstraps from the
    last action's value chain rather than treating a long tool observation as a
    sequence of zero-reward transitions.  The returned advantages and returns
    are zero on masked positions.
    """

    import torch

    _validate_matching_shapes(token_level_rewards, values, response_mask)
    if not 0 <= float(gamma) <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not 0 <= float(lam) <= 1:
        raise ValueError("lam must be in [0, 1]")
    if token_level_rewards.ndim != 2:
        raise ValueError("SAO GAE expects [batch, response_length] tensors")

    mask = response_mask.to(torch.bool)
    rewards = token_level_rewards.to(dtype=values.dtype)
    values = values.to(dtype=rewards.dtype)
    advantages = torch.zeros_like(rewards)
    running_advantage = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    next_value = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)

    for index in range(rewards.shape[1] - 1, -1, -1):
        active = mask[:, index]
        # At an observation/padding position, preserve the next action's
        # bootstrap state.  At an action position, use the usual TD residual.
        delta = rewards[:, index] + float(gamma) * next_value - values[:, index]
        running_advantage = torch.where(
            active,
            delta + float(gamma) * float(lam) * running_advantage,
            running_advantage,
        )
        advantages[:, index] = torch.where(
            active, running_advantage, torch.zeros_like(running_advantage)
        )
        next_value = torch.where(active, values[:, index], next_value)

    returns = torch.where(mask, advantages + values, torch.zeros_like(values))
    return advantages, returns


def align_rewards_to_action_tokens(token_level_rewards, response_mask):
    """Move terminal rewards emitted on tool observations to the preceding action.

    The veRL reward loop places a terminal score on the last non-padding
    response token.  A shopping episode can end immediately after a tool call,
    so that token may be an environment observation and therefore have a zero
    action mask.  SAO must keep the score while still skipping observation
    tokens; this function transfers every masked reward to the closest previous
    action token in the same trajectory.
    """

    import torch

    _validate_matching_shapes(token_level_rewards, response_mask)
    if token_level_rewards.ndim != 2:
        raise ValueError("reward alignment expects [batch, response_length] tensors")
    aligned = token_level_rewards.clone()
    mask = response_mask.to(torch.bool)
    for row in range(aligned.shape[0]):
        active_positions = torch.nonzero(mask[row], as_tuple=False).flatten()
        if active_positions.numel() == 0:
            if torch.any(aligned[row] != 0):
                raise ValueError("cannot align a reward without an action token")
            continue
        for index in range(aligned.shape[1]):
            if mask[row, index]:
                continue
            reward = aligned[row, index]
            if reward == 0:
                continue
            previous = active_positions[active_positions < index]
            if previous.numel() == 0:
                raise ValueError("cannot align an observation reward before the first action")
            aligned[row, previous[-1]] += reward
            aligned[row, index] = 0
    return aligned


def dis_metrics(ratio, keep, response_mask) -> dict[str, float]:
    """Return scalar DIS diagnostics without depending on veRL aggregation."""

    import torch

    active = response_mask.to(torch.bool)
    active_count = int(active.sum().item())
    if active_count == 0:
        return {
            "sao/dis_keep_ratio": 0.0,
            "sao/dis_ratio_mean": 0.0,
            "sao/dis_ratio_max": 0.0,
        }
    active_ratios = ratio[active]
    return {
        "sao/dis_keep_ratio": float(keep[active].float().mean().item()),
        "sao/dis_ratio_mean": float(active_ratios.mean().item()),
        "sao/dis_ratio_max": float(active_ratios.max().item()),
    }
