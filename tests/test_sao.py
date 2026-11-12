"""CPU contracts for the project SAO math."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from shopping_grpo.training.grpo.sao import (
    align_rewards_to_action_tokens,
    direct_double_sided_mask,
    sao_dis_policy_loss,
    skip_observation_gae,
)


class SAOMathTest(unittest.TestCase):
    def test_dis_is_strict_and_masks_tool_observations(self):
        rollout = torch.zeros((1, 5))
        current = torch.log(torch.tensor([[1.0, 1.1, 0.9, 1.3, 1.0]]))
        response_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool)

        ratio, keep = direct_double_sided_mask(
            current,
            rollout,
            response_mask,
            low_clip=0.2,
            high_clip=0.2,
        )

        self.assertTrue(
            torch.allclose(ratio, torch.tensor([[1.0, 1.1, 0.9, 1.3, 1.0]]))
        )
        self.assertEqual(keep.tolist(), [[True, True, True, False, False]])

    def test_dis_loss_zeroes_rejected_and_observation_tokens(self):
        rollout = torch.zeros((1, 4))
        current = torch.log(torch.tensor([[1.0, 1.1, 1.3, 1.0]])).requires_grad_()
        advantages = torch.ones((1, 4))
        response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

        loss, keep, _ = sao_dis_policy_loss(
            current,
            rollout,
            advantages,
            response_mask,
            low_clip=0.2,
            high_clip=0.2,
        )
        self.assertEqual(keep.tolist(), [[True, True, False, False]])
        self.assertAlmostEqual(float(loss[0, 0].detach()), 0.0, places=6)
        self.assertAlmostEqual(
            float(loss[0, 1].detach()),
            -1.1 * float(torch.log(torch.tensor(1.1))),
            places=5,
        )
        self.assertAlmostEqual(
            float(loss[0, 2].detach()), -1.3 * float(torch.log(torch.tensor(1.3))), places=5
        )
        self.assertEqual(float(loss[0, 3].detach()), 0.0)

    def test_skip_observation_gae_bridges_to_next_action(self):
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.2, 9.0, 0.5]])
        response_mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)

        advantages, returns = skip_observation_gae(
            rewards,
            values,
            response_mask,
            gamma=1.0,
            lam=1.0,
        )

        self.assertTrue(torch.allclose(advantages, torch.tensor([[0.8, 0.0, 0.5]])))
        self.assertTrue(torch.allclose(returns, torch.tensor([[1.0, 0.0, 1.0]])))

    def test_terminal_observation_reward_is_attached_to_previous_action(self):
        rewards = torch.tensor([[0.0, 0.0, 2.0, 0.0, 1.0]])
        response_mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool)
        aligned = align_rewards_to_action_tokens(rewards, response_mask)
        self.assertTrue(
            torch.allclose(aligned, torch.tensor([[0.0, 3.0, 0.0, 0.0, 0.0]]))
        )

    def test_dis_rejects_invalid_bounds(self):
        values = torch.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, "low_clip"):
            direct_double_sided_mask(values, values, values.bool(), low_clip=1.0)
        with self.assertRaisesRegex(ValueError, "high_clip"):
            direct_double_sided_mask(values, values, values.bool(), high_clip=-0.1)

    def test_frozen_attention_selects_only_attention_parameters(self):
        from shopping_grpo.training.grpo.sao_policy import freeze_critic_attention

        module = nn.Module()
        module.self_attn = nn.Linear(2, 2)
        module.mlp = nn.Linear(2, 2)
        frozen = freeze_critic_attention(module)
        self.assertGreater(frozen, 0)
        self.assertTrue(all(not p.requires_grad for p in module.self_attn.parameters()))
        self.assertTrue(all(p.requires_grad for p in module.mlp.parameters()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
