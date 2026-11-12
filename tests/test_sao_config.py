"""Configuration gate tests for the independent SAO recipe."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.check_grpo_runtime import (
    compose_runtime_config,
    validate_fully_async,
    validate_sao,
)


class SAOConfigTest(unittest.TestCase):
    def _config(self, *overrides):
        with patch.dict(os.environ, {"GRPO_CONFIG_NAME": "sao"}, clear=False):
            return compose_runtime_config(list(overrides))

    def test_sao_recipe_enables_single_rollout_dis_and_gae(self):
        config = self._config()
        validate_sao(config)
        self.assertTrue(config.shopping_sao.enable)
        self.assertEqual(config.actor_rollout_ref.rollout.n, 1)
        self.assertEqual(config.algorithm.adv_estimator, "gae")
        self.assertEqual(config.algorithm.rollout_correction.loss_type, "sao_dis")
        self.assertEqual(config.critic.ppo_epochs, 2)
        self.assertTrue(config.shopping_sao.freeze_critic_attention)

        # The actor worker materializes its typed config before the driver
        # applies bypass mode; keep the project loss sentinel in that config.
        from verl.utils.config import omega_conf_to_dataclass

        actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        self.assertEqual(actor.policy_loss.loss_mode, "bypass_mode")
        self.assertEqual(actor.policy_loss.rollout_correction.loss_type, "sao_dis")

    def test_sao_rejects_group_rollouts(self):
        config = self._config("actor_rollout_ref.rollout.n=2")
        with self.assertRaisesRegex(SystemExit, "exactly one rollout"):
            validate_sao(config)

    def test_baseline_config_is_not_sao(self):
        with patch.dict(os.environ, {"GRPO_CONFIG_NAME": "grpo"}, clear=False):
            config = compose_runtime_config([])
        validate_sao(config)

    def test_fully_async_recipe_decouples_resources_and_bounds_staleness(self):
        with patch.dict(
            os.environ,
            {"GRPO_CONFIG_NAME": "sao_fully_async", "GRPO_FULLY_ASYNC": "1"},
            clear=False,
        ):
            config = compose_runtime_config([])
            validate_sao(config)
            validate_fully_async(config)

        self.assertFalse(config.actor_rollout_ref.hybrid_engine)
        self.assertTrue(config.actor_rollout_ref.actor.use_rollout_log_probs)
        self.assertEqual(config.actor_rollout_ref.rollout.checkpoint_engine.backend, "nccl")
        self.assertEqual(config.async_training.trigger_parameter_sync_step, 1)
        self.assertGreater(config.async_training.staleness_threshold, 0)
        self.assertEqual(config.rollout.n, 1)

    def test_fully_async_rejects_zero_staleness(self):
        with patch.dict(
            os.environ,
            {"GRPO_CONFIG_NAME": "sao_fully_async", "GRPO_FULLY_ASYNC": "1"},
            clear=False,
        ):
            config = compose_runtime_config(["async_training.staleness_threshold=0"])
            with self.assertRaisesRegex(SystemExit, "staleness_threshold"):
                validate_fully_async(config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
