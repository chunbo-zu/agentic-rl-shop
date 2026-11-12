"""veRL 的项目内 runtime 兼容行为。"""

import os
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class VerlCompatTest(unittest.TestCase):
    @staticmethod
    def _fake_ray(gpu_ids=("3",)):
        ray = ModuleType("ray")
        context = type(
            "RuntimeContext",
            (),
            {"get_accelerator_ids": lambda self: {"GPU": list(gpu_ids)}},
        )()
        ray.get_runtime_context = lambda: context
        return ray

    def test_worker_defers_verl_import_until_ray_gpu_assignment(self):
        ray = self._fake_ray(gpu_ids=("3",))
        setter_calls = []

        def set_visible_accelerator_ids(*args, **kwargs):
            setter_calls.append(os.environ["CUDA_VISIBLE_DEVICES"])
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"
            return {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}

        ray_utils = ModuleType("ray._private.utils")
        ray_utils.set_visible_accelerator_ids = set_visible_accelerator_ids
        ray_private = ModuleType("ray._private")
        ray_private.utils = ray_utils
        ray._private = ray_private

        modules = {
            "ray": ray,
            "ray._private": ray_private,
            "ray._private.utils": ray_utils,
        }
        with patch(
            "shopping_grpo.training.grpo.compat._install_torch_padding_fallback_impl"
        ) as install_impl, patch.dict(
            sys.modules, modules
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()
            self.assertEqual(setter_calls, [])
            install_impl.assert_not_called()
            ray_utils.set_visible_accelerator_ids()

            self.assertEqual(setter_calls, ["0,1,2,3"])
            install_impl.assert_called_once_with()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")

    def test_deferred_hook_installs_runtime_patches_after_assignment(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        ray = self._fake_ray(gpu_ids=("3",))
        ray_utils = ModuleType("ray._private.utils")

        def set_visible_accelerator_ids():
            os.environ["CUDA_VISIBLE_DEVICES"] = "3"

        ray_utils.set_visible_accelerator_ids = set_visible_accelerator_ids
        ray_private = ModuleType("ray._private")
        ray_private.utils = ray_utils
        ray._private = ray_private
        modules = {
            "ray": ray,
            "ray._private": ray_private,
            "ray._private.utils": ray_utils,
            "verl": verl,
            "verl.utils": utils,
            "verl.utils.attention_utils": attention,
            "verl.utils.npu_flash_attn_utils": fallback,
            "verl.trainer": trainer,
            "verl.trainer.ppo": ppo,
            "verl.trainer.ppo.ray_trainer": ray_trainer,
        }
        with patch(
            "shopping_grpo.training.grpo.compat.install_valuehead_attention_fallback"
        ), patch.dict(sys.modules, modules), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()

            ray_utils.set_visible_accelerator_ids()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(attention._get_attention_functions(), expected)

    def test_driver_preflight_keeps_cuda_visibility_unchanged(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_setup_hook_isolates_compiler_caches_per_ray_worker(self):
        import tempfile
        from pathlib import Path

        from shopping_grpo.training.grpo.compat import (
            _isolate_ray_worker_compile_caches,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
                    "VLLM_CACHE_ROOT": str(root / "vllm"),
                    "TRITON_CACHE_DIR": str(root / "triton"),
                    "TORCHINDUCTOR_CACHE_DIR": str(root / "torchinductor"),
                },
                clear=True,
            ), patch("os.getpid", return_value=12345):
                _isolate_ray_worker_compile_caches()

                triton = root / "triton/ray-worker-12345"
                inductor = root / "torchinductor/ray-worker-12345"
                vllm = root / "vllm/ray-worker-12345"
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(triton))
                self.assertEqual(
                    os.environ["TORCHINDUCTOR_CACHE_DIR"],
                    str(inductor),
                )
                self.assertEqual(os.environ["VLLM_CACHE_ROOT"], str(vllm))
                self.assertTrue(triton.is_dir())
                self.assertTrue(inductor.is_dir())
                self.assertTrue(vllm.is_dir())

                _isolate_ray_worker_compile_caches()
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(triton))

    def test_setup_hook_does_not_guess_gpu_from_local_rank(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            sys.modules,
            {"ray": self._fake_ray(gpu_ids=())},
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d",
                "RANK": "6",
                "RAY_LOCAL_WORLD_SIZE": "4",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b,GPU-c,GPU-d")

    def test_setup_hook_keeps_non_contiguous_visibility_until_ray_assignment(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            sys.modules,
            {"ray": self._fake_ray(gpu_ids=())},
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "1, 3",
                "RANK": "1",
                "RAY_LOCAL_WORLD_SIZE": "2",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1, 3")

    def test_setup_hook_without_verl_rank_metadata_keeps_visibility(self):
        from shopping_grpo.training.grpo.compat import _pin_ray_assigned_cuda_device

        with patch.dict(
            sys.modules,
            {"ray": self._fake_ray(gpu_ids=())},
        ), patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR": "compat-hook",
            },
            clear=True,
        ):
            _pin_ray_assigned_cuda_device()

            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        with patch(
            "shopping_grpo.training.grpo.compat.install_valuehead_attention_fallback"
        ), patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
                "verl.trainer": trainer,
                "verl.trainer.ppo": ppo,
                "verl.trainer.ppo.ray_trainer": ray_trainer,
            },
        ):
            from shopping_grpo.training.grpo.compat import install_torch_padding_fallback

            install_torch_padding_fallback()

        self.assertEqual(attention._get_attention_functions(), expected)
        self.assertTrue(RayPPOTrainer._update_actor._shopping_trace)

    def test_fully_async_critic_compat_uses_legacy_fsdp_fields(self):
        import types

        separation = types.ModuleType("verl.experimental.separation.ray_trainer")

        class SeparateRayPPOTrainer:
            def _create_critic_class(self):  # pragma: no cover - replaced below
                raise AssertionError("original implementation should be replaced")

        separation.SeparateRayPPOTrainer = SeparateRayPPOTrainer
        role = types.SimpleNamespace(Critic="critic")
        ray_module = types.ModuleType("verl.single_controller.ray")
        ray_module.RayClassWithInitArgs = lambda **kwargs: kwargs
        utils_config = types.ModuleType("verl.utils.config")
        workers_config = types.ModuleType("verl.workers.config")
        workers_engine = types.ModuleType("verl.workers.engine_workers")

        class FSDPEngineConfig:
            pass

        class TrainingWorkerConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        workers_config.FSDPEngineConfig = FSDPEngineConfig
        workers_engine.TrainingWorkerConfig = TrainingWorkerConfig
        utils_config.omega_conf_to_dataclass = lambda config: config

        modules = {
            "verl": types.ModuleType("verl"),
            "verl.experimental": types.ModuleType("verl.experimental"),
            "verl.experimental.separation": types.ModuleType(
                "verl.experimental.separation"
            ),
            "verl.single_controller": types.ModuleType("verl.single_controller"),
            "verl.trainer": types.ModuleType("verl.trainer"),
            "verl.trainer.ppo": types.ModuleType("verl.trainer.ppo"),
            "verl.utils": types.ModuleType("verl.utils"),
            "verl.workers": types.ModuleType("verl.workers"),
            "verl.experimental.separation.ray_trainer": separation,
            "verl.single_controller.ray": ray_module,
            "verl.trainer.ppo.utils": types.SimpleNamespace(Role=role),
            "verl.utils.config": utils_config,
            "verl.workers.config": workers_config,
            "verl.workers.engine_workers": workers_engine,
        }
        critic_model = object()
        engine = types.SimpleNamespace()
        critic_config = types.SimpleNamespace(
            strategy="fsdp",
            engine=engine,
            model=critic_model,
            optim=object(),
            checkpoint=object(),
            ppo_infer_max_token_len_per_gpu=11,
            ppo_max_token_len_per_gpu=22,
        )

        class ResourcePools:
            def get_resource_pool(self, name):
                self.role = name
                return "critic-pool"

        trainer = types.SimpleNamespace(
            use_critic=True,
            resource_pool_manager=ResourcePools(),
            config=types.SimpleNamespace(critic=critic_config),
            role_worker_mapping={"critic": "critic-worker"},
            resource_pool_to_cls={"critic-pool": {}},
        )

        with patch.dict(sys.modules, modules):
            from shopping_grpo.training.grpo.compat import (
                install_fully_async_critic_compat,
            )

            install_fully_async_critic_compat()
            SeparateRayPPOTrainer._create_critic_class(trainer)
            registered = trainer.resource_pool_to_cls["critic-pool"]["critic"]
            self.assertEqual(registered["config"].model_config, critic_model)
            self.assertIs(registered["config"].engine_config, engine)
            self.assertEqual(engine.infer_max_token_len_per_gpu, 11)
            self.assertEqual(engine.max_token_len_per_gpu, 22)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
