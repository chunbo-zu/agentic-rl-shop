"""veRL 0.8 的窄范围运行时兼容。"""


def _isolate_ray_worker_compile_caches():
    """Prevent concurrent Ray workers from racing on compiler cache files.

    vLLM starts several Ray actors at once. If they share one Triton cache,
    TorchInductor autotuning can observe another process's partially-published
    kernel metadata and fail with a transient ``FileNotFoundError``. A
    per-worker directory keeps the caches on the configured data volume while
    removing that cross-process race. EngineCore children inherit the
    directory selected by their parent Ray worker.
    """
    import os
    from pathlib import Path

    if "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR" not in os.environ:
        return

    worker_directory = f"ray-worker-{os.getpid()}"
    for variable in (
        "VLLM_CACHE_ROOT",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    ):
        configured = os.environ.get(variable)
        if not configured:
            continue
        path = Path(configured).expanduser().absolute()
        if path.name != worker_directory:
            path /= worker_directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


def _pin_ray_assigned_cuda_device():
    """Compatibility no-op kept for callers of the old startup hook.

    Ray applies ``CUDA_VISIBLE_DEVICES`` in its worker loop *after* the
    process setup hook.  Guessing a device from ``RANK`` here is unsafe for
    the fully-async topology: every single-GPU rollout replica has rank zero,
    while Ray may place those replicas on different physical GPUs.  The
    actual fix is installed by :func:`_defer_runtime_hooks_until_ray_gpu`.
    """
    return None


def _defer_runtime_hooks_until_ray_gpu():
    """Run veRL imports immediately after Ray assigns the worker's GPU.

    ``worker_process_setup_hook`` runs before Ray calls
    ``set_visible_accelerator_ids``.  Importing veRL/PyTorch in that hook can
    therefore initialize CUDA against the driver's full device list.  Wrap
    Ray's private setter, let it finish first, and only then install the
    project runtime patches.  The wrapper is process-local and idempotent;
    later Ray tasks in a reused worker only execute the original setter.
    """
    import ray
    from ray._private import utils as ray_utils
    original = ray_utils.set_visible_accelerator_ids
    if getattr(original, "_shopping_deferred_runtime_hooks", False):
        return

    def set_visible_accelerator_ids_after_ray_assignment(*args, **kwargs):
        result = original(*args, **kwargs)
        _install_torch_padding_fallback_impl()
        return result

    set_visible_accelerator_ids_after_ray_assignment._shopping_deferred_runtime_hooks = (
        True
    )
    ray_utils.set_visible_accelerator_ids = (
        set_visible_accelerator_ids_after_ray_assignment
    )


def _install_trace_actor_update():
    """Score private gold targets before the existing veRL actor update."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._update_actor, "_shopping_trace", False):
        return

    from shopping_grpo.training.grpo.trace import (
        apply_trace_advantages,
        build_trace_score_batch,
    )

    original_update_actor = RayPPOTrainer._update_actor

    def update_actor_with_trace(self, batch):
        config = self.config.get("shopping_trace", {})
        enabled = bool(config.get("enable", False))
        if not enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            return original_update_actor(self, batch)
        if not self.ref_in_actor:
            raise RuntimeError("shopping TRACE requires LoRA so the frozen base model is available")

        score_batch, state_counts = build_trace_score_batch(
            batch,
            self.tokenizer,
            max_sequence_length=int(config["max_sequence_length"]),
        )
        reference = self._compute_ref_log_prob(score_batch)
        target_mask = score_batch.batch["response_mask"]
        mean_log_probs = (
            reference.batch["ref_log_prob"] * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1)
        batch.non_tensor_batch.pop("trace_target", None)
        trace_metrics = apply_trace_advantages(
            batch,
            mean_log_probs,
            state_counts,
            epsilon=float(config["epsilon"]),
            horizon=int(config["horizon"]),
            discount=float(config["discount"]),
            terminal_weight=float(config["terminal_weight"]),
            outcome_weight=float(config["outcome_weight"]),
            turn_weight=float(config["turn_weight"]),
        )
        trace_metrics["trace/target_log_prob_mean"] = float(mean_log_probs.mean())
        output = original_update_actor(self, batch)
        output.meta_info["metrics"].update(trace_metrics)
        return output

    update_actor_with_trace._shopping_trace = True
    RayPPOTrainer._update_actor = update_actor_with_trace


def install_fully_async_critic_compat():
    """Bridge veRL 0.8's separation trainer to its critic config schema.

    The published fully-async trainer reads ``critic.model.fsdp_config`` and
    ``critic.model_config``.  veRL 0.8's ``FSDPCriticConfig`` stores the model
    under ``critic.model`` and the engine under ``critic.engine`` instead.
    Keep the project recipe on the stable public config shape and adapt only
    this narrow worker-construction boundary.
    """
    from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer

    original = SeparateRayPPOTrainer._create_critic_class
    if getattr(original, "_shopping_fully_async_critic_compat", False):
        return

    def create_critic_class(self):
        if not self.use_critic:
            return

        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.trainer.ppo.utils import Role
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import FSDPEngineConfig
        from verl.workers.engine_workers import TrainingWorkerConfig

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        critic_config = omega_conf_to_dataclass(self.config.critic)
        self.orig_critic_cfg = critic_config
        if critic_config.strategy != "fsdp":
            raise NotImplementedError(
                f"Unknown strategy for fully-async critic: {critic_config.strategy=}"
            )

        # FSDPCriticConfig exposes the engine as ``engine`` (and ``fsdp``),
        # while newer separation code expects it below the model config.
        engine_config: FSDPEngineConfig = getattr(critic_config, "engine", None)
        if engine_config is None:
            engine_config = getattr(critic_config.model, "fsdp_config", None)
        if engine_config is None:
            raise AttributeError(
                "fully-async critic requires an FSDP engine config at "
                "critic.engine or critic.model.fsdp_config"
            )
        engine_config.infer_max_token_len_per_gpu = (
            critic_config.ppo_infer_max_token_len_per_gpu
        )
        engine_config.max_token_len_per_gpu = critic_config.ppo_max_token_len_per_gpu

        # TrainingWorker consumes the regular HF model config.  The upstream
        # separation trainer's ``model_config`` attribute is absent in veRL 0.8.
        worker_config = TrainingWorkerConfig(
            model_type="value_model",
            model_config=critic_config.model,
            engine_config=engine_config,
            optimizer_config=critic_config.optim,
            checkpoint_config=critic_config.checkpoint,
        )
        critic_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Critic],
            config=worker_config,
        )
        self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

    create_critic_class._shopping_fully_async_critic_compat = True
    SeparateRayPPOTrainer._create_critic_class = create_critic_class


def install_fully_async_trainer_compat():
    """Install veRL 0.8's missing async critic initialization compatibility."""
    from functools import partial

    from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from verl.trainer.ppo.utils import Role
    from verl.workers.utils.losses import value_loss

    # ``@ray.remote`` returns an ActorClass wrapper. Patching a method on that
    # wrapper does not change the class Ray serializes to the worker; the
    # executable class is stored in ``__ray_metadata__.modified_class``.
    target = getattr(
        getattr(FullyAsyncTrainer, "__ray_metadata__", None),
        "modified_class",
        FullyAsyncTrainer,
    )
    original = target._init_models
    if getattr(original, "_shopping_fully_async_trainer_compat", False):
        return

    def init_models(self):
        # veRL 0.8's SeparateRayPPOTrainer creates the critic with the unified
        # TrainingWorker, whose public initialization method is ``reset``.
        # The experimental async trainer calls ``init_model`` here, which is
        # only available on actor/rollout worker groups.
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            self.critic_wg.set_loss_fn(partial(value_loss, config=self.orig_critic_cfg))

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg

    init_models._shopping_fully_async_trainer_compat = True
    target._init_models = init_models


def install_valuehead_attention_fallback():
    """Use SDPA for veRL's value-head loader when FlashAttention is absent.

    veRL 0.8 hard-codes ``flash_attention_2`` in ``load_valuehead_model``.
    The actor recipe already selects SDPA, but the critic takes a separate
    value-model path and otherwise fails before training starts when the
    optional ``flash-attn`` wheel is not installed.  Keep this patch local to
    the critic loader so actor and rollout attention settings are unchanged.
    """
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        return

    try:
        from verl.utils import model as model_utils
    except (ImportError, ModuleNotFoundError, AttributeError):
        # Lightweight unit-test fakes do not provide veRL's model module.
        return

    original = model_utils.load_valuehead_model
    if getattr(original, "_shopping_sdpa_fallback", False):
        return

    def load_valuehead_model(local_path, torch_dtype, model_config, trust_remote_code):
        from transformers import AutoModelForCausalLM, AutoModelForTokenClassification

        attention_implementation = "sdpa"
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation=attention_implementation,
                trust_remote_code=trust_remote_code,
            )
            return model
        except Exception as exc:
            if not model_utils.is_trl_available():
                raise RuntimeError(
                    f"model({local_path}) is not a value head model, "
                    "please install trl to make it valid"
                ) from exc

        from trl import AutoModelForCausalLMWithValueHead

        vision_auto_model = model_utils.AutoModelForVision2Seq
        if (
            vision_auto_model is not None
            and type(model_config) in vision_auto_model._model_mapping.keys()
        ):
            module_class = vision_auto_model
        else:
            module_class = AutoModelForCausalLM
        ori_model = module_class.from_pretrained(
            pretrained_model_name_or_path=local_path,
            torch_dtype=torch_dtype,
            config=model_config,
            attn_implementation=attention_implementation,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(model_config, "text_config"):
            ori_model.config.hidden_size = model_config.text_config.hidden_size
        model = AutoModelForCausalLMWithValueHead.from_pretrained(ori_model)
        model_utils.patch_valuehead_model(model)
        return model

    load_valuehead_model._shopping_sdpa_fallback = True
    model_utils.load_valuehead_model = load_valuehead_model


def _install_torch_padding_fallback_impl():
    """Install hooks once, after CUDA visibility has been finalized."""
    import os

    if os.environ.get("SHOPPING_GRPO_RUNTIME_HOOKS_INSTALLED") == "1":
        return
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions
    install_valuehead_attention_fallback()
    # SAO uses veRL bypass mode to carry the asynchronous rollout log-probs
    # into the actor update.  The guarded adapter is a no-op for the existing
    # GRPO configuration and only dispatches when the typed sao_dis loss
    # sentinel is explicitly enabled.
    from shopping_grpo.training.grpo.sao_policy import (
        install_sao_gae,
        install_sao_critic_freezing,
        install_sao_policy_loss,
    )

    try:
        # Ray workers import veRL afresh, so the driver-side wrapper alone is
        # insufficient. Install the async shims in the runtime setup hook too.
        if os.environ.get("GRPO_FULLY_ASYNC") == "1":
            install_fully_async_critic_compat()
            install_fully_async_trainer_compat()
        install_sao_policy_loss()
        install_sao_gae()
        install_sao_critic_freezing()
    except ImportError as exc:
        # Unit tests provide a minimal fake veRL module for the CUDA/attention
        # hook.  A real veRL 0.8 runtime imports core_algos here; if it does
        # not, SAO is rejected by the training preflight before workers start.
        if "core_algos" not in str(exc):
            raise
    _install_trace_actor_update()
    os.environ["SHOPPING_GRPO_RUNTIME_HOOKS_INSTALLED"] = "1"


def install_torch_padding_fallback():
    """安装项目所需的 veRL 窄范围 runtime hooks。

    In a Ray worker the setup hook only installs a small wrapper around Ray's
    accelerator setter.  This keeps all CUDA-touching imports after Ray has
    selected the physical device.  Driver-side preflight calls still install
    the hooks immediately.
    """
    import os

    _isolate_ray_worker_compile_caches()
    if "__RAY_WORKER_PROCESS_SETUP_HOOK_ENV_VAR" in os.environ:
        _defer_runtime_hooks_until_ray_gpu()
        return
    _install_torch_padding_fallback_impl()
