"""Project entrypoint for veRL fully-async training.

veRL 0.8's experimental separation trainer needs a small critic-config
compatibility shim before its Ray task class is created.  Import the shim
first, then delegate all training behavior to veRL's maintained entrypoint.
"""

from shopping_grpo.training.grpo.compat import (
    install_fully_async_critic_compat,
    install_fully_async_trainer_compat,
    install_valuehead_attention_fallback,
)

install_fully_async_critic_compat()
install_fully_async_trainer_compat()
install_valuehead_attention_fallback()

from verl.experimental.fully_async_policy.fully_async_main import main  # noqa: E402


if __name__ == "__main__":  # pragma: no cover
    main()
