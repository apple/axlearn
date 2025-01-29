# Copyright © 2023 Apple Inc.

"""Trainer config utilities."""
from typing import Optional

from typing_extensions import Protocol

from axlearn.common.config import InstantiableConfig, config_class
from axlearn.common.flash_attention.layer import FlashBlockSizeModifier


class TrainerConfigFn(Protocol):
    """A TrainerConfigFn takes a data_dir as argument and returns a Config for instantiating a
    Trainer, e.g. SpmdTrainer.
    """

    # Note: avoid using SpmdTrainer.Config so we don't need to introduce a dependency to trainer.
    # This also makes it possible to define custom trainers with the same protocol.
    def __call__(self, data_dir: Optional[str] = None) -> InstantiableConfig:
        ...


def with_overrides(trainer_config_fn: TrainerConfigFn, **kwargs) -> TrainerConfigFn:
    """Patches the trainer config produced by the trainer_config_fn."""

    def wrapped_fn():
        trainer_cfg = trainer_config_fn()
        trainer_cfg.set(**kwargs)
        return trainer_cfg

    return wrapped_fn


class V6eFlashConfigModifier(FlashBlockSizeModifier):
    """Modified the tpu_block_size config for better performance on TPU v6e."""

    @config_class
    class Config(FlashBlockSizeModifier.Config):
        """Configures V6eFlashConfigModifier."""

        tpu_block_size: int = 1024
