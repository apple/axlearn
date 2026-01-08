# Copyright © 2025 Apple Inc.

"""Convenience exports for axlearn.

Usage:
    from axlearn.experiments.logistic_regression import ax

    class MyModel(ax.BaseModel):
        @ax.config_class
        class Config(ax.BaseModel.Config):
            backbone: ax.config.Required[ax.config.InstantiableConfig] = ax.config.REQUIRED
"""

__all__ = [
    # Submodules
    "base_model",
    "checkpointer",
    "config",
    "evaler",
    "input_fake",
    "input_grain",
    "input_tf_data",
    "layers",
    "learner",
    "module",
    "optimizers",
    "schedule",
    "trainer",
    # Classes
    "BaseLayer",
    "BaseModel",
    "Module",
    "SpmdTrainer",
    "Nested",
    "Tensor",
    # Config utilities
    "config_class",
    "config_for_class",
    "config_for_function",
]

# Allow frequently-used submodules to be imported as ax.xxx instead of ax.common.xxx.
from axlearn.common import (
    base_model,
    checkpointer,
    config,
    evaler,
    input_fake,
    input_grain,
    input_tf_data,
    layers,
    learner,
    module,
    optimizers,
    schedule,
    trainer,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel

# Config utilities (avoid ax.config.config_xxx)
from axlearn.common.config import config_class, config_for_class, config_for_function

# Allow frequently-used classes to be imported as ax.class instead of ax.module.class.
#
# Module hierarchy
from axlearn.common.module import Module

# Trainer
from axlearn.common.trainer import SpmdTrainer

# Common types
from axlearn.common.utils import Nested, Tensor
