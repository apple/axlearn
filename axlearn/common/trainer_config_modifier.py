# Copyright © 2024 Apple Inc.

"""Defines trainer config modifiers, which will be used in model definitions."""

from typing import Dict, Sequence, Union

from axlearn.common import config
from axlearn.common.base_layer import RematSpec
from axlearn.common.config import (
    REQUIRED,
    ConfigModifier,
    ConfigOr,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.gradient_accumulation import with_minibatch_steps
from axlearn.common.metrics import MetricAccumulator
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import HybridMeshShape, MeshShape


class GradientAccumulationModifier(ConfigModifier):
    """Accumulate gradients for grad_acc_steps steps."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configure GradientAccumulationModifier.

        Attributes:
            grad_acc_steps: The number of steps to accumulate the gradients from mini-batches.
            metric_accumulator: The metric accumulator to export the metrics.
        """

        grad_acc_steps: Required[int] = REQUIRED
        metric_accumulator: MetricAccumulator.Config = MetricAccumulator.default_config()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._grad_acc_steps = cfg.grad_acc_steps
        self._metric_accumulator = cfg.metric_accumulator

    def __call__(self, cfg: SpmdTrainer.Config) -> SpmdTrainer.Config:
        """Overwrite the forward_fn_transformation to accumulate gradients for grad_acc_steps steps.

        Note this would not affect the global batch size or the logical training steps.
        The optimization step is applied each time after grad_acc_steps steps of
        forward and backward passes on mini-batches.

        global_bs=mini_bs*grad_acc_steps
        train_steps=mini_steps/grad_acc_steps

        Args:
            cfg: The trainer config to be modified.

        Returns:
            The modified trainer config.
        """
        cfg.learner.forward_fn_transformation = config.config_for_function(
            with_minibatch_steps
        ).set(
            steps=self._grad_acc_steps,
            metric_accumulator=self._metric_accumulator,
        )
        return cfg


class RematSpecModifier(ConfigModifier):
    """Update the remat policies for specified modules."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configure RematSpecModifier.

        Attributes:
            remat_policies: A mapping from module path
                (e.g. `model.decoder.transformer.layer`) to remat spec.
        """

        remat_policies: Required[Dict[str, RematSpec]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._remat_policies = cfg.remat_policies

    def __call__(self, cfg: SpmdTrainer.Config) -> SpmdTrainer.Config:
        """Update the remat policy for the specified modules.

        Args:
            cfg: The trainer config to be modified.

        Raises:
            ValueError: The target module is not found.
            ValueError: The remat_spec attribute is not found.

        Returns:
            The modified trainer config.
        """

        for module_name, remat_spec in self._remat_policies.items():
            # Here we assume x.y.z format.
            # One example would be model.decoder.transformer.layer.
            target_modules = module_name.split(".")
            curr_module = cfg
            for target_module in target_modules:
                if not hasattr(curr_module, target_module):
                    raise ValueError(f"{target_module} is not found in {curr_module}.")
                curr_module = getattr(curr_module, target_module)
            # Here we assume all modules have remat_spec attribute.
            if not hasattr(curr_module, "remat_spec"):
                raise ValueError(f"{curr_module} does not have remat_spec attribute")
            curr_module.remat_spec = remat_spec
        return cfg


class MeshShapeModifier(ConfigModifier):
    """Update the mesh_shape for the trainer config."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configure MeshShapeModifier.

        Attributes:
            mesh_shape: The mesh shape to be updated to.
        """

        mesh_shape: Required[Union[MeshShape, HybridMeshShape]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._mesh_shape = cfg.mesh_shape

    def __call__(self, cfg: SpmdTrainer.Config) -> SpmdTrainer.Config:
        """Overwrite the mesh shape.

        Args:
            cfg: The trainer config to be modified.

        Returns:
            The modified trainer config.
        """
        cfg.mesh_shape = self._mesh_shape
        return cfg


class ChainConfigModifier(ConfigModifier):
    """Chain multiple config modifiers together."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configure MeshShapeModifier.

        Attributes:
            config_modifiers: A list of config modifiers to be applied.
        """

        config_modifiers: Required[Sequence[ConfigOr[ConfigModifier]]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        cfg = self.config
        self._config_modifiers = [
            maybe_instantiate(cfg_modifier) for cfg_modifier in cfg.config_modifiers
        ]

    def __call__(self, cfg: SpmdTrainer.Config) -> SpmdTrainer.Config:
        """Chain multiple config modifiers together.
        The config modifiers will be applied in the order they are provided.

        Args:
            cfg: The trainer config to be modified.

        Returns:
            The modified trainer config.
        """
        for config_modifier_fn in self._config_modifiers:
            cfg = config_modifier_fn(cfg)
        return cfg
