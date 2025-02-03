# Copyright © 2024 Apple Inc.

"""Defines trainer config modifiers, which will be used in model definitions."""

from typing import Dict, Sequence, Union

from axlearn.common import config
from axlearn.common.base_layer import RematSpec
from axlearn.common.config import (
    REQUIRED,
    ConfigModifier,
    ConfigOr,
    Configurable,
    Required,
    config_class,
    maybe_instantiate,
)
from axlearn.common.gradient_accumulation import with_minibatch_steps
from axlearn.common.metrics import MetricAccumulator
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import HybridMeshShape, MeshShape, PartitionSpec


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
            cfg.set_recursively(module_name.split(".") + ["remat_spec"], value=remat_spec)

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


class ModuleConfigModifier(ConfigModifier):
    """Update the model config for the trainer config."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configure ModuleConfigModifier.

        Attributes:
            target_config: Target module path
                (e.g. `model.decoder.transformer.layer`) to be modified.
            modification: The new config to replace the target module's config.
        """

        target_config: Required[str] = REQUIRED
        modification: Required[Configurable.Config] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._target_config = self.config.target_config
        self._modification = self.config.modification

    def _merge_configs(
        self, target_cfg: Configurable.Config, found_module: Configurable.Config
    ) -> Configurable.Config:
        """Merge configurations from the config being replaced on a best effort basis.

        Merge Rules:
            - Klass is not changed, use target cfg.
            - If field exists in both then use from class being replaced.
            - Otherwise keep the value from target_cfg.

        Args:
            target_cfg: Configuration that will replace found_module.
            found_module: Existing configuration whose class will be replaced
                but it's confguration will be merged with target_cfg.

        Returns:
            The modified config.

        """
        for key in target_cfg.keys():
            if key == "klass":
                continue
            elif hasattr(found_module, key) and hasattr(target_cfg, key):
                setattr(target_cfg, key, getattr(found_module, key))
        return target_cfg

    def __call__(self, cfg: SpmdTrainer.Config) -> SpmdTrainer.Config:
        """Overwrite the model config of the specified modules.

        Args:
            cfg: The trainer config to be modified.

        Raises:
            ValueError: The target module is not found.

        Returns:
            The modified trainer config.
        """

        found_module = cfg.get_recursively(self._target_config.split("."))
        self._modification = self._merge_configs(self._modification, found_module)
        cfg.set_recursively(self._target_config.split("."), value=self._modification)
        return cfg


class PartitionSpecModifier(ConfigModifier):
    """Update the partition spec attribute for the specified modules."""

    @config_class
    class Config(ConfigModifier.Config):
        """Configure PartitionSpecModifier.

        Attributes:
            partition_specs: A nested mapping from module path
                (e.g. `model.decoder.transformer.layer`) to another
                mapping of model attribute to PartitionSpec.
        """

        partition_specs: Required[Dict[str, PartitionSpec]] = REQUIRED

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._attribute_dicts = self.config.partition_specs

    def __call__(self, cfg: SpmdTrainer.Config) -> SpmdTrainer.Config:
        """Update the partition_spec attributes for the specified modules.

        Args:
            cfg: The trainer config to be modified.

        Raises:
            ValueError: The target module is not found.
            ValueError: The partition_spec attribute is not found.

        Returns:
            The modified trainer config.
        """
        for module_name, partition_spec_dict in self._attribute_dicts.items():
            for partition_spec_name, partition_spec in partition_spec_dict.items():
                cfg.set_recursively(
                    module_name.split(".") + [partition_spec_name], value=partition_spec
                )

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
