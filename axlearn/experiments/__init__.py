# Copyright © 2023 Apple Inc.

"""AXLearn experiments."""
from importlib import import_module
from typing import Dict

from absl import logging

from axlearn.common.config import similar_names
from axlearn.experiments.trainer_config_utils import TrainerConfigFn


def _load_trainer_configs(
    config_module: str, *, root_module: str, optional: bool = False
) -> Dict[str, TrainerConfigFn]:
    try:
        module = import_module(f"{root_module}.experiments.{config_module}")
        return module.named_trainer_configs()
    except ImportError:
        if not optional:
            raise
        logging.warning(
            "Missing dependencies for %s but it's marked optional -- skipping.", config_module
        )
    return {}


def get_named_trainer_config(
    config_name: str, *, config_module: str, root_module: str
) -> TrainerConfigFn:
    """Looks up TrainerConfigFn by config name.

    Args:
        config_name: Candidate config name.
        config_module: Config module name.
        root_module: Root module containing config_module.

    Returns:
        A TrainerConfigFn corresponding to the config name.

    Raises:
        KeyError: Error containing the message to show to the user.
    """
    config_map = _load_trainer_configs(config_module, root_module=root_module)
    if callable(config_map):
        return config_map(config_name)

    try:
        return config_map[config_name]
    except KeyError as e:
        similar = similar_names(config_name, set(config_map.keys()))
        if similar:
            message = f"Unrecognized config {config_name}; did you mean [{', '.join(similar)}]"
        else:
            message = (
                f"Unrecognized config {config_name}; "
                "please see the 'experiments' directory for available configs."
            )
        raise KeyError(message) from e
