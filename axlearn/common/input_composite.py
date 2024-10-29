# Copyright © 2023 Apple Inc.

"""A library to support composite inputs."""

from collections.abc import Sequence
from typing import Optional

import tensorflow as tf

from axlearn.common import input_base
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.module import Module


class ConcatenatedInput(input_base.Input):
    """A Module to generate input batches from a sequence of sub inputs.

    The sub inputs will be iterated in the order that they are given. All sub inputs except the
    last one are expected to have finite numbers of elements.
    """

    @config_class
    class Config(input_base.Input.Config):
        is_training: Required[bool] = REQUIRED
        inputs: Sequence[InstantiableConfig] = []

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if not cfg.inputs:
            raise ValueError("Empty inputs for ConcatenatedInput")
        self._inputs = [
            self._add_child(f"input_{i}", input_cfg.set(is_training=cfg.is_training))
            for i, input_cfg in enumerate(cfg.inputs)
        ]

    def dataset(self) -> tf.data.Dataset:
        ds = self._inputs[0].dataset()
        for sub_input in self._inputs[1:]:
            ds = ds.concatenate(sub_input.dataset())
        return ds


class ZipInput(input_base.Input):
    """A Module to generate input batches from a sequence of sub inputs.

    The sub inputs will be combined in the order that they are given.
    Iteration stops when any input iterator has been exhausted
        (i.e, it is limited by the size of the smallest dataset).
    """

    @config_class
    class Config(input_base.Input.Config):
        is_training: Required[bool] = REQUIRED
        inputs: dict[str, InstantiableConfig] = {}

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._inputs = tuple(
            self._add_child(f"input_{name}", input_cfg.set(is_training=cfg.is_training))
            for name, input_cfg in cfg.inputs.items()
        )
        self._inputs_name = list(cfg.inputs)

    def dataset(self) -> tf.data.Dataset:
        dataset = tf.data.Dataset.zip(tuple(x.dataset() for x in self._inputs))
        return dataset.map(lambda *x: {self._inputs_name[i]: data for i, data in enumerate(x)})
