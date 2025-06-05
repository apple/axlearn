# Copyright © 2023 Apple Inc.

"""A library to support writing inference outputs."""

import json
import os.path
from typing import Optional, Union

import jax
import numpy as np
import tensorflow as tf
from jax import numpy as jnp

from axlearn.common import file_system as fs
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.utils import (
    DataPartitionType,
    NestedTensor,
    Tensor,
    flatten_items,
    get_data_dir,
)


class BaseOutputWriter(Module):
    """Base class for OutputWriter, which writes records for inference outputs."""

    @config_class
    class Config(Module.Config):
        # How input and output batches are partitioned.
        batch_partition_spec: Required[DataPartitionType] = REQUIRED

    def write(
        self,
        *,
        input_batch: NestedTensor,
        output_batch: NestedTensor,
    ):
        """Writes records extracted from the given input/output batch."""
        raise NotImplementedError(type(self))

    def flush(self):
        """Flushes the written records."""
        raise NotImplementedError(type(self))


class BaseRecordSink(Module):
    def write(self, record: NestedTensor):
        """Writes `record` to the sink."""
        raise NotImplementedError(type(self))

    def flush(self):
        """Flushes the written records."""
        raise NotImplementedError(type(self))


def _tf_feature(value: Union[Tensor, tf.Tensor]) -> tf.train.Feature:
    if isinstance(value, (Tensor, np.ndarray)):
        value = jnp.reshape(value, [-1])
        if jnp.issubdtype(value.dtype, jnp.bool_):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))
        if jnp.issubdtype(value.dtype, jnp.integer):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))
        if jnp.issubdtype(value.dtype, jnp.floating):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))
    elif isinstance(value, tf.Tensor):
        value = tf.reshape(value, [-1])
        if value.dtype == tf.dtypes.string:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.numpy().tolist()))

    raise NotImplementedError(f"{value.dtype}: {value}")


def _json_feature(
    value: Union[Tensor, tf.Tensor],
) -> Union[int, float, bool, str, list[Union[int, float, bool, str]]]:
    if isinstance(value, tf.Tensor):
        value = value.numpy()

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if value.dtype == object:
        value = np.char.decode(value.astype(np.bytes_), "utf-8")

    return value.tolist()


class TfExampleRecordSink(BaseRecordSink):
    """A sink that writes each example as a record to a TF record file."""

    @config_class
    class Config(Module.Config):
        # The path should commonly contain substitution patterns for:
        #
        # - `data_dir`: The data directory name from `get_data_dir()`
        # - `process_index`: `jax.process_index()`
        # - `process_count`: `jax.process_count()`
        #
        # E.g., output_path = "{data_dir}/out-records-{process_index:05d}-of-{process_count:05d}".
        output_path: Required[str] = REQUIRED

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
    ):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        output_path = cfg.output_path.format(
            data_dir=get_data_dir(),
            process_index=jax.process_index(),
            process_count=jax.process_count(),
        )

        fs.makedirs(os.path.dirname(output_path))
        self._writer = tf.io.TFRecordWriter(output_path)

    def write(self, record: NestedTensor):
        feature_dict = {path: _tf_feature(value) for path, value in flatten_items(record)}
        self._writer.write(
            tf.train.Example(features=tf.train.Features(feature=feature_dict)).SerializeToString()
        )

    def flush(self):
        self._writer.flush()


class JsonlExampleRecordSink(BaseRecordSink):
    """A sink that writes each example as a record to a JSON Lines file."""

    @config_class
    class Config(Module.Config):
        # The path should commonly contain substitution patterns for:
        #
        # - `data_dir`: The data directory name from `get_data_dir()`
        # - `process_index`: `jax.process_index()`
        # - `process_count`: `jax.process_count()`
        #
        # E.g., output_path = "{data_dir}/out-records-{process_index:05d}-of-{
        # process_count:05d}.jsonl".
        output_path: Required[str] = REQUIRED

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
    ):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        output_path = cfg.output_path.format(
            data_dir=get_data_dir(),
            process_index=jax.process_index(),
            process_count=jax.process_count(),
        )
        fs.makedirs(os.path.dirname(output_path))
        self._writer = fs.open(output_path, "w")

    def write(self, record: NestedTensor):
        feature_dict = {path: _json_feature(value) for path, value in flatten_items(record)}
        self._writer.write(json.dumps(feature_dict) + "\n")

    def flush(self):
        self._writer.flush()


class OutputRecordWriter(BaseOutputWriter):
    """An output writer that writes each output example as a separate record to a given sink.

    Extracts examples accessible by the local host and writes every example as a record to
    `self.sink`. Assumes that the input and output batches are fully partitioned along
    the batch axis across hosts such that each host has access to a disjoint partition of the
    examples. Users should define a custom subclass of `BaseOutputWriter` if this assumption
    does not hold.
    """

    @config_class
    class Config(BaseOutputWriter.Config):
        sink: BaseRecordSink.Config = TfExampleRecordSink.default_config()

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
    ):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("sink", cfg.sink)

    def write(self, *, input_batch: NestedTensor, output_batch: NestedTensor):
        """Writes records extracted from the given input/output batch pair.

        Args:
            input_batch: A NestedTensor whose leaves must be tensors of shape [batch_size, ...].
            output_batch: A NestedTensor whose leaves must be tensors of shape [batch_size, ...].
        """
        local_data = dict(input=input_batch, output=output_batch)
        local_batch_size = jax.tree_util.tree_leaves(local_data)[0].shape[0]

        for i in range(local_batch_size):
            example = jax.tree.map(lambda x, index=i: x[index], local_data)
            self.sink.write(
                self._build_record(input_example=example["input"], output_example=example["output"])
            )

    # pylint: disable-next=no-self-use
    def _build_record(
        self, *, input_example: NestedTensor, output_example: NestedTensor
    ) -> NestedTensor:
        """Writes only the output example by default."""
        del input_example
        return output_example

    def flush(self):
        self.sink.flush()


class InputOutputRecordWriter(OutputRecordWriter):
    """An output writer that writes each example's input/output as a separate record to a sink.

    The input is keyed under "input" and output is keyed under "output".

    Extracts examples accessible by the local host and writes every example as a record to
    `self.sink`. Assumes that the input and output batches are fully partitioned along
    the batch axis across hosts such that each host has access to a disjoint partition of the
    examples. Users should define a custom subclass of `BaseOutputWriter` if this assumption
    does not hold.
    """

    # pylint: disable-next=no-self-use
    def _build_record(
        self, *, input_example: NestedTensor, output_example: NestedTensor
    ) -> NestedTensor:
        """Writes both the input and output for each example."""
        return dict(input=input_example, output=output_example)
