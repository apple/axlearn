# Copyright © 2023 Apple Inc.

"""An inference pipeline consists of an input, a runner, and an output writer."""
import time
from typing import Optional

import jax
import numpy as np
import tensorflow as tf
from absl import logging
from jax.experimental import multihost_utils

from axlearn.common import utils
from axlearn.common.config import InstantiableConfig, config_class
from axlearn.common.inference import InferenceRunner, MethodRunner
from axlearn.common.inference_output import BaseOutputWriter, OutputRecordWriter
from axlearn.common.input_tf_data import Input
from axlearn.common.module import Module
from axlearn.common.summary_writer import BaseWriter, SummaryWriter
from axlearn.common.utils import NestedTensor, Tensor


def pop_string_tensors(batch: NestedTensor) -> tuple[NestedTensor, NestedTensor]:
    """Remove string tensors from a batch, returning (batch w/o string tensors, string tensors).

    Args:
        batch: A batch of input where some fields could be string tensors.

    Returns:
        A 2-tuple where:
            - the first element is the original batch with tensors of string type removed.
            - the second element is a dictionary of the string tensors.
    """

    def prune_tf_str_tensor_leaves(_: str, subtree: NestedTensor):
        return tf.is_tensor(subtree) and subtree.dtype == tf.string

    def prune_non_tf_str_tensor_leaves(_: str, subtree: NestedTensor):
        if isinstance(subtree, (Tensor, np.ndarray)):
            return True

        if not tf.is_tensor(subtree):
            return False

        return subtree.dtype != tf.string

    batch_without_str_tensors = utils.prune_tree(batch, prune_tf_str_tensor_leaves)
    str_tensors = utils.prune_tree(batch, prune_non_tf_str_tensor_leaves)
    return batch_without_str_tensors, str_tensors


def merge_with_string_tensors(t1: NestedTensor, t2: NestedTensor) -> NestedTensor:
    """Add string tensors back to the batch.

    Generally, this is called with the return values from ``pop_string_tensors``.

    We assume t1 and t2 have the "complementary" structure:
        t1 and t2 have disjoint leaf paths.

    Args:
        t1: A NestedTensor without string tensors.
        t2: A NestedTensor consisting of string tensors.

    Returns:
        A combined NestedTensor with string tensors added back to t1.

    Raises:
        ValueError: If method is called with leaf tensor arguments.
    """
    if isinstance(t1, (Tensor, np.ndarray)) or isinstance(t2, (Tensor, np.ndarray)):
        raise ValueError(f"Expect args to be non-leaf nodes. Got t1={t1}, t2={t2}.")

    for k, v in t2.items():
        if k not in t1:
            t1[k] = v
        else:
            t1[k] = merge_with_string_tensors(t1[k], v)

    return t1


class InferencePipeline(Module):
    """A pipeline consisting of an input, a runner, and output writer.

    It supports running inference across multiple hosts.
    """

    @config_class
    class Config(Module.Config):
        """Configures InferencePipeline."""

        # Input for the pipeline.
        input: InstantiableConfig = Input.default_config()
        # The runner is responsible for computing output batches given the input batches.
        runner: InferenceRunner.Config = InferenceRunner.default_config()
        # The method to invoke on the model.
        model_method: str = "predict"
        # The output writer.
        output_writer: BaseOutputWriter.Config = OutputRecordWriter.default_config()
        # The summary writer.
        summary_writer: BaseWriter.Config = SummaryWriter.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("input", cfg.input)
        self._add_child("runner", cfg.runner)
        self._add_child(
            "output_writer",
            cfg.output_writer.set(batch_partition_spec=cfg.runner.input_batch_partition_spec),
        )
        self._add_child("summary_writer", cfg.summary_writer)

    def run(self, **kwargs):
        cfg = self.config
        method_runner = self.runner.create_method_runner(method=cfg.model_method, **kwargs)

        start_time = time.perf_counter()

        for batch_index, input_batch in enumerate(self.input.dataset()):
            input_batch, input_batch_str_tensors = pop_string_tensors(input_batch)
            input_batch = utils.as_numpy_array(input_batch)
            # pylint: disable-next=protected-access
            with method_runner._mesh:
                global_input_batch = utils.host_to_global_device_array(
                    input_batch, partition=cfg.runner.input_batch_partition_spec
                )
            output: MethodRunner.Output = method_runner(global_input_batch)
            output_batch = utils.global_to_host_array(
                output.output_batch, partition=cfg.runner.input_batch_partition_spec
            )
            if len(input_batch_str_tensors) != 0:
                input_batch = merge_with_string_tensors(input_batch, input_batch_str_tensors)
            self.output_writer.write(
                input_batch=input_batch,
                output_batch=output_batch,
            )
            self.summary_writer(step=batch_index, values=output.summaries)

            if (batch_index + 1) % 10 == 0:
                global_batch_size = len(jax.tree_util.tree_leaves(global_input_batch)[0])
                logging.info(
                    "Processed %d batches and %d examples",
                    batch_index + 1,
                    (batch_index + 1) * global_batch_size,
                )
                now = time.perf_counter()
                average_batch_time = (now - start_time) / 10
                logging.info("Average time per batch: %.2f seconds", average_batch_time)
                start_time = now

        self.output_writer.flush()
        # Synchronize flush across hosts.
        multihost_utils.sync_global_devices(self.path())
