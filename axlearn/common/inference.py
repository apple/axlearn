# Copyright © 2023 Apple Inc.

"""Tools for running inference on AXLearn models.

On compatible trainer checkpoints for `InferenceRunner`:
    Default ckpt builder should succeed without errors for restoration of any trainer checkpoint
    that contains at a minimum the set of inference model state + PRNG Key, even if dtypes differ
    (as is the case if we want a model.dtype of bfloat16 for inference when the trainer
    state is float32).
    If you wish to restore a model for inference from a trainer checkpoint which contains
    only a subset of the model state, or a checkpoint with different scope names, use a suitable
    state builder.
"""
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from jax.experimental.pjit import pjit

from axlearn.common import utils
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import CheckpointValidationType
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.state_builder import Builder, TensorStoreStateStorageBuilder
from axlearn.common.trainer import SpmdTrainer
from axlearn.common.utils import (
    DataPartitionType,
    NestedPartitionSpec,
    NestedTensor,
    PartitionSpec,
    TensorSpec,
)


# pylint: disable-next=too-few-public-methods
class MethodRunner:
    """Class for running a pjit-lowered model's method on a batch of data."""

    def __init__(
        self,
        *,
        prng_key: jax.random.KeyArray,
        mesh: jax.sharding.Mesh,
        input_batch_partition_spec: DataPartitionType,
        jit_run_on_batch: Callable[
            [jax.random.KeyArray, NestedTensor],
            Tuple[jax.random.KeyArray, NestedTensor, NestedTensor],
        ],
    ):
        """Initializes MethodRunner object.

        Args:
            prng_key: the random key used for the first run, then updated by each call.
            mesh: mesh to be used during method running, same as the one used for pjit.
            input_batch_partition_spec: partition spec for input batches.
            jit_run_on_batch: callable which takes prng key, input batch and outputs
                updated prng key, outputs and summaries.
        """
        self._prng_key = prng_key
        self._mesh = mesh
        self._input_batch_partition_spec = input_batch_partition_spec
        self._jit_run_on_batch = jit_run_on_batch

    @dataclass(frozen=True)
    class Output:
        """Output class of MethodRunner."""

        # Output batch as a partitioned global array.
        output_batch: NestedTensor
        # Input batch as a partitioned global array.
        input_batch: NestedTensor
        # Summaries.
        summaries: NestedTensor

    def __call__(self, input_batch: NestedTensor) -> Output:
        """Computes outputs and summaries for the given input.

        The convention is for input_batch to be global arrays.
        Output batches are global arrays.
        This symmetry in global arrays for both input and output batches allows
        users to chain multiple `MethodRunner`s together without extra host-device transfer.
        If the input_batch is host-local, it will be automatically converted to
            global input batch for ease-of-use.

        Args:
            input_batch: An input batch of data. By convention, these are global arrays.
                Host-local input batches are also accepted and converted to global input batches.

        Returns:
            An Output object containing global batch inputs, outputs and summaries.
            N.B. the returned input and output batches will have the same partitioning.
        """
        with self._mesh:
            # TODO(zhucheng_tu,tom_gunter): Handle a mixture of pre-sharded and host-local inputs.
            is_host_local_input_check = lambda x: (
                isinstance(x, jax.Array) and len(x.devices()) == 1
            ) or isinstance(x, np.ndarray)
            all_host_local_inputs = all(
                is_host_local_input_check(t) for t in jax.tree_util.tree_leaves(input_batch)
            )

            if all_host_local_inputs:
                global_input_batch = utils.host_to_global_device_array(
                    input_batch, partition=self._input_batch_partition_spec
                )
            else:
                global_input_batch = input_batch

            (
                self._prng_key,
                global_output_batch,
                summaries,
            ) = self._jit_run_on_batch(
                self._prng_key,
                global_input_batch,
            )
            return self.Output(
                output_batch=global_output_batch,
                input_batch=global_input_batch,
                summaries=summaries,
            )


class _InferenceRunnerState(NamedTuple):
    """Contains inference runner {state | state-partition-specs}."""

    prng_key: Union[jax.random.KeyArray, NestedPartitionSpec]
    model: Union[NestedTensor, NestedPartitionSpec]
    learner: Optional[Union[NestedTensor, NestedPartitionSpec]] = None


class InferenceRunner(Module):
    """Handles loading a model and running inference.

    The following methods can be used after an `InferenceRunner` is instantiated:

    `create_method_runner(...)` creates a runner which can be used for computing outputs
    and summaries for `input_batch` (see `MethodRunner` for details).

    Alternatively one can use `run(...)` which takes takes iterable over input batches and
    returns outputs generator - convenient for basic inference loops etc.
    """

    @config_class
    class Config(Module.Config):
        """Configures InferenceRunner."""

        # The device mesh shape in the form of a tuple of ints.
        # Must have the same length as mesh_axis_names.
        mesh_shape: Required[Sequence[int]] = REQUIRED
        # The mesh axis names. The names can be referenced in TensorSpec.mesh_axes.
        mesh_axis_names: Required[Sequence[str]] = REQUIRED

        # The model config.
        model: Required[BaseModel.Config] = REQUIRED

        # The builder to load checkpoint.
        init_state_builder: Builder.Config = TensorStoreStateStorageBuilder.default_config().set(
            validation=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE,
        )

        # Cast float inputs and model parameters to this dtype before executing.
        # N.B. Model dtypes restored from ckpt are controlled in the model cfg.
        inference_dtype: Optional[jnp.dtype] = None

        # How to partition input batches. Also used for output batches.
        input_batch_partition_spec: DataPartitionType = DataPartitionType.FULL

    @classmethod
    def config_from_trainer(cls, trainer_cfg: SpmdTrainer.Config) -> Config:
        """Creates a runner config with mesh, name, and model populated from a trainer config."""
        cfg = cls.default_config().set(
            name=trainer_cfg.name,
            mesh_shape=trainer_cfg.mesh_shape,
            mesh_axis_names=trainer_cfg.mesh_axis_names,
            model=trainer_cfg.model,
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)

        cfg = self.config
        if cfg.inference_dtype is not None:
            utils.validate_float_dtype(cfg.inference_dtype)

        # Create device mesh.
        logging.info(
            "Devices: global=%s local=%s %s",
            jax.device_count(),
            jax.local_device_count(),
            [device.platform for device in jax.local_devices()],
        )
        logging.info("Mesh shape: %s", cfg.mesh_shape)
        devices = utils.create_device_mesh(cfg.mesh_shape)
        mesh = jax.sharding.Mesh(devices, cfg.mesh_axis_names)
        logging.info("Global mesh: %s", mesh)
        self._mesh = mesh

        # Create child objects within mesh-context.
        with self.mesh():
            self._add_child("model", cfg.model)
            self._model_param_specs = self.model.create_parameter_specs_recursively()
            self._inference_runner_state_specs = _InferenceRunnerState(
                prng_key=TensorSpec(dtype=jnp.uint32, shape=[4], mesh_axes=PartitionSpec(None)),
                model=self._model_param_specs,
            )
            self._inference_runner_state_partition_specs = jax.tree_util.tree_map(
                lambda spec: spec.mesh_axes, self._inference_runner_state_specs
            )
            logging.info("Building ckpt state from %s", cfg.init_state_builder.klass.__name__)
            builder = cfg.init_state_builder.set(
                name="init_state_builder",
            ).instantiate(parent=None)

            # Check that builder should expect tensor specs.
            if builder.input_state_type() != Builder.StateType.TENSOR_SPECS:
                logging.warning(
                    "init_state_builder %s expects input_state_type StateType.TENSOR "
                    "but inference runner gives StateType.TENSOR_SPECS.",
                    cfg.init_state_builder.klass.__name__,
                )

            # See "On compatible trainer checkpoints for `InferenceRunner`" in the file docstring.
            self._inference_runner_state = builder(
                Builder.State(
                    step=0, trainer_state=self._inference_runner_state_specs, built_keys=set()
                )
            ).trainer_state

    @property
    def inference_runner_state(self):
        """Get the inference runner state."""
        return self._inference_runner_state

    def run(
        self,
        input_batches: Iterable[NestedTensor],
        *,
        method: str,
        prng_key: Optional[jax.random.KeyArray] = None,
        **kwargs,
    ) -> Generator[NestedTensor, None, None]:
        """Runs inference on the provided input batches.

        Note, one can also use `create_method_runner` if more flexibility of input's handling is
        needed.

        Args:
            input_batches: an iterable of input batches.
            method: the method name of self.model to invoke. The method should take an
                `input_batch` arg and return a NestedTensor. Both `input_batch` and the
                returned value are NestedTensors containing Tensors with a leading dimension of
                `batch_size` and will be partitioned with input_partition_spec.
            prng_key: the random key used for inference. Use restored key if None.
            kwargs: Keyword arguments to pass to the method.

        Yields:
            Dict[str, NestedTensor]: Input and output tensor batches. The dict will consist of keys:
                * 'inputs': The named key-value pairs of the input batch as global arrays.
                * 'outputs': The output of the method as global arrays.

        Raises:
            AttributeError: if method is not found at self.model.
        """
        runner = self.create_method_runner(method=method, prng_key=prng_key, **kwargs)
        for input_batch in input_batches:
            logging.log_first_n(logging.INFO, "Input batch: %s", 3, input_batch)
            runner_output = runner(input_batch)
            self.vlog(
                2,
                "Output batch: %s, summaries=%s",
                runner_output.output_batch,
                utils.flatten_items(runner_output.summaries),
            )
            yield {"inputs": runner_output.input_batch, "outputs": runner_output.output_batch}

    def create_method_runner(
        self,
        *,
        method: str,
        prng_key: Optional[jax.random.KeyArray] = None,
        **kwargs,
    ) -> MethodRunner:
        """Creates MethodRunner for the specified method and arguments.

        Args:
            method: the method name of self.model to invoke. The method should take an
                `input_batch` arg and return a NestedTensor. Both `input_batch` and the
                returned value are NestedTensors containing Tensors with a leading dimension of
                `batch_size` and will be partitioned with input_batch_partition_spec.
            prng_key: the random key used for inference. Use restored key if None.
            kwargs: Keyword arguments to pass to the method.

        Returns:
            MethodRunner for computing output results.

        Raises:
            AttributeError: if method is not found at self.model.
        """

        cfg: InferenceRunner.Config = self.config
        available_methods = {m for m in dir(self.model) if not m.startswith("_")}
        if method not in available_methods:
            raise AttributeError(
                f"{self.path()}.model does not have method {method}. "
                f"Available methods are {available_methods}."
            )

        with self.mesh():

            def inference_iter(model_params, prng_key, input_batch):
                return self._inference_iter(
                    prng_key,
                    model_params,
                    input_batch,
                    method=method,
                    **kwargs,
                )

            data_partition_spec = utils.data_partition_type_to_spec(cfg.input_batch_partition_spec)
            jit_inference_iter_fn = pjit(
                inference_iter,
                in_shardings=(
                    self._inference_runner_state_partition_specs.model,
                    self._inference_runner_state_partition_specs.prng_key,
                    data_partition_spec,  # Input batch.
                ),
                out_shardings=(
                    self._inference_runner_state_partition_specs.prng_key,
                    data_partition_spec,  # Output batch.
                    None,  # Summaries.
                ),
            )
            self.vlog(1, "jit complete for %s", method)
            prng_key = self._inference_runner_state.prng_key if prng_key is None else prng_key

            return MethodRunner(
                prng_key=prng_key,
                mesh=self.mesh(),
                input_batch_partition_spec=cfg.input_batch_partition_spec,
                jit_run_on_batch=partial(jit_inference_iter_fn, self._inference_runner_state.model),
            )

    def _inference_iter(
        self,
        prng_key: jax.random.KeyArray,
        model_params: NestedTensor,
        input_batch: Dict[str, Any],
        *,
        method,
        **kwargs,
    ) -> Tuple[jax.random.KeyArray, NestedTensor, NestedTensor]:
        """Implements inference for a single input batch."""
        cfg = self.config
        new_prng_key, iter_key = jax.random.split(prng_key)

        def inference_cast(in_tree: NestedTensor) -> NestedTensor:
            if cfg.inference_dtype is not None:
                return utils.cast_floats(in_tree, to_dtype=cfg.inference_dtype)
            return in_tree

        input_batch = utils.shard_input_batch(inference_cast(input_batch))
        output_batch, output_collection = F(
            self.model,
            prng_key=iter_key,
            state=inference_cast(model_params),
            inputs={"input_batch": input_batch, **kwargs},
            is_training=False,
            method=method,
        )
        return (
            new_prng_key,
            output_batch,
            output_collection.summaries,
        )

    def mesh(self):
        """Used as a context manager when calling/creating mesh-aware objects."""
        return jax.sharding.Mesh(self._mesh.devices, self._mesh.axis_names)
