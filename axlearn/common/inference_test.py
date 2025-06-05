# Copyright © 2023 Apple Inc.

"""Tests inference.py.

Some tests are intended to be run on TPU.
"""

import itertools
import os
import tempfile
from collections.abc import Generator
from typing import Callable, Optional, Union
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

from axlearn.common import layers, test_utils, utils
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import CheckpointValidationType, TensorStoreStateStorage
from axlearn.common.config import Configurable, config_class, config_for_function
from axlearn.common.inference import InferenceRunner
from axlearn.common.inference_output import InputOutputRecordWriter, _json_feature
from axlearn.common.inference_pipeline import (
    InferencePipeline,
    merge_with_string_tensors,
    pop_string_tensors,
)
from axlearn.common.input_tf_data import identity
from axlearn.common.module import Module, child_context
from axlearn.common.optimizers import ParamEmaState
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    DefaultInitializer,
    FanAxes,
    Initializer,
    Shape,
)
from axlearn.common.state_builder import (
    EmaParamsConverter,
    RestoreAndConvertBuilder,
    TensorStoreStateStorageBuilder,
)
from axlearn.common.trainer import TrainerState
from axlearn.common.utils import (
    DataPartitionType,
    NestedTensor,
    Tensor,
    as_tensor,
    get_data_dir,
    set_data_dir,
)

X_DIM = 4
Y_DIM = 8
NUM_BATCHES = 3


def _build_input(
    global_batch_size: int, *, data_partition: DataPartitionType, include_str_key: bool = False
) -> Callable[[], Generator]:
    def data_gen() -> Generator:
        for batch_ix in range(NUM_BATCHES):
            # Generate global input.
            prng_key = jax.random.PRNGKey(batch_ix)
            x = jax.random.normal(prng_key, shape=(global_batch_size, X_DIM), dtype=jnp.float32)
            # Manually slice per process.
            if data_partition == DataPartitionType.FULL:
                examples_per_process = global_batch_size // jax.process_count()
                start_ix = jax.process_index() * examples_per_process
            else:
                assert data_partition == DataPartitionType.REPLICATED
                examples_per_process = global_batch_size
                start_ix = 0

            batch_input = x[start_ix : start_ix + examples_per_process, :]
            if include_str_key:
                yield dict(
                    x=batch_input,
                    q=tf.constant(
                        [
                            f"Example {batch_ix * global_batch_size + start_ix + i}"
                            for i in range(len(batch_input))
                        ]
                    ),
                )
            else:
                yield dict(x=batch_input)

    return data_gen


class RangeWeightInitializer(Initializer, Configurable):
    """Initializes weight array using a product of ranges to avoid PRNG diffs across meshes."""

    @config_class
    class Config(Configurable.Config):
        x_offset: float = 2
        y_offset: float = 5

    def initialize(
        self,
        name: str,
        *,
        prng_key: Tensor,
        shape: Shape,
        dtype: jnp.dtype,
        axes: Optional[FanAxes] = None,
    ) -> jnp.ndarray:
        del name, prng_key, axes
        assert len(shape) == 2, "Only 2D weights are supported."
        cfg = self.config
        return (cfg.y_offset + jnp.arange(shape[0], dtype=dtype).reshape(-1, 1)) * (
            cfg.x_offset + jnp.arange(shape[1], dtype=dtype).reshape(1, -1)
        )

    def debug_string(
        self,
        name: Optional[str] = None,
        shape: Optional[Shape] = None,
        axes: Optional[FanAxes] = None,
    ) -> str:
        del name, shape, axes
        cfg = self.config
        return f"range(x_offset={cfg.x_offset}, y_offset={cfg.y_offset})"


class DummyModel(BaseModel):
    """A dummy model."""

    def __init__(self, cfg: BaseModel.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._add_child(
            "linear",
            layers.Linear.default_config().set(
                input_dim=X_DIM,
                output_dim=Y_DIM,
                bias=True,
                param_partition_spec=(None, "model"),
            ),
        )
        self.predict_dtypes = []

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        y = self.predict(input_batch)
        return y.mean(), {}

    def predict(self, input_batch: NestedTensor) -> Tensor:
        x = input_batch["x"]
        with child_context("input_stats", module=self):
            self.add_module_output("x_mean", x.mean())
            self.add_module_output("x_max", x.max())
        self.predict_dtypes.append(x.dtype)
        return self.linear(x)

    def predict_batch(self, input_batch: NestedTensor) -> NestedTensor:
        x = input_batch["x"]
        self.predict_dtypes.append(x.dtype)
        return {"x": x, "y": self.linear(x)}


def is_supported(
    platform: str,
    mesh_shape: tuple[int, int],
    param_dtype: jnp.dtype,
    inference_dtype: Optional[jnp.dtype],
    global_batch_size: int,
    data_partition: DataPartitionType,
    use_ema: bool = False,
):
    del param_dtype, use_ema  # not used
    # TODO(xuan-zou): jax 0.4.25 breaks bfloat16 on CPU due to high variance on
    # the final result (up to 10% precision diff), will re-enable when fixed.
    # NOTE: bfloat16 test on GPU is added and verified.
    return (
        test_utils.is_supported_platform(platform)
        and np.prod(mesh_shape) == jax.device_count()
        and (data_partition != DataPartitionType.FULL or global_batch_size >= jax.device_count())
        and ((inference_dtype != jnp.bfloat16) or platform != "cpu")
    )


class InferenceTest(test_utils.TestCase):
    """Inference tests."""

    @parameterized.parameters(
        (tf.constant("query"), "query"),
        (tf.constant(["query"]), ["query"]),
        (tf.constant(1), 1),
        (tf.constant([1]), [1]),
        (jnp.array(1), 1),
        (jnp.array([1, 2, 3]), [1, 2, 3]),
        (tf.constant(["豆豆"]), ["豆豆"]),
    )
    def test_jsonl_feature(
        self,
        value: Union[Tensor, tf.Tensor],
        expectation: Union[int, float, bool, str, list[Union[int, float, bool, str]]],
    ):
        self.assertEqual(_json_feature(value), expectation)

    # pylint: disable-next=no-self-use
    def _runner_config(
        self,
        *,
        mesh_shape: tuple[int, int],
        mesh_axis_names: tuple[str, str],
        param_dtype: jnp.dtype,
        inference_dtype: Optional[jnp.dtype],
        data_partition: DataPartitionType,
        ckpt_dir: str,
        use_ema: bool = False,
    ):
        inference_runner_cfg = InferenceRunner.default_config().set(
            mesh_shape=mesh_shape,
            mesh_axis_names=mesh_axis_names,
            model=DummyModel.default_config().set(dtype=param_dtype),
            inference_dtype=inference_dtype,
            input_batch_partition_spec=data_partition,
        )
        if use_ema:
            inference_runner_cfg.init_state_builder = RestoreAndConvertBuilder.default_config().set(
                name="builder",
                builder=TensorStoreStateStorageBuilder.default_config().set(
                    dir=ckpt_dir, validation=CheckpointValidationType.CONTAINS_STATE_UP_TO_DTYPE
                ),
                converter=EmaParamsConverter.default_config(),
            )
        else:
            inference_runner_cfg.init_state_builder.set(dir=ckpt_dir)
        return inference_runner_cfg

    # pylint: disable-next=no-self-use
    def _build_ckpt(
        self,
        *,
        root_dir: str,
        mesh_shape: tuple[int, int],
        mesh_axis_names: tuple[str, str],
        prng_key: Tensor,
        use_ema: bool = False,
    ) -> tuple[NestedTensor, str]:
        devices = mesh_utils.create_device_mesh(mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_axis_names)
        logging.info("Global mesh: %s", mesh)
        # Create model state within mesh-context.
        with mesh:
            model_cfg = DummyModel.default_config().set(
                name="dummy",
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: RangeWeightInitializer.default_config()
                    }
                ),
                dtype=jnp.float32,
            )
            model = model_cfg.instantiate(parent=None)

            def init_state(prng_key):
                params = model.initialize_parameters_recursively(prng_key)
                if use_ema:
                    learner_state = dict(
                        ema=ParamEmaState(
                            count=0,
                            # pylint: disable-next=unnecessary-lambda
                            ema=jax.tree.map(lambda p: jnp.ones_like(p), params),
                        ),
                    )
                else:
                    learner_state = dict(
                        x=jnp.zeros([], dtype=jnp.int32), y=jnp.ones([2], dtype=jnp.float32)
                    )
                return TrainerState(
                    prng_key=prng_key,
                    model=params,
                    learner=learner_state,
                )

            state = pjit(init_state)(prng_key)

            step = 1000
            ckpt_dir = os.path.join(root_dir, f"step_{step:08d}")
            storage = TensorStoreStateStorage.default_config().instantiate()
            storage.save_to_dir(step=1000, state=state, ckpt_dir=ckpt_dir)
            storage.wait_until_finished()
            return state, ckpt_dir

    @parameterized.parameters(
        filter(
            lambda params: is_supported(*params),
            itertools.product(
                ("cpu", "gpu", "tpu"),  # platform,
                ((1, 1), (4, 1), (2, 2), (8, 1), (4, 2)),  # mesh_shape
                (jnp.float32, jnp.bfloat16),  # param_dtype
                (None, jnp.float32, jnp.bfloat16),  # inference_dtype
                (1, 16),  # global_batch_size
                (DataPartitionType.FULL, DataPartitionType.REPLICATED),  # data_partition
                (True, False),  # whether use ema weight
            ),
        )
    )
    def test_runner(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        param_dtype: jnp.dtype,
        inference_dtype: Optional[jnp.dtype],
        global_batch_size: int,
        data_partition: DataPartitionType,
        use_ema: bool,
    ):
        logging.info(
            "platform=%s mesh_shape=%s global_batch_size=%s data_partition=%s",
            platform,
            mesh_shape,
            global_batch_size,
            data_partition,
        )
        with tempfile.TemporaryDirectory() as local_tmp_dir:
            prng_key = jax.random.PRNGKey(11)
            local_run = jax.process_count() == 1
            gs_dir = os.path.join(
                "gs://axlearn-public/testdata/",
                "inference_ema_test" if use_ema else "inference_test",
            )
            root_dir = local_tmp_dir if local_run else gs_dir
            mesh_axis_names = ("data", "model")
            # Save ckpt.
            state, ckpt_dir = self._build_ckpt(
                prng_key=prng_key,
                root_dir=root_dir,
                mesh_shape=mesh_shape,
                mesh_axis_names=mesh_axis_names,
                use_ema=use_ema,
            )

            cfg = self._runner_config(
                mesh_shape=mesh_shape,
                mesh_axis_names=mesh_axis_names,
                param_dtype=param_dtype,
                inference_dtype=inference_dtype,
                ckpt_dir=ckpt_dir,
                data_partition=data_partition,
                use_ema=use_ema,
            )
            inference_runner = cfg.set(name="test_inference_runner").instantiate(parent=None)

        # Check that correct state was loaded.
        restored_state = inference_runner.inference_runner_state
        self.assertNestedEqual(restored_state.prng_key, prng_key)
        expected_model = state.learner["ema"].ema if use_ema else state.model
        if inference_dtype is not None:
            expected_model = utils.cast_floats(expected_model, to_dtype=inference_dtype)
        if param_dtype == jnp.float32 and local_run:
            # We check for model state equality only if restored dtype matches,
            # and if we are doing a local run (to avoid process gather on weights).
            self.assertNestedEqual(restored_state.model, expected_model)

        for value, restored_value in zip(
            utils.flatten_items(expected_model, separator="/"),
            utils.flatten_items(restored_state.model, separator="/"),
        ):
            # Same path.
            self.assertEqual(value[0], restored_value[0])
            # Restored dtype matches param_dtype.
            self.assertEqual(
                restored_value[1].dtype, param_dtype if inference_dtype is None else inference_dtype
            )

        # Now try to run inference.
        input_generator_fn = _build_input(global_batch_size, data_partition=data_partition)
        global_inputs = []
        global_outputs = []
        for batch in inference_runner.run(input_generator_fn(), method="predict"):
            inputs = batch["inputs"]
            outputs = batch["outputs"]
            # Validate that inputs and outputs conform to the same sharding spec.
            for v in inputs.values():
                self.assertTrue(v.sharding.is_equivalent_to(outputs.sharding, ndim=v.ndim))
            global_inputs.append(utils.replicate_to_local_data(inputs))
            global_outputs.append(utils.replicate_to_local_data(outputs))
        # Check input cast happened.
        self.assertEqual(len(inference_runner.model.predict_dtypes), 1)
        self.assertEqual(
            inference_runner.model.predict_dtypes.pop(),
            inference_dtype or jnp.float32,
        )
        if use_ema:
            weight = state.learner["ema"].ema["linear"]["weight"].astype(inference_dtype)
            bias = state.learner["ema"].ema["linear"]["bias"].astype(inference_dtype)
        else:
            weight = state.model["linear"]["weight"].astype(inference_dtype)
            bias = state.model["linear"]["bias"].astype(inference_dtype)
        expected_outputs = [el["x"].astype(inference_dtype) @ weight + bias for el in global_inputs]
        self.assertEqual(utils.shapes(global_outputs), utils.shapes(expected_outputs))
        self.assertNestedAllClose(global_outputs, expected_outputs)

    @parameterized.parameters(
        filter(
            lambda params: is_supported(*params),
            itertools.product(
                ("cpu", "gpu"),  # platform,
                ((1, 1), (4, 1), (8, 1)),  # mesh_shape
                (jnp.float32,),  # param_dtype
                (jnp.float32,),  # inference_dtype
                (16,),  # global_batch_size
                (DataPartitionType.FULL,),  # data_partition
            ),
        )
    )
    def test_runner_module_outputs(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        param_dtype: jnp.dtype,
        inference_dtype: Optional[jnp.dtype],
        global_batch_size: int,
        data_partition: DataPartitionType,
    ):
        logging.info(
            "platform=%s mesh_shape=%s global_batch_size=%s data_partition=%s",
            platform,
            mesh_shape,
            global_batch_size,
            data_partition,
        )
        with tempfile.TemporaryDirectory() as local_tmp_dir:
            prng_key = jax.random.PRNGKey(11)
            local_run = jax.process_count() == 1
            gs_dir = os.path.join(
                "gs://axlearn-public/testdata/",
                "inference_test",
            )
            root_dir = local_tmp_dir if local_run else gs_dir
            mesh_axis_names = ("data", "model")
            # Save ckpt.
            _, ckpt_dir = self._build_ckpt(
                prng_key=prng_key,
                root_dir=root_dir,
                mesh_shape=mesh_shape,
                mesh_axis_names=mesh_axis_names,
            )

            cfg = self._runner_config(
                mesh_shape=mesh_shape,
                mesh_axis_names=mesh_axis_names,
                param_dtype=param_dtype,
                inference_dtype=inference_dtype,
                ckpt_dir=ckpt_dir,
                data_partition=data_partition,
            )
            inference_runner = cfg.set(name="test_inference_runner").instantiate(parent=None)

        input_generator_fn = _build_input(global_batch_size, data_partition=data_partition)

        # Run inference with module outputs.
        module_outputs_path = "input_stats/x_mean"
        method_runner = inference_runner.create_method_runner(
            method="predict",
            drop_module_outputs=lambda path: path not in module_outputs_path,
        )
        output = method_runner(next(input_generator_fn()))
        module_outputs = utils.replicate_to_local_data(output.module_outputs)

        # Check that only the expected module outputs are returned.
        expected_module_outputs = {
            "input_stats": {
                "x_mean": utils.replicate_to_local_data(
                    output.input_batch["x"].mean(),
                )
            }
        }
        self.assertNestedAllClose(module_outputs, expected_module_outputs)

        # Run inference without module outputs (default behavior).
        method_runner = inference_runner.create_method_runner(
            method="predict",
        )
        output = method_runner(next(input_generator_fn()))
        self.assertEqual(output.module_outputs, {})

    @parameterized.parameters(
        filter(
            lambda params: is_supported(*params),
            itertools.product(
                ("cpu", "gpu", "tpu"),  # platform,
                ((1, 1), (4, 1), (2, 2), (8, 1), (4, 2)),  # mesh_shape
                (jnp.float32, jnp.bfloat16),  # param_dtype
                (None, jnp.float32, jnp.bfloat16),  # inference_dtype
                (1, 16),  # global_batch_size
                (DataPartitionType.FULL, DataPartitionType.REPLICATED),  # data_partition
                (True, False),  # whether use ema weight
            ),
        )
    )
    def test_pipeline(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        param_dtype: jnp.dtype,
        inference_dtype: Optional[jnp.dtype],
        global_batch_size: int,
        data_partition: DataPartitionType,
        use_ema: bool,
    ):
        del platform  # only used by is_supported_platform().
        local_run = jax.process_count() == 1
        with tempfile.TemporaryDirectory() as local_tmp_dir:
            gs_dir = os.path.join(
                "gs://axlearn-public/testdata/",
                "inference_ema_test" if use_ema else "inference_test",
            )
            root_dir = local_tmp_dir if local_run else gs_dir
            with set_data_dir(root_dir):
                prng_key = jax.random.PRNGKey(11)
                mesh_axis_names = ("data", "model")
                # Save ckpt.
                _, ckpt_dir = self._build_ckpt(
                    prng_key=prng_key,
                    root_dir=root_dir,
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                    use_ema=use_ema,
                )
                cfg = InferencePipeline.default_config().set(name="pipeline")
                cfg.model_method = "predict_batch"
                cfg.input.set(
                    is_training=False,
                    source=config_for_function(_build_input).set(
                        global_batch_size=global_batch_size, data_partition=data_partition
                    ),
                    processor=config_for_function(identity),
                    batcher=config_for_function(identity),
                )
                cfg.runner = self._runner_config(
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                    param_dtype=param_dtype,
                    inference_dtype=inference_dtype,
                    ckpt_dir=ckpt_dir,
                    data_partition=data_partition,
                    use_ema=use_ema,
                )
                output_path = (
                    "{data_dir}/outputs/examples-{process_index:05d}-of-{process_count:05d}"
                )
                cfg.output_writer.sink.output_path = output_path
                cfg.summary_writer.dir = os.path.join(local_tmp_dir, "summaries")
                pipeline = cfg.instantiate(parent=None)
                pipeline.run()

                output_filenames = [
                    output_path.format(
                        data_dir=get_data_dir(),
                        process_index=index,
                        process_count=jax.process_count(),
                    )
                    for index in range(jax.process_count())
                ]

                def decode_fn(record_bytes):
                    return tf.io.parse_single_example(
                        # Data.
                        record_bytes,
                        # Schema.
                        {
                            "x": tf.io.FixedLenFeature([X_DIM], dtype=tf.float32),
                            "y": tf.io.FixedLenFeature([Y_DIM], dtype=tf.float32),
                        },
                    )

                num_examples = 0
                for _ in tf.data.TFRecordDataset(output_filenames).map(decode_fn):
                    num_examples += 1
                if data_partition == DataPartitionType.FULL:
                    self.assertEqual(global_batch_size * NUM_BATCHES, num_examples)
                else:
                    self.assertEqual(DataPartitionType.REPLICATED, data_partition)
                    self.assertEqual(
                        global_batch_size * NUM_BATCHES * jax.process_count(), num_examples
                    )

    @parameterized.parameters(
        {
            "original_batch": {
                "k1": jax.random.uniform(jax.random.PRNGKey(0), shape=(2, 3)),
            },
        },
        {
            "original_batch": {
                "k1": jax.random.uniform(jax.random.PRNGKey(0), shape=(2, 3)),
                "s1": tf.constant(["abc", "def"]),
            },
        },
        {
            "original_batch": {
                "k1": jax.random.uniform(jax.random.PRNGKey(0), shape=(2, 3)),
                "s1": tf.constant(["abc", "def"]),
                "b1": {
                    "k2": jax.random.uniform(jax.random.PRNGKey(1), shape=(2, 3)),
                    "s2": tf.constant(["ghi", "jkl"]),
                },
                "b2": {
                    "s3": tf.constant(["mno", "pqr"]),
                },
            },
        },
    )
    def test_str_tensor_pop_merge_roundtrip(self, original_batch: NestedTensor):
        batch, batch_str_tensors = pop_string_tensors(original_batch)
        new_batch = merge_with_string_tensors(batch, batch_str_tensors)
        self.assertNestedEqual(original_batch, new_batch)

    @parameterized.parameters(
        {
            "batch": {
                "k1": jax.random.uniform(jax.random.PRNGKey(0), shape=(2, 3)),
            },
            "batch_str_tensors": {
                "k1": tf.constant(["abc", "def"]),
            },
        },
    )
    def test_merge_with_string_tensors_bad_input(
        self, batch: NestedTensor, batch_str_tensors: NestedTensor
    ):
        with self.assertRaisesRegex(ValueError, "Expect args to be non-leaf nodes"):
            merge_with_string_tensors(batch, batch_str_tensors)

    @parameterized.parameters(
        filter(
            lambda params: is_supported(*params),
            itertools.product(
                ("cpu", "gpu", "tpu"),  # platform,
                ((1, 1), (4, 1), (2, 2), (8, 1), (4, 2)),  # mesh_shape
                (jnp.float32,),  # param_dtype
                (jnp.float32,),  # inference_dtype
                (1, 64),  # global_batch_size
                (DataPartitionType.FULL, DataPartitionType.REPLICATED),  # data_partition
            ),
        )
    )
    def test_pipeline_with_string_tensors(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        param_dtype: jnp.dtype,
        inference_dtype: Optional[jnp.dtype],
        global_batch_size: int,
        data_partition: DataPartitionType,
    ):
        del platform  # only used by is_supported_platform().
        local_run = jax.process_count() == 1
        mesh_axis_names = ("data", "model")

        with tempfile.TemporaryDirectory() as local_tmp_dir:
            root_dir = local_tmp_dir if local_run else "gs://axlearn-public/testdata/inference_test"
            with set_data_dir(root_dir):
                prng_key = jax.random.PRNGKey(11)
                # Save ckpt.
                _, ckpt_dir = self._build_ckpt(
                    prng_key=prng_key,
                    root_dir=root_dir,
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                )
                cfg = InferencePipeline.default_config().set(name="pipeline")
                cfg.model_method = "predict_batch"
                cfg.input.set(
                    is_training=False,
                    source=config_for_function(_build_input).set(
                        global_batch_size=global_batch_size,
                        data_partition=data_partition,
                        include_str_key=True,
                    ),
                    processor=config_for_function(identity),
                    batcher=config_for_function(identity),
                )
                cfg.runner = self._runner_config(
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                    param_dtype=param_dtype,
                    inference_dtype=inference_dtype,
                    ckpt_dir=ckpt_dir,
                    data_partition=data_partition,
                )
                cfg.output_writer = InputOutputRecordWriter.default_config()

                output_path = (
                    "{data_dir}/outputs/examples-{process_index:05d}-of-{process_count:05d}"
                )
                cfg.output_writer.sink.output_path = output_path
                cfg.summary_writer.dir = os.path.join(local_tmp_dir, "summaries")
                pipeline = cfg.instantiate(parent=None)
                pipeline.run()

                output_filenames = [
                    output_path.format(
                        data_dir=get_data_dir(),
                        process_index=index,
                        process_count=jax.process_count(),
                    )
                    for index in range(jax.process_count())
                ]

                def decode_fn(record_bytes):
                    return tf.io.parse_single_example(
                        # Data.
                        record_bytes,
                        # Schema.
                        {
                            "input/q": tf.io.FixedLenFeature([], dtype=tf.string),
                            "input/x": tf.io.FixedLenFeature([X_DIM], dtype=tf.float32),
                            "output/y": tf.io.FixedLenFeature([Y_DIM], dtype=tf.float32),
                        },
                    )

                expected_query_to_x_mapping = {}
                inputs = cfg.input.set(name="input").instantiate(parent=None)
                for input_batch in inputs.dataset():
                    queries = input_batch.pop("q")
                    for i, query in enumerate(queries):
                        key = query.numpy().decode("utf-8")
                        assert key not in expected_query_to_x_mapping
                        expected_query_to_x_mapping[key] = input_batch["x"][i]

                num_examples = 0
                for record in tf.data.TFRecordDataset(output_filenames).map(decode_fn):
                    num_examples += 1
                    # Check the original input features associated with the query match.
                    query = record["input/q"].numpy().decode("utf-8")
                    if query in expected_query_to_x_mapping:
                        expected_x = expected_query_to_x_mapping[query]
                        self.assertNestedAllClose(as_tensor(record["input/x"]), expected_x)

                if data_partition == DataPartitionType.FULL:
                    self.assertEqual(global_batch_size * NUM_BATCHES, num_examples)
                else:
                    self.assertEqual(DataPartitionType.REPLICATED, data_partition)
                    self.assertEqual(
                        global_batch_size * NUM_BATCHES * jax.process_count(), num_examples
                    )

    @parameterized.parameters(
        filter(
            lambda params: is_supported(*params),
            itertools.product(
                ("cpu", "gpu"),  # platform,
                (
                    (1, 1),
                    (4, 1),
                    (8, 1),
                ),  # mesh_shape
                (jnp.float32,),  # param_dtype
                (jnp.float32,),  # inference_dtype
                (16,),  # global_batch_size
                (DataPartitionType.FULL,),  # data_partition
            ),
        )
    )
    def test_pipeline_summary_writer(
        self,
        platform: str,
        mesh_shape: tuple[int, int],
        param_dtype: jnp.dtype,
        inference_dtype: Optional[jnp.dtype],
        global_batch_size: int,
        data_partition: DataPartitionType,
    ):
        del platform  # only used by is_supported_platform().
        local_run = jax.process_count() == 1
        mesh_axis_names = ("data", "model")

        mock_summary_writer = mock.Mock(return_value=None)

        with mock.patch(
            "axlearn.common.summary_writer.SummaryWriter.Config.instantiate",
            mock.MagicMock(return_value=mock_summary_writer),
        ), tempfile.TemporaryDirectory() as local_tmp_dir:
            root_dir = local_tmp_dir if local_run else "gs://axlearn-public/testdata/inference_test"
            with set_data_dir(root_dir):
                prng_key = jax.random.PRNGKey(11)
                # Save ckpt.
                _, ckpt_dir = self._build_ckpt(
                    prng_key=prng_key,
                    root_dir=root_dir,
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                )

                cfg = InferencePipeline.default_config().set(name="pipeline")
                cfg.model_method = "predict_batch"
                cfg.input.set(
                    is_training=False,
                    source=config_for_function(_build_input).set(
                        global_batch_size=global_batch_size, data_partition=DataPartitionType.FULL
                    ),
                    processor=config_for_function(identity),
                    batcher=config_for_function(identity),
                )
                cfg.runner = self._runner_config(
                    mesh_shape=mesh_shape,
                    mesh_axis_names=mesh_axis_names,
                    param_dtype=param_dtype,
                    inference_dtype=inference_dtype,
                    ckpt_dir=ckpt_dir,
                    data_partition=data_partition,
                )
                output_path = (
                    "{data_dir}/outputs/examples-{process_index:05d}-of-{process_count:05d}"
                )
                cfg.output_writer.sink.output_path = output_path
                cfg.summary_writer.dir = os.path.join(local_tmp_dir, "summaries")
                pipeline = cfg.instantiate(parent=None)
                pipeline.run()

                mock_summary_writer.assert_any_call(step=0, values=mock.ANY)
                mock_summary_writer.assert_any_call(step=1, values=mock.ANY)
                mock_summary_writer.assert_any_call(step=2, values=mock.ANY)


if __name__ == "__main__":
    # TODO(altimofeev): The following doesn't have any effect since `utils_spmd.setup()` was
    # already called from `test_utils`. Fix and remove mesh (1, 1) from the test cases.
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    absltest.main()
