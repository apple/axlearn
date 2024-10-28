# Copyright © 2023 Apple Inc.

"""Test evalers.

Some tests are intended to be run on TPU.
"""

# pylint: disable=no-self-use
import graphlib
import json
import os
import tempfile
from collections.abc import Sequence
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl.testing import absltest, parameterized
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from tensorflow.python.framework import tensor_util  # pylint: disable=no-name-in-module
from tensorflow.python.summary.summary_iterator import (  # pylint: disable=no-name-in-module
    summary_iterator,
)

from axlearn.common import file_system as fs
from axlearn.common import param_init, test_utils
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import config_class, config_for_function
from axlearn.common.evaler import (
    BaseMetricCalculator,
    CompositeMetricCalculator,
    GlobalMetricCalculator,
    ModelSummaryAccumulator,
    SpmdEvaler,
)
from axlearn.common.inference_output import (
    BaseRecordSink,
    JsonlExampleRecordSink,
    OutputRecordWriter,
    TfExampleRecordSink,
)
from axlearn.common.layers import Linear
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import REQUIRED, Module, OutputCollection, Required
from axlearn.common.summary_writer import SummaryWriter
from axlearn.common.test_utils import DummyForwardModel, TestCase
from axlearn.common.utils import (
    DataPartitionType,
    Nested,
    NestedTensor,
    Tensor,
    get_data_dir,
    replicate_to_local_data,
    set_data_dir,
)

_EXAMPLE_SHAPE = [
    32,
]


class DummyInput(Module):
    """A dummy input."""

    @config_class
    class Config(Module.Config):
        """Configures DummyInput."""

        is_training: Required[bool] = REQUIRED
        batch_size: Required[int] = REQUIRED
        shape: Required[list[int]] = REQUIRED
        total_num_batches: Optional[int] = None

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.shape = _EXAMPLE_SHAPE
        return cfg

    def batches(self, it: tf.data.Iterator):
        yield from it

    def dataset(self):
        cfg: DummyInput.Config = self.config
        num_batches = 0
        shape = [cfg.batch_size, *cfg.shape]
        while cfg.total_num_batches is None or num_batches < cfg.total_num_batches:
            num_batches += 1
            inputs = jnp.ones(shape=shape, dtype=jnp.float32)
            yield dict(inputs=inputs)

    def __iter__(self):
        return self.dataset()


class DummyModel(BaseModel):
    """A dummy model."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyModel."""

        layer: Required[BaseLayer.Config] = REQUIRED

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.dtype = jnp.float32  # pylint: disable=no-member
        cfg.layer = Linear.default_config().set(
            input_dim=_EXAMPLE_SHAPE[-1],
            output_dim=_EXAMPLE_SHAPE[-1],
            bias=False,
            param_partition_spec=("model", None),
        )
        cfg.name = cls.__name__
        cfg.param_init = param_init.ConstantInitializer.default_config().set(value=1.0)
        return cfg

    def __init__(self, cfg: BaseModel.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("linear", cfg.layer)
        self.forward_dtypes = []

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        inputs = input_batch["inputs"]
        self.forward_dtypes.append(inputs.dtype)
        logits = self.linear(inputs)
        self.add_summary("model_logits_sum", WeightedScalar(logits.sum(), logits.shape[0]))
        return logits.mean(), {"logits": logits}

    # pylint: disable-next=no-self-use,unused-argument
    def alt_predict(self, input_batch: NestedTensor, **kwargs) -> NestedTensor:
        return input_batch["mock_output"]


class DummyMetricCalculator(ModelSummaryAccumulator):
    """Computes "mean_prediction", "min_prediction", "max_prediction" based on model outputs."""

    @config_class
    class Config(ModelSummaryAccumulator.Config):
        # A subset of {"mean_prediction", "min_prediction", "max_prediction"}
        metrics: Required[Sequence[str]] = REQUIRED
        output_model_predictions: bool = False

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.metrics = ["mean_prediction"]
        return cfg

    def _per_example_outputs(self, model_outputs: NestedTensor) -> NestedTensor:
        _, aux_out = model_outputs
        return dict(logits=aux_out["logits"])

    def _call_model(
        self,
        *,
        method: str,
        prng_key: Tensor,
        model_params: NestedTensor,
        input_batch: NestedTensor,
        **kwargs,
    ) -> tuple[NestedTensor, OutputCollection]:
        cfg = self.config
        model_outputs, model_output_collection = super()._call_model(
            method=method,
            prng_key=prng_key,
            model_params=model_params,
            input_batch=input_batch,
            **kwargs,
        )
        _, aux_out = model_outputs
        predictions = jax.nn.sigmoid(aux_out["logits"])
        if "mean_prediction" in cfg.metrics:
            model_output_collection.summaries["mean_prediction"] = WeightedScalar(
                jnp.mean(predictions), predictions.shape[0]
            )
        if "min_prediction" in cfg.metrics:
            model_output_collection.summaries["min_prediction"] = WeightedScalar(
                jnp.min(predictions), predictions.shape[0]
            )
        if "max_prediction" in cfg.metrics:
            model_output_collection.summaries["max_prediction"] = WeightedScalar(
                jnp.max(predictions), predictions.shape[0]
            )
        return model_outputs, model_output_collection


class EvalerTest(TestCase):
    @parameterized.parameters(
        ("cpu", (1, 1), None),
        ("cpu", (1, 1), jnp.bfloat16),
        ("tpu", (4, 1), jnp.bfloat16),
        ("tpu", (8, 1), jnp.bfloat16),
        ("tpu", (2, 4), jnp.float32),
    )
    def test_spmd_evaler(self, platform, mesh_shape, step_dtype):
        if not test_utils.is_supported_platform(platform):
            return
        with jax.sharding.Mesh(mesh_utils.create_device_mesh(mesh_shape), ("data", "model")):
            # Create model state.
            model_cfg = DummyModel.default_config()
            model = model_cfg.instantiate(parent=None)
            model_param_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
            )
            model_state = pjit(
                model.initialize_parameters_recursively,
                in_shardings=(None,),
                out_shardings=model_param_partition_specs,
            )(jax.random.PRNGKey(0))
            # Instantiate evaler:
            with tempfile.TemporaryDirectory() as temp_dir:
                process_batch_size = 8
                num_batches = 10
                evaler = (
                    SpmdEvaler.default_config()
                    .set(
                        input=DummyInput.default_config().set(
                            total_num_batches=num_batches, batch_size=process_batch_size
                        ),
                        name="spmd_evaler",
                        summary_writer=SummaryWriter.default_config().set(dir=temp_dir),
                        eval_dtype=step_dtype,
                        metric_calculator=DummyMetricCalculator.default_config(),
                        trace_at_iters=[1],
                    )
                    .instantiate(
                        parent=None,
                        model=model,
                        model_param_partition_specs=model_param_partition_specs,
                    )
                )
                # Run the evaler.
                evaler.eval_step(1, prng_key=jax.random.PRNGKey(789), model_params=model_state)
                # Check that we honored the step type.
                self.assertEqual(
                    len(
                        [
                            el
                            for el in model.forward_dtypes
                            if el == (step_dtype or model_cfg.dtype)  # pylint: disable=no-member
                        ]
                    ),
                    1,  # Only once for the eval_step trace.
                )

                if jax.process_index() != 0:
                    # Don't examine summaries if not on process 0.
                    return

                events_files = os.listdir(temp_dir)
                self.assertNotEmpty(events_files)
                self.assertTrue("plugins" in events_files)
                summaries_dict = {}
                for event_file in events_files:
                    event_file_path = os.path.join(temp_dir, event_file)
                    if event_file == "plugins":
                        self.assertTrue("profile" in os.listdir(event_file_path))
                        # Single profile was logged.
                        self.assertEqual(
                            len(os.listdir(os.path.join(event_file_path, "profile"))), 1
                        )
                        continue
                    for summary in summary_iterator(event_file_path):
                        for val in summary.summary.value:
                            summaries_dict[val.tag] = tensor_util.MakeNdarray(val.tensor)

                self.assertGreater(summaries_dict.pop("eval_time_secs"), 0)

                # Check all the summaries line up.
                expected_values = np.array(
                    replicate_to_local_data(model_state["linear"]["weight"]).sum()
                    * process_batch_size
                    * jax.process_count(),
                    dtype=jnp.float32,
                )
                self.assertNestedAllClose(
                    dict(model_logits_sum=expected_values, mean_prediction=1.0),
                    summaries_dict,
                    atol=1e-4,
                    rtol=1e-6,
                )

    def test_min_step(self):
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            # Create model state.
            model_cfg = DummyModel.default_config()
            model = model_cfg.instantiate(parent=None)
            model_param_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
            )
            model_state = pjit(
                model.initialize_parameters_recursively,
                in_shardings=(None,),
                out_shardings=model_param_partition_specs,
            )(jax.random.PRNGKey(0))
            # Instantiate evaler:
            with tempfile.TemporaryDirectory() as temp_dir:
                process_batch_size = 8
                num_batches = 10
                evaler = (
                    SpmdEvaler.default_config()
                    .set(
                        input=DummyInput.default_config().set(
                            total_num_batches=num_batches, batch_size=process_batch_size
                        ),
                        name="spmd_evaler",
                        summary_writer=SummaryWriter.default_config().set(dir=temp_dir),
                        metric_calculator=DummyMetricCalculator.default_config(),
                        trace_at_iters=[1],
                    )
                    .instantiate(
                        parent=None,
                        model=model,
                        model_param_partition_specs=model_param_partition_specs,
                    )
                )
                # Run the evaler.
                _, summary, _ = evaler.eval_step(
                    0, prng_key=jax.random.PRNGKey(789), model_params=model_state
                )
                self.assertIsNone(summary)
                _, summary, _ = evaler.eval_step(
                    1, prng_key=jax.random.PRNGKey(789), model_params=model_state
                )
                self.assertIsNotNone(summary)

    @parameterized.parameters(TfExampleRecordSink, JsonlExampleRecordSink)
    def test_output_writer(self, sink: type[BaseRecordSink]):
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            with tempfile.TemporaryDirectory() as temp_dir:
                with set_data_dir(temp_dir):
                    # Create model state.
                    model_cfg = DummyModel.default_config()
                    model = model_cfg.instantiate(parent=None)
                    model_param_partition_specs = jax.tree.map(
                        lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
                    )
                    model_state = pjit(
                        model.initialize_parameters_recursively,
                        in_shardings=(None,),
                        out_shardings=model_param_partition_specs,
                    )(jax.random.PRNGKey(0))
                    # Instantiate evaler.
                    process_batch_size = 8
                    num_batches = 10
                    output_path = "{data_dir}/outputs-{process_index:05d}-of-{process_count:05d}"
                    output_writer_cfg = OutputRecordWriter.default_config().set(
                        batch_partition_spec=DataPartitionType.FULL,
                        sink=sink.default_config().set(output_path=output_path),
                    )
                    evaler = (
                        SpmdEvaler.default_config()
                        .set(
                            input=DummyInput.default_config().set(
                                total_num_batches=num_batches, batch_size=process_batch_size
                            ),
                            name="spmd_evaler",
                            summary_writer=SummaryWriter.default_config().set(dir=temp_dir),
                            metric_calculator=DummyMetricCalculator.default_config(),
                            output_writer=output_writer_cfg,
                        )
                        .instantiate(
                            parent=None,
                            model=model,
                            model_param_partition_specs=model_param_partition_specs,
                        )
                    )
                    # Run the evaler.
                    _, summary, _ = evaler.eval_step(
                        1, prng_key=jax.random.PRNGKey(789), model_params=model_state
                    )
                    self.assertIsNotNone(summary)
                    evaler.output_writer.flush()

                    actual_output_path = output_path.format(
                        data_dir=get_data_dir(),
                        process_index=jax.process_index(),
                        process_count=jax.process_count(),
                    )

                    if sink == TfExampleRecordSink:

                        def decode_record(record_bytes):
                            # pytype: disable=module-attr
                            return tf.io.parse_single_example(
                                record_bytes,
                                {
                                    "logits": tf.io.FixedLenFeature(
                                        [_EXAMPLE_SHAPE[-1]], dtype=tf.float32
                                    )
                                },
                            )
                            # pytype: enable=module-attr

                        ds = tf.data.TFRecordDataset(actual_output_path).map(decode_record)
                        num_output_records = 0
                        for _ in ds:
                            num_output_records += 1
                        self.assertEqual(process_batch_size * num_batches, num_output_records)
                    else:
                        assert sink == JsonlExampleRecordSink
                        num_output_records = 0
                        with fs.open(actual_output_path) as f:
                            for line in f:
                                record = json.loads(line)
                                self.assertIsInstance(record["logits"], list)
                                num_output_records += 1
                        self.assertEqual(process_batch_size * num_batches, num_output_records)

    def test_eval_policy(self):
        # For simplicity, test on single host.
        if not test_utils.is_supported_platform("cpu"):
            return

        # Only eval if a metric exceeds a threshold.
        def metric_threshold_policy(*, metric: str, threshold: int):
            def fn(*, step, train_summaries) -> bool:
                del step
                return train_summaries.get(metric, 0) > threshold

            return fn

        model_cfg = DummyModel.default_config()
        model = model_cfg.instantiate(parent=None)
        model_state = model.initialize_parameters_recursively(jax.random.PRNGKey(0))
        mesh = mesh_utils.create_device_mesh((1, 1))

        with jax.sharding.Mesh(mesh, ("data", "model")), tempfile.TemporaryDirectory() as temp_dir:
            num_batches = 3
            evaler = (
                SpmdEvaler.default_config()
                .set(
                    name="spmd_evaler",
                    input=DummyInput.default_config().set(
                        total_num_batches=num_batches, batch_size=1
                    ),
                    summary_writer=SummaryWriter.default_config().set(dir=temp_dir),
                    eval_policy=config_for_function(metric_threshold_policy).set(
                        metric="test", threshold=1
                    ),
                )
                .instantiate(
                    parent=None,
                    model=model,
                    model_param_partition_specs=None,
                )
            )
            keys = jax.random.split(jax.random.PRNGKey(123), num=4)
            # When we have no train summaries, or when the target metric is below threshold, eval
            # should be skipped.
            for i, train_summaries in enumerate([None, {"test": 0.5}, {"test": 1}]):
                _, summaries, _ = evaler.eval_step(
                    i + 1,
                    prng_key=keys[i],
                    model_params=model_state,
                    train_summaries=train_summaries,
                )
                self.assertIsNone(summaries)

            # When metric is above threshold, eval should run.
            _, summaries, _ = evaler.eval_step(
                2, prng_key=keys[2], model_params=model_state, train_summaries={"test": 1.1}
            )
            self.assertIsNotNone(summaries)

            # If metric dips below threshold again, eval should not run.
            _, summaries, _ = evaler.eval_step(
                3, prng_key=keys[3], model_params=model_state, train_summaries={"test": 0.9}
            )
            self.assertIsNone(summaries)

            # If metric dips below threshold again, eval should not run. But force it to run.
            _, summaries, _ = evaler.eval_step(
                3,
                prng_key=keys[3],
                model_params=model_state,
                train_summaries={"test": 0.9},
                force_run=True,
            )
            self.assertIsNotNone(summaries)


class ModelSummaryAccumulatorTest(absltest.TestCase):
    def test_accumulated_summaries_match(self):
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            model = DummyForwardModel.default_config().set(name="model").instantiate(parent=None)
            accumulator: ModelSummaryAccumulator = (
                ModelSummaryAccumulator.default_config()
                .set(name="test")
                .instantiate(parent=None, model=model, model_param_partition_specs={})
            )
            # pylint: disable-next=protected-access
            state = accumulator.init_state(prng_key=jax.random.PRNGKey(0), model_params=None)
            self.assertCountEqual(state.keys(), ["prng_key"])

            def update(model_collection):
                # pylint: disable-next=protected-access
                accumulator._process_summaries(model_collection.summaries)

            def collection_from(summaries):
                return OutputCollection(summaries, state_updates={}, module_outputs={})

            update(
                collection_from(
                    summaries=dict(
                        a=WeightedScalar(1, 1),
                        b=dict(b1=WeightedScalar(2, 6), b2=WeightedScalar(3, 12)),
                        c=WeightedScalar(2, 4),
                        d=dict(d1=WeightedScalar(1, 1), d2=WeightedScalar(6, 12)),
                    )
                ),
            )
            update(
                collection_from(
                    summaries=dict(
                        a=WeightedScalar(3, 1),
                        b=dict(b1=WeightedScalar(12, 24), b2=WeightedScalar(15, 3)),
                        c=WeightedScalar(1, 1),
                        d=dict(d1=WeightedScalar(2, 4), d2=WeightedScalar(1, 4)),
                    )
                ),
            )
            self.assertEqual(
                {
                    "a": WeightedScalar(2.0, 2),
                    "b": {"b1": WeightedScalar(10.0, 30), "b2": WeightedScalar(5.4, 15)},
                    "c": WeightedScalar(1.8, 5),
                    "d": {"d1": WeightedScalar(1.8, 5), "d2": WeightedScalar(4.75, 16)},
                },
                accumulator.get_summaries(model_params={}, state=state, all_forward_outputs=[]),
            )


class CompositeMetricCalculatorTest(TestCase):
    def setup_model_and_calculator_inputs(
        self,
        calculator_cfg: CompositeMetricCalculator.Config,
    ) -> tuple[CompositeMetricCalculator, NestedTensor, NestedTensor, list[NestedTensor]]:
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            model: DummyModel = (
                DummyModel.default_config().set(name="model").instantiate(parent=None)
            )
            model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(0))
            partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
            )

            calculator: CompositeMetricCalculator = calculator_cfg.instantiate(
                parent=None, model=model, model_param_partition_specs=partition_specs
            )

            state = calculator.init_state(prng_key=jax.random.PRNGKey(1), model_params=model_params)
            inputs = (
                DummyInput.default_config()
                .set(name="input", is_training=False, batch_size=4, total_num_batches=3)
                .instantiate(parent=None)
            )
            outputs = []
            for batch in inputs:
                forward_output = calculator.forward(batch, model_params=model_params, state=state)
                state = forward_output["state"]
                outputs.append(forward_output["output"])

            return calculator, model_params, state, outputs

    def test_multiple_summaries_saved(self):
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            calculator_cfg = CompositeMetricCalculator.default_config().set(
                name="calc",
                metric_calculators=dict(
                    calculator1=DummyMetricCalculator.default_config().set(
                        metrics=["mean_prediction"],
                    ),
                    calculator2=DummyMetricCalculator.default_config().set(
                        metrics=["min_prediction"],
                    ),
                ),
            )
            calculator, model_params, state, outputs = self.setup_model_and_calculator_inputs(
                calculator_cfg
            )

            summaries = calculator.get_summaries(
                model_params=model_params, state=state, all_forward_outputs=outputs
            )

            self.assertIn("calculator1/mean_prediction", summaries)
            self.assertIn("calculator2/min_prediction", summaries)

    def test_different_calculator_same_metric_name(self):
        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            calculator_cfg = CompositeMetricCalculator.default_config().set(
                name="calc",
                metric_calculators=dict(
                    calculator1=DummyMetricCalculator.default_config().set(
                        metrics=["mean_prediction"],
                    ),
                    calculator2=DummyMetricCalculator.default_config().set(
                        metrics=["mean_prediction"],
                    ),
                ),
            )

            calculator, model_params, state, outputs = self.setup_model_and_calculator_inputs(
                calculator_cfg
            )

            summaries = calculator.get_summaries(
                model_params=model_params, state=state, all_forward_outputs=outputs
            )

            # Check we can get mean_prediction without problems.
            self.assertIn("calculator1/mean_prediction", summaries)
            self.assertIn("calculator2/mean_prediction", summaries)
            self.assertEqual(
                summaries["calculator1/mean_prediction"].mean,
                summaries["calculator2/mean_prediction"].mean,
            )
            self.assertEqual(
                summaries["calculator1/mean_prediction"].weight,
                summaries["calculator2/mean_prediction"].weight,
            )

    @parameterized.parameters(
        # dependencies defines a mapping (src, dst, dst_key). If a calculator is listed as a src, we
        # expect its outputs to appear in the input batch for dst, under the provided dst_key.
        dict(
            # Each calculator reads from calculator1.
            dependencies=[
                # dst regex expands to calculator2 and calculator3.
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator[2-3]",
                    dst_key="forward_outputs",
                ),
                # The edge to calculator4 can be named differently.
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator4",
                    dst_key="forward_outputs2",
                ),
            ],
            # Define the expected input structure (excluding original inputs).
            expected=dict(
                calculator1={},
                calculator2=dict(forward_outputs="calculator1_outputs"),
                calculator3=dict(forward_outputs="calculator1_outputs"),
                calculator4=dict(forward_outputs2="calculator1_outputs"),
            ),
        ),
        dict(
            # Omit dst_key to use src calculator name.
            dependencies=[
                CompositeMetricCalculator.Dependency(src="calculator1", dst="calculator[2-4]"),
            ],
            # Define the expected input structure (excluding original inputs).
            expected=dict(
                calculator1={},
                calculator2=dict(calculator1="calculator1_outputs"),
                calculator3=dict(calculator1="calculator1_outputs"),
                calculator4=dict(calculator1="calculator1_outputs"),
            ),
        ),
        dict(
            # calculator1 reads from calculator2;
            # calculator2 reads from calculator3 and 4.
            dependencies=[
                CompositeMetricCalculator.Dependency(
                    src="calculator2", dst="calculator1", dst_key="calculator2_outputs"
                ),
                CompositeMetricCalculator.Dependency(
                    src="calculator3", dst="calculator2", dst_key="calculator3_outputs"
                ),
                CompositeMetricCalculator.Dependency(
                    src="calculator4", dst="calculator2", dst_key="calculator4_outputs"
                ),
            ],
            # Define the expected input structure:
            # calculator1's input should contain calculator2's outputs.
            # calculator2's input should contain calculator3/4's outputs.
            expected=dict(
                calculator1=dict(
                    calculator2_outputs="calculator2_outputs",
                ),
                calculator2=dict(
                    calculator3_outputs="calculator3_outputs",
                    calculator4_outputs="calculator4_outputs",
                ),
                calculator3={},
                calculator4={},
            ),
        ),
        dict(
            # Edges form a cycle, should fail.
            dependencies=[
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator2",
                ),
                CompositeMetricCalculator.Dependency(
                    src="calculator2",
                    dst="calculator3",
                ),
                CompositeMetricCalculator.Dependency(
                    src="calculator3",
                    dst="calculator1",
                ),
            ],
            expected=graphlib.CycleError("cycle"),
        ),
        dict(
            # Duplicate edges, should fail.
            dependencies=[
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator2",
                    dst_key="forward_outputs",
                ),
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator2",
                    dst_key="forward_outputs",
                ),
            ],
            expected=ValueError(r"duplicate edge \(calculator1, calculator2\)"),
        ),
        dict(
            # Multiple sources produce the same key, should fail.
            dependencies=[
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator2",
                    dst_key="forward_outputs",
                ),
                CompositeMetricCalculator.Dependency(
                    src="calculator3",
                    dst="calculator4",
                    dst_key="forward_outputs",
                ),
            ],
            expected=ValueError("calculator1 and calculator3 produce key forward_outputs"),
        ),
        dict(
            # Source produces a key that `calculator2` already has, should fail.
            dependencies=[
                CompositeMetricCalculator.Dependency(
                    src="calculator1",
                    dst="calculator2",
                    dst_key="inputs",
                ),
            ],
            expected=ValueError("calculator2 already has key inputs"),
        ),
    )
    def test_dependencies(self, dependencies, expected: Union[Nested[Tensor], Exception]):
        def check_input_batch(name: str, input_batch: Nested[Tensor]):
            if not isinstance(expected, Exception):
                expected_outputs = expected.get(name, {})
                for k, v in expected_outputs.items():
                    self.assertNestedEqual(v, input_batch[k])

        class OutputRuleCalculator(BaseMetricCalculator):
            """A dummy calculator that checks the input batch and returns a fixed output."""

            def init_state(self, *, prng_key, model_params):
                del prng_key, model_params
                return {}

            def forward(self, input_batch, *, model_params, state):
                del model_params, state
                check_input_batch(self.name, input_batch)
                return dict(output=f"{self.name}_outputs", state={})

            def get_summaries(self, *, model_params, state, all_forward_outputs):
                del model_params, state, all_forward_outputs
                return {}

        with jax.sharding.Mesh(mesh_utils.create_device_mesh((1, 1)), ("data", "model")):
            calculator_cfg = CompositeMetricCalculator.default_config().set(
                name="calc",
                metric_calculators=dict(
                    calculator1=OutputRuleCalculator.default_config(),
                    calculator2=OutputRuleCalculator.default_config(),
                    calculator3=OutputRuleCalculator.default_config(),
                    calculator4=OutputRuleCalculator.default_config(),
                ),
                dependencies=dependencies,
            )
            if isinstance(expected, Exception):
                with self.assertRaisesRegex(type(expected), str(expected)):
                    self.setup_model_and_calculator_inputs(calculator_cfg)
            else:
                self.setup_model_and_calculator_inputs(calculator_cfg)


class GlobalEvalerTest(TestCase):
    def test_alt_predict(self):
        with jax.sharding.Mesh(
            jax.experimental.mesh_utils.create_device_mesh((1, 1)), ("data", "model")
        ):
            model = DummyModel.default_config().set(name="model").instantiate(parent=None)
            model_param_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
            )
            calculator_cfg = GlobalMetricCalculator.default_config().set(
                predict_method="alt_predict", predict_input_field="side_input"
            )

            calculator = calculator_cfg.set(name="calculator").instantiate(
                parent=None, model=model, model_param_partition_specs=model_param_partition_specs
            )
            model_params = pjit(
                model.initialize_parameters_recursively,
                in_shardings=(None,),
                out_shardings=model_param_partition_specs,
            )(jax.random.PRNGKey(0))

            state = calculator.init_state(prng_key=jax.random.PRNGKey(0), model_params=model_params)
            input_batch = {"side_input": {"mock_output": {"alt_predicted": jnp.ones((1, 2, 3))}}}
            forward_outputs = calculator.forward(
                input_batch, model_params=model_params, state=state
            )
            predict_output = forward_outputs["output"].predict_outputs
            self.assertNestedAllClose({"alt_predicted": jnp.ones((1, 2, 3))}, predict_output)


if __name__ == "__main__":
    absltest.main()
