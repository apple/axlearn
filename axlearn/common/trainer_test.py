# Copyright © 2023 Apple Inc.

"""Tests SpmdTrainer."""

import copy
import dataclasses
import math

# pylint: disable=no-self-use
import os
import os.path
import shutil
import tempfile
import unittest
from collections.abc import Sequence
from typing import Any, Callable, Literal, Optional

import chex
import jax
import jax.random
import numpy as np
import pytest
import tensorflow as tf
from absl import flags, logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax._src.interpreters import pxla
from jax.experimental import checkify
from jax.sharding import PartitionSpec

from axlearn.common import (
    debug_utils,
    flax_struct_test,
    layers,
    learner,
    optimizers,
    param_init,
    test_utils,
    utils_spmd,
)
from axlearn.common.base_layer import NestedParameterSpec, ParameterSpec, RematSpec
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import (
    Checkpointer,
    CheckpointPolicy,
    every_n_steps_and_last_policy,
    every_n_steps_policy,
)
from axlearn.common.config import REQUIRED, Required, config_class, config_for_function
from axlearn.common.evaler import SpmdEvaler
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps_policy
from axlearn.common.input_base import Input
from axlearn.common.input_dispatch import SpmdInputDispatcher
from axlearn.common.learner import UpdateType, should_update_with_optimizers
from axlearn.common.module import Module
from axlearn.common.monitoring.device_monitor import DeviceMonitor
from axlearn.common.state_builder import Builder as TrainerStateBuilder
from axlearn.common.trainer import SpmdTrainer, TrainerState, aot_model_analysis, select_mesh_config
from axlearn.common.trainer_config_modifier import (
    ChainConfigModifier,
    GradientAccumulationModifier,
    MeshShapeModifier,
    RematSpecModifier,
)
from axlearn.common.utils import (
    Nested,
    NestedTensor,
    Tensor,
    as_tensor,
    flatten_items,
    match_regex_rules,
    tree_paths,
)

FLAGS = flags.FLAGS

NUM_CLASSES = 16

os.environ["TPU_SKIP_MDS_QUERY"] = "1"


class DummyInput(Input):
    """A dummy input."""

    @config_class
    class Config(Input.Config):
        """Configures DummyInput."""

        is_training: Required[bool] = REQUIRED
        batch_size: int = 8  # The batch size.
        total_num_batches: Optional[int] = None  # The total number of batches. If None, unlimited.
        include_labels: bool = True

    def __init__(self, cfg: Config, *, parent=None):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if cfg.is_training != (cfg.total_num_batches is None):
            raise ValueError("total_num_batches should be None iff is_training")
        self._tmp_dir = tempfile.mkdtemp()
        self._record_file = self._write_records(self._tmp_dir)

    @property
    def _batch_size(self):
        cfg: DummyInput.Config = self.config
        if "input_dispatcher" in self.children:
            if cfg.batch_size != self.input_dispatcher.feed_logical_batch_size:
                logging.info(
                    "Replacing batch_size=%s with feed_logical_batch_size=%s.",
                    cfg.batch_size,
                    self.input_dispatcher.feed_logical_batch_size,
                )
                return self.input_dispatcher.feed_logical_batch_size
        return cfg.batch_size

    def __del__(self):
        shutil.rmtree(self._tmp_dir)

    def _write_records(self, tmp_dir) -> str:
        """Writes records to `tmp_dir` and returns the filename."""
        cfg: DummyInput.Config = self.config
        filename = os.path.join(tmp_dir, "records")
        with tf.io.TFRecordWriter(filename) as file_writer:
            for batch in self._datagen(cfg.total_num_batches or 5):
                feature_dict = {
                    "image": tf.train.Feature(
                        float_list=tf.train.FloatList(value=batch["image"].reshape([-1]).tolist())
                    ),
                }
                if "label" in batch:
                    feature_dict["label"] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=batch["label"].tolist())
                    )
                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                file_writer.write(example.SerializeToString())
        return filename

    def _datagen(self, total_num_batches):
        cfg: DummyInput.Config = self.config
        num_batches = 0
        prng_key = jax.random.PRNGKey(1)

        while num_batches < total_num_batches:
            num_batches += 1
            prng_key, image_key, label_key = jax.random.split(prng_key, 3)
            batch = dict(
                image=jax.random.randint(
                    image_key,
                    shape=[self._batch_size, 224, 224, 3],
                    minval=0,
                    maxval=256,
                    dtype=np.int32,
                ).astype(np.float32),
            )
            if cfg.include_labels:
                batch["label"] = jax.random.randint(
                    label_key,
                    shape=[self._batch_size],
                    minval=0,
                    maxval=NUM_CLASSES,
                    dtype=np.int32,
                )
            yield batch

    def dataset(self) -> tf.data.Dataset:
        cfg = self.config
        features = {
            "image": tf.io.FixedLenFeature(shape=(self._batch_size, 224, 224, 3), dtype=tf.float32),
        }
        if cfg.include_labels:
            features["label"] = tf.io.FixedLenFeature(shape=(self._batch_size,), dtype=tf.int64)
        ds = tf.data.TFRecordDataset(filenames=[self._record_file]).map(
            lambda record_bytes: tf.io.parse_single_example(record_bytes, features=features)
        )
        if cfg.total_num_batches is None:
            ds = ds.repeat()
        return ds

    def __iter__(self):
        # Use a different __iter__ than iter(self.dataset()), to test that input iter can be
        # checkpointed properly even with a custom __iter__ (note that a custom __iter__ is not
        # guaranteed to be savable).
        yield from self.dataset()

    def element_spec(self):
        return jax.tree.map(
            lambda tf_spec: jax.ShapeDtypeStruct(
                shape=tf_spec.shape, dtype=tf_spec.dtype.as_numpy_dtype
            ),
            self.dataset().element_spec,
        )


class DummyModel(BaseModel):
    """A dummy model."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyModel."""

        # Whether to explicitly init dummy state to test pruning backwards compat.
        init_dummy_state: bool = False
        linear: layers.Linear.Config = layers.Linear.default_config()

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "fc",
            cfg.linear.set(
                input_dim=3,
                output_dim=NUM_CLASSES,
                bias=True,
                param_partition_spec=(None, "model"),
            ),
        )
        self.predict_dtypes = []

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        specs = super().create_parameter_specs_recursively()
        if self.config.init_dummy_state:
            specs["dummy"] = {"nested": {"empty": {}}}  # type: ignore
        return dict(sorted(specs.items()))  # type: ignore

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]]
    ) -> NestedTensor:
        params = super().initialize_parameters_recursively(prng_key, prebuilt=prebuilt)
        if self.config.init_dummy_state:
            params["dummy"] = {"nested": {"empty": {}}}
        return params

    # We drop the kwargs from BaseModel, since they aren't used here.
    # pylint: disable-next=arguments-differ
    def forward(self, input_batch: NestedTensor):
        self.add_state_update(
            "dummy",
            {
                "nested": {
                    "empty": {},
                },
            },
        )

        # [batch, 3].
        logits = self.predict(input_batch)
        label: Tensor = input_batch["label"]
        loss = (
            -(jax.nn.log_softmax(logits) * jax.nn.one_hot(label, NUM_CLASSES, dtype=logits.dtype))
            .sum(axis=-1)
            .mean()
        )
        return loss, {"prng_key": self.prng_key}

    def predict(self, input_batch: NestedTensor) -> Tensor:
        image = input_batch["image"]
        self.predict_dtypes.append(image.dtype)
        hidden = image.mean(axis=(1, 2))
        return self.fc(hidden)


class NonPartitionedModel(BaseModel):
    """A dummy replicated model."""

    def __init__(self, cfg: BaseModel.Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        self._add_child(
            "fc",
            layers.Linear.default_config().set(
                input_dim=3,
                output_dim=NUM_CLASSES,
                bias=True,
                param_partition_spec=(None, "model"),
            ),
        )

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        return None

    # We drop the kwargs from BaseModel, since they aren't used here.
    # pylint: disable-next=arguments-differ
    def forward(self, input_batch: NestedTensor):
        # [batch, 3].
        logits = self.predict(input_batch)
        label: Tensor = input_batch["label"]
        loss = (
            -(jax.nn.log_softmax(logits) * jax.nn.one_hot(label, NUM_CLASSES, dtype=logits.dtype))
            .sum(axis=-1)
            .mean()
        )
        return loss, {"prng_key": self.prng_key}

    def predict(self, input_batch: NestedTensor) -> Tensor:
        image = input_batch["image"]
        hidden = image.mean(axis=(1, 2))
        return self.fc(hidden)


class DummyStateBuilder(TrainerStateBuilder):
    """A dummy builder that "builds" state from fixed values."""

    @config_class
    class Config(Module.Config):
        step: Required[int] = REQUIRED
        model_state: Required[Callable[[], NestedTensor]] = REQUIRED
        input_state_type: Required[TrainerStateBuilder.StateType] = REQUIRED

    def input_state_type(self) -> TrainerStateBuilder.StateType:
        return self.config.input_state_type

    def __call__(self, state: TrainerStateBuilder.State) -> TrainerStateBuilder.State:
        built_state = TrainerStateBuilder.State(
            step=state.step,
            trainer_state=state.trainer_state,
            built_keys=copy.deepcopy(state.built_keys),
        )
        built_keys = {f"model/{key}" for key, _ in flatten_items(built_state.trainer_state.model)}
        logging.info("built_keys=%s", built_keys)
        return TrainerStateBuilder.State(
            step=self.config.step,
            trainer_state=TrainerState(
                prng_key=built_state.trainer_state.prng_key,
                model=self.config.model_state(),
                learner=built_state.trainer_state.learner,
            ),
            built_keys=built_keys,
        )


class TrainerTest(test_utils.TestCase):
    """Tests SpmdTrainer."""

    def _trainer_config(self, input_cfg: Optional[Input.Config] = None) -> SpmdTrainer.Config:
        if input_cfg is None:
            input_cfg = DummyInput.default_config()
        return SpmdTrainer.default_config().set(
            name="base_trainer",
            vlog=3,
            dir=tempfile.mkdtemp(),  # TODO(markblee): Use a context.
            mesh_axis_names=("data", "model"),
            mesh_shape=(1, 1),
            model=DummyModel.default_config().set(
                dtype=jnp.float32, param_init=param_init.DefaultInitializer.default_config()
            ),
            input=input_cfg,
            learner=learner.Learner.default_config().set(
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1,
                    decouple_weight_decay=True,
                    momentum=0.9,
                    weight_decay=1e-4,
                ),
            ),
            max_step=6,
            checkpointer=Checkpointer.default_config().set(
                save_policy=config_for_function(every_n_steps_policy).set(n=5)
            ),
            evalers=dict(
                eval_dummy=SpmdEvaler.default_config().set(
                    input=input_cfg.clone(total_num_batches=2),
                ),
            ),
        )

    # Previously there was a test for pruning backwards compatibility with no pruning here.
    # This has been removed as there appear to be no models in the source tree that
    # do not use pruning, and keeping this test required maintaining the `prune_empty_state_updates`
    # config option.

    # A similar test exists for evaler.
    # pylint: disable=duplicate-code
    @parameterized.parameters(
        {"platform": "cpu", "mesh_shape": (1, 1), "step_dtype": None},
        {"platform": "cpu", "mesh_shape": (1, 1), "step_dtype": jnp.bfloat16},
        {"platform": "cpu", "mesh_shape": (1, 1), "step_dtype": None, "ema_decay": 0.99},
        {"platform": "cpu", "mesh_shape": (1, 1), "step_dtype": None, "start_trace_steps": (1, 5)},
        {"platform": "tpu", "mesh_shape": (8, 1), "step_dtype": jnp.bfloat16},
        {"platform": "tpu", "mesh_shape": (2, 4), "step_dtype": jnp.float32},
    )
    # pylint: enable=duplicate-code
    def test_trainer(
        self,
        platform,
        mesh_shape,
        step_dtype,
        *,
        ema_decay=None,
        start_trace_steps=tuple(),
    ):
        if not test_utils.is_supported_platform(platform):
            return
        cfg = SpmdTrainer.default_config().set(name="test_trainer", train_dtype=step_dtype)
        cfg.dir = tempfile.mkdtemp()
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = mesh_shape
        cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=True,
                momentum=0.9,
                weight_decay=1e-4,
            )
        )
        if ema_decay:
            cfg.learner.ema.decay = ema_decay
        if start_trace_steps:
            cfg.start_trace_steps = start_trace_steps
        evaler_cfg = SpmdEvaler.default_config().set(
            input=DummyInput.default_config().set(total_num_batches=2),
            eval_dtype=step_dtype,
        )
        evaler_cfg.summary_writer.vlog = 5
        cfg.evalers = dict(eval_dummy=evaler_cfg)
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_policy).set(n=5)
        cfg.summary_writer.vlog = 5
        cfg.max_step = 12
        cfg.watchdog_timeout_seconds = 0.1
        cfg.device_monitor = DeviceMonitor.default_config().set(check_interval_in_sec=0.1)
        cfg.vlog = 2
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        output_a = trainer.run(prng_key=jax.random.PRNGKey(123))
        if ema_decay is None:
            self.assertIs(trainer.model_params_for_eval(), trainer.trainer_state.model)
        else:
            self.assertIs(trainer.model_params_for_eval(), trainer.trainer_state.learner["ema"].ema)
        # Check we did the compute in step_dtype.
        self.assertEqual(output_a["loss"].dtype, step_dtype or cfg.model.dtype)
        # Check that we also did the eval step in step_dtype.
        self.assertEqual(
            len(
                [el for el in trainer.model.predict_dtypes if el == (step_dtype or cfg.model.dtype)]
            ),
            2,  # Once for the train step trace, once for the eval_step trace.
        )
        with open(os.path.join(cfg.dir, "trainer_state_tree.txt"), encoding="utf-8") as f:
            self.assertStartsWith(f.read(), "PyTreeDef(CustomNode(namedtuple[TrainerState], [*, ")

        with open(os.path.join(cfg.dir, "model_analysis.txt"), encoding="utf-8") as f:
            self.assertStartsWith(
                f.read(), "##################### Model analysis #####################"
            )

        if start_trace_steps:
            trace_dir = os.path.join(cfg.dir, "summaries", "train_train", "plugins", "profile")
            profile_files = []
            for root, unused_dirs, files in os.walk(trace_dir):
                for name in files:
                    profile_files.append(os.path.join(root, name))
            print(f"profile_files={profile_files}")
            self.assertNotEmpty(profile_files)

        trainer2: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer2.mesh():
            self.assertEqual(10, trainer2.restore_checkpoint())

        # Re-create 'trainer'.
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        # Since we will be resuming from the checkpoint at step 10, a different prng_key doesn't
        # matter.
        output_b = trainer.run(prng_key=jax.random.PRNGKey(456))
        # Check we did the training compute in step_dtype.
        self.assertEqual(output_b["loss"].dtype, step_dtype or cfg.model.dtype)
        # Check that we also did the eval step in step_dtype.
        self.assertEqual(
            len(
                [el for el in trainer.model.predict_dtypes if el == (step_dtype or cfg.model.dtype)]
            ),
            2,  # Once for the train step trace, once for the eval_step trace.
        )
        # The prng_key per step is deterministic.
        np.testing.assert_array_equal(output_a["aux"]["prng_key"], output_b["aux"]["prng_key"])

    @parameterized.product(
        [{"platform": "cpu", "mesh_shape": (1, 1)}, {"platform": "tpu", "mesh_shape": (4, 1)}],
        enable_python_cache=[True, False],
    )
    # pylint: enable=duplicate-code
    def test_xsc_check_policy_and_compilation_cache(
        self,
        *,
        platform,
        mesh_shape,
        enable_python_cache,
    ):
        if not test_utils.is_supported_platform(platform):
            return
        cfg: SpmdTrainer.Config = SpmdTrainer.default_config().set(
            name="test_trainer", train_dtype=jnp.bfloat16
        )
        cfg.dir = tempfile.mkdtemp()
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = mesh_shape
        cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=True,
                momentum=0.9,
                weight_decay=1e-4,
            )
        )
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_policy).set(n=100)
        cfg.summary_writer.vlog = 5
        cfg.max_step = 12
        cfg.vlog = 2
        # Set XSC policy.
        cfg.xsc_check_policy = lambda step: (step in [7, 8])
        cfg.cache_compiled_train_step = enable_python_cache

        # Test training run.
        trainer: SpmdTrainer = cfg.set(max_step=12).instantiate(parent=None)

        compiled_with_options_call_count = [0]

        original_compile_train_step_fn = trainer.compile_train_step

        def mock_compile_train_step(*args, compiler_options=None, **kwargs):
            if compiler_options:
                compiled_with_options_call_count[0] += 1
            return original_compile_train_step_fn(
                *args, compiler_options=compiler_options, **kwargs
            )

        with unittest.mock.patch.object(
            trainer, "compile_train_step", side_effect=mock_compile_train_step
        ) as mocked_compile_fn:
            # pylint: disable=protected-access
            start_cache_hits = pxla._cached_lowering_to_hlo.cache_info().hits
            output_a = trainer.run(prng_key=jax.random.PRNGKey(123))
            end_cache_hits = pxla._cached_lowering_to_hlo.cache_info().hits
            # pylint: enable=protected-access
            if platform == "tpu":
                if not enable_python_cache:
                    # As of Jax >= 0.6.0, enable_python_cache=False no longer affects the
                    # AOT compilation path. We now expect the cache hits to be 0
                    if jax.__version__ >= "0.6.0":
                        pytest.skip(
                            # pylint: disable-next=line-too-long
                            "AOT compilation path is not affected by 'enable_python_cache' with Jax >= 0.6.0"
                        )
                    # We expect to have hit the lowering cache on all but one step.
                    self.assertEqual(end_cache_hits - start_cache_hits, cfg.max_step - 1)
                    self.assertEqual(mocked_compile_fn.call_count, cfg.max_step)
                else:
                    # We expect to have hit the lowering cache on xsc steps.
                    self.assertEqual(end_cache_hits - start_cache_hits, 2)
                    self.assertEqual(mocked_compile_fn.call_count, 3)
                # Should have been called with compile options on two steps.
                self.assertEqual(compiled_with_options_call_count[0], 2)
            else:
                if not enable_python_cache:
                    if jax.__version__ >= "0.6.0":
                        pytest.skip(
                            # pylint: disable-next=line-too-long
                            "AOT compilation path is not affected by 'enable_python_cache' with Jax >= 0.6.0"
                        )
                    self.assertEqual(end_cache_hits - start_cache_hits, cfg.max_step - 1)
                    self.assertEqual(mocked_compile_fn.call_count, cfg.max_step)
                else:
                    # We won't hit any cache since we have python cache.
                    self.assertEqual(end_cache_hits - start_cache_hits, 0)
                    self.assertEqual(mocked_compile_fn.call_count, 1)
                # XSC check should be disabled.
                self.assertEqual(compiled_with_options_call_count[0], 0)

        # Test with XSC check disabled.
        cfg2 = cfg.clone().set(xsc_check_policy=None)
        trainer2: SpmdTrainer = cfg2.instantiate(parent=None)
        output_b = trainer2.run(prng_key=jax.random.PRNGKey(123))

        # The prng_key per step is deterministic whether we run with XSC or not.
        np.testing.assert_array_equal(output_a["aux"]["prng_key"], output_b["aux"]["prng_key"])

    @parameterized.parameters(
        {"platform": "cpu", "mesh_shape": (1, 1)},
        {"platform": "tpu", "mesh_shape": (4, 1)},
        {"platform": "gpu", "mesh_shape": (8, 1)},
    )
    # pylint: enable=duplicate-code
    def test_compile_train_step(self, *, platform, mesh_shape):
        if not test_utils.is_supported_platform(platform):
            self.skipTest(f"Unsupported config: {platform=}, {mesh_shape=}.")
        cfg = SpmdTrainer.default_config().set(name="test_trainer", train_dtype=jnp.bfloat16)
        cfg.dir = tempfile.mkdtemp()
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = mesh_shape
        cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=True,
                momentum=0.9,
                weight_decay=1e-4,
            )
        )
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        compiled_without_args = trainer.compile_train_step()
        # pylint: disable=protected-access
        input_batch = jax.tree_util.tree_map(
            jnp.array, next(trainer.input.batches(trainer._input_iter))
        )
        # pylint: enable=protected-access
        compiled_with_input_batch = trainer.compile_train_step(input_batch=input_batch)
        # In a single-host environment, both compiled functions should match.
        self.assertEqual(compiled_without_args.as_text(), compiled_with_input_batch.as_text())
        self.assertEqual(
            aot_model_analysis(compiled_without_args),
            aot_model_analysis(compiled_with_input_batch),
        )

        # A version compiled with non-default compiled args should be different.
        compiled_with_compiler_options = trainer.compile_train_step(
            compiler_options=dict(xla_embed_ir_in_executable=True, xla_dump_max_hlo_modules=200)
        )
        self.assertNotEqual(compiled_without_args, compiled_with_compiler_options)

        # Validate that passing full trainer state is the same as compiled without args.
        compiled_with_trainer_state_and_input_batch = trainer.compile_train_step(
            trainer_state=trainer.trainer_state, input_batch=input_batch
        )
        self.assertEqual(
            compiled_without_args.as_text(), compiled_with_trainer_state_and_input_batch.as_text()
        )
        self.assertEqual(
            aot_model_analysis(compiled_without_args),
            aot_model_analysis(compiled_with_trainer_state_and_input_batch),
        )

    @parameterized.parameters(
        {"return_evaler_summaries": None},
        {"return_evaler_summaries": True},
        {"return_evaler_summaries": False},
        {"return_evaler_summaries": {"wrong"}},
        {"return_evaler_summaries": {"eval_dummy"}},
        {"return_evaler_summaries": {"eval_dummy", "eval_dummy2"}},
    )
    def test_return_evaler_summaries(self, return_evaler_summaries):
        step_dtype = None
        cfg = SpmdTrainer.default_config().set(
            name="test_return_evaler_summaries_trainer", train_dtype=step_dtype
        )
        cfg.dir = tempfile.mkdtemp()
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = (1, 1)
        cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=True,
                momentum=0.9,
                weight_decay=1e-4,
            )
        )
        evaler_cfg = SpmdEvaler.default_config().set(
            input=DummyInput.default_config().set(total_num_batches=2),
            eval_dtype=step_dtype,
            eval_policy=config_for_function(eval_every_n_steps_policy).set(n=10),
        )
        evaler_cfg.summary_writer.vlog = 5
        cfg.evalers = dict(eval_dummy=evaler_cfg, eval_dummy2=evaler_cfg.clone())
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_policy).set(n=5)
        cfg.summary_writer.vlog = 5
        cfg.max_step = 3
        cfg.watchdog_timeout_seconds = 0.1
        cfg.vlog = 2
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        # Check exception when return_evaler_summaries contains names that don't match evalers.
        if isinstance(return_evaler_summaries, set) and "wrong" in return_evaler_summaries:
            with self.assertRaises(ValueError):
                trainer.run(
                    prng_key=jax.random.PRNGKey(123),
                    return_evaler_summaries=return_evaler_summaries,
                )
            return
        else:
            output = trainer.run(
                prng_key=jax.random.PRNGKey(123), return_evaler_summaries=return_evaler_summaries
            )
        if return_evaler_summaries is None or return_evaler_summaries is False:
            self.assertTrue("evaler_summaries" not in output)
        else:
            self.assertTrue("evaler_summaries" in output)
            # evalers not force run will have None as value in evaler_summaries.
            if return_evaler_summaries is True:
                expected_non_empty_keys = {"eval_dummy", "eval_dummy2"}
            elif isinstance(return_evaler_summaries, set):
                expected_non_empty_keys = return_evaler_summaries
            else:
                raise ValueError(
                    f"return_evaler_summaries {return_evaler_summaries} not supported!"
                )
            self.assertTrue(
                expected_non_empty_keys
                == {k for k, v in output["evaler_summaries"].items() if v is not None}
            )

    def test_stop_on_exception(self):
        """Test that trainer exits cleanly if there's an exception in the main loop."""
        if not test_utils.is_supported_platform("cpu"):
            return

        class RaiseInput(DummyInput):
            """A dummy input that raises an exception after N batches."""

            @config_class
            class Config(DummyInput.Config):
                raise_on_batch: Required[int] = REQUIRED

            def dataset(self):
                cfg = self.config
                yield from super().dataset().take(cfg.raise_on_batch - 1)
                raise ValueError(f"Raising on batch {cfg.raise_on_batch}")

        cfg = self._trainer_config().set(max_step=3, watchdog_timeout_seconds=10)
        cfg.input = RaiseInput.default_config().set(raise_on_batch=cfg.max_step - 1)
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        with self.assertRaisesRegex(ValueError, f"Raising on batch {cfg.input.raise_on_batch}"):
            trainer.run(prng_key=jax.random.PRNGKey(123))
        # pylint: disable=protected-access
        self.assertTrue(trainer._watchdog_stopping.is_set())
        self.assertIsNone(trainer._watchdog_thread)
        # pylint: enable=protected-access

    def test_non_partitioned_model(self):
        if jax.default_backend() != "cpu":
            return
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.dir = tempfile.mkdtemp()
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = (1, 1)
        cfg.model = NonPartitionedModel.default_config().set(
            dtype=jnp.float32, param_init=param_init.DefaultInitializer.default_config()
        )
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=True,
                momentum=0.9,
                weight_decay=1e-4,
            )
        )
        cfg.evalers = dict(
            eval_dummy=SpmdEvaler.default_config().set(
                input=DummyInput.default_config().set(total_num_batches=2),
            )
        )
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_policy).set(n=5)
        cfg.max_step = 12
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        trainer.run(prng_key=jax.random.PRNGKey(123))

    @parameterized.parameters(
        TrainerStateBuilder.StateType.TENSORS,
        TrainerStateBuilder.StateType.TENSOR_SPECS,
    )
    def test_restore_from_builder(self, builder_input_state_type):
        cfg = SpmdTrainer.default_config().set(
            name="test_trainer",
            dir=tempfile.mkdtemp(),
            mesh_axis_names=("data", "model"),
            mesh_shape=(1, 1),
            model=DummyModel.default_config().set(dtype=jnp.float32),
            input=DummyInput.default_config(),
            learner=learner.Learner.default_config().set(
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1,
                    decouple_weight_decay=True,
                    momentum=0.9,
                    weight_decay=1e-4,
                ),
            ),
            max_step=12,
            checkpointer=Checkpointer.default_config().set(
                save_policy=config_for_function(every_n_steps_policy).set(n=5),
            ),
            evalers=dict(
                eval_dummy=SpmdEvaler.default_config().set(
                    input=DummyInput.default_config().set(total_num_batches=2),
                ),
            ),
        )
        # Construct a dummy builder that loads some pre-determined state.
        new_model_state = (
            cfg.model.clone(name="test_model")
            .instantiate(parent=None)
            .initialize_parameters_recursively(jax.random.PRNGKey(4321), prebuilt=None)
        )
        new_model_state = as_tensor(new_model_state)
        state_builder = DummyStateBuilder.default_config().set(
            step=100,
            model_state=lambda: new_model_state,
            input_state_type=builder_input_state_type,
        )

        # Init without builder.
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer.mesh():
            trainer.init(jax.random.PRNGKey(0))
            # Sanity check that the initial model state is not already equal.
            with self.assertRaises(AssertionError):
                # pylint: disable-next=protected-access
                self.assertNestedAllClose(trainer._trainer_state.model, new_model_state)

        # Restore from builder, and check model state and step are now updated.
        cfg.set(init_state_builder=state_builder)
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer.mesh():
            trainer.init(jax.random.PRNGKey(0))
            # pylint: disable-next=protected-access
            self.assertNestedAllClose(trainer._trainer_state.model, new_model_state)
            self.assertEqual(trainer.step, state_builder.step)

    @parameterized.named_parameters(
        ("default", []),
        ("skip_bias", [(".*/bias$", UpdateType.NO_UPDATE)]),
        ("skip_all", [(".*", UpdateType.NO_UPDATE)]),
    )
    def test_should_compute_gradients(self, update_rules):
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = (1, 1)
        cfg.dir = tempfile.mkdtemp()
        cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
        cfg.input = DummyInput.default_config()
        cfg.learner = learner.Learner.default_config().set(
            optimizer=config_for_function(optimizers.sgd_optimizer).set(
                learning_rate=0.1,
                decouple_weight_decay=True,
                momentum=0.9,
                weight_decay=0,
            ),
            update_rules=update_rules,
        )
        evaler_cfg = SpmdEvaler.default_config().set(
            input=DummyInput.default_config().set(total_num_batches=2),
        )
        evaler_cfg.summary_writer.vlog = 5
        cfg.evalers = dict(eval_dummy=evaler_cfg)
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_policy).set(n=5)
        cfg.summary_writer.vlog = 5
        cfg.max_step = 12
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer.mesh():
            trainer.init(prng_key=jax.random.PRNGKey(123))
            init_params = trainer.trainer_state.model
            trainer.run(prng_key=jax.random.PRNGKey(123))
            updated_params = trainer.trainer_state.model
            for (path, init_p), (_, updated_p) in zip(
                flatten_items(init_params), flatten_items(updated_params)
            ):
                if should_update_with_optimizers(
                    match_regex_rules(
                        path, rules=update_rules, default_value=UpdateType.ALL_UPDATES
                    )
                ):
                    self.assertGreater(np.max(np.abs(updated_p - init_p)), 1e-3, msg=path)
                else:
                    np.testing.assert_allclose(init_p, updated_p, err_msg=path)

    @parameterized.parameters(True, False)
    def test_run_builder(self, restore_from_builder: bool):
        model_cfg = DummyModel.default_config().set(dtype=jnp.float32)
        cfg = SpmdTrainer.default_config().set(
            name="test_trainer",
            dir=tempfile.mkdtemp(),
            mesh_axis_names=("data", "model"),
            mesh_shape=(1, 1),
            model=model_cfg,
            input=DummyInput.default_config(),
            learner=learner.Learner.default_config().set(
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1,
                    decouple_weight_decay=True,
                    momentum=0.9,
                    weight_decay=1e-4,
                ),
            ),
            max_step=0,
            checkpointer=Checkpointer.default_config().set(
                save_policy=config_for_function(every_n_steps_policy).set(n=5),
            ),
            evalers=dict(
                eval_dummy=SpmdEvaler.default_config().set(
                    input=DummyInput.default_config().set(total_num_batches=2),
                ),
            ),
            vlog=1,
        )
        if restore_from_builder:
            cfg.set(
                init_state_builder=DummyStateBuilder.default_config().set(
                    step=0,
                    model_state=lambda: (
                        model_cfg.clone(name="test_model")
                        .instantiate(parent=None)
                        .initialize_parameters_recursively(jax.random.PRNGKey(4321), prebuilt=None)
                    ),
                    input_state_type=TrainerStateBuilder.StateType.TENSOR_SPECS,
                ),
            )
            # Initial run should load from builder, if provided.
            trainer: SpmdTrainer = cfg.set(max_step=0).instantiate(parent=None)
            trainer.run(prng_key=jax.random.PRNGKey(123))

            # Validate that the initial state is as expected.
            self.assertNestedAllClose(
                trainer.trainer_state.model, cfg.init_state_builder.model_state()
            )

        # Run until first checkpoint.
        trainer: SpmdTrainer = cfg.set(max_step=6).instantiate(parent=None)
        first_output = trainer.run(prng_key=jax.random.PRNGKey(123))

        assert os.path.exists(os.path.join(cfg.dir, "trainer_state_tree.txt"))
        assert os.path.exists(os.path.join(cfg.dir, "model_analysis.txt"))
        # Make sure checkpoint exists.
        trainer2: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer2.mesh():
            self.assertEqual(5, trainer2.restore_checkpoint())

        # Once checkpoint is reached, next run should read from checkpoint, not builder.
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        second_output = trainer.run(prng_key=jax.random.PRNGKey(456))

        # Make sure the final state matches expected step and PRNGKey from first run.
        self.assertEqual(trainer.step, cfg.max_step)
        self.assertNestedAllClose(first_output["aux"], second_output["aux"])

    @parameterized.product(
        save_input_iterator=[False, True],
        restore_input_iterator=[False, True],
        max_concurrent_gb=[None, 1],
    )
    def test_checkpoint_policy(
        self,
        *,
        save_input_iterator: bool,
        restore_input_iterator: bool,
        max_concurrent_gb: Optional[int],
    ):
        """Test checkpoint policy when evaler and checkpointer run at different cadences."""
        model_cfg = DummyModel.default_config().set(dtype=jnp.float32)

        def checkpoint_if_all_evalers_run(evaler_names: Sequence[str]) -> CheckpointPolicy:
            def fn(*, step: int, evaler_summaries: dict[str, Any]):
                del step
                for evaler_name in evaler_names:
                    if evaler_summaries.get(evaler_name) is None:
                        return False
                return True

            return fn

        cfg: SpmdTrainer.Config = SpmdTrainer.default_config().set(
            name="test_trainer",
            dir=tempfile.mkdtemp(),
            mesh_axis_names=("data", "model"),
            mesh_shape=(1, 1),
            model=model_cfg,
            input=DummyInput.default_config(),
            learner=learner.Learner.default_config().set(
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1, decouple_weight_decay=True
                ),
            ),
            max_step=8,
            checkpointer=Checkpointer.default_config().set(
                save_policy=config_for_function(checkpoint_if_all_evalers_run).set(
                    evaler_names=["eval_every_2", "eval_every_3"]
                ),
            ),
            evalers=dict(
                eval_every_2=SpmdEvaler.default_config().set(
                    input=DummyInput.default_config().set(total_num_batches=1),
                    eval_policy=config_for_function(eval_every_n_steps_policy).set(n=2),
                ),
                eval_every_3=SpmdEvaler.default_config().set(
                    input=DummyInput.default_config().set(total_num_batches=2),
                    eval_policy=config_for_function(eval_every_n_steps_policy).set(n=3),
                ),
            ),
            save_input_iterator=save_input_iterator,
        )
        cfg.checkpointer.storage.max_concurrent_gb = max_concurrent_gb

        # Run trainer.
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        trainer.run(prng_key=jax.random.PRNGKey(123))

        assert os.path.exists(os.path.join(cfg.dir, "trainer_state_tree.txt"))
        assert os.path.exists(os.path.join(cfg.dir, "model_analysis.txt"))
        trainer2: SpmdTrainer = cfg.clone(save_input_iterator=restore_input_iterator).instantiate(
            parent=None
        )
        with trainer2.mesh():
            # We should have checkpointed at step 6, when all evalers ran.
            #
            # Note that we expect `restore_checkpoint` to succeed even if
            # `save_input_iterator` != `restore_input_iterator`, # so that users can turn on/off
            # `save_input_iterator` for models without breaking backwards compatibility.
            self.assertEqual(6, trainer2.restore_checkpoint())

    def test_last_step_checkpoint_policy(self):
        """Test checkpoint policy saving at the last step."""
        model_cfg = DummyModel.default_config().set(dtype=jnp.float32)

        cfg = SpmdTrainer.default_config().set(
            name="test_trainer",
            dir=tempfile.mkdtemp(),
            mesh_axis_names=("data", "model"),
            mesh_shape=(1, 1),
            model=model_cfg,
            input=DummyInput.default_config(),
            learner=learner.Learner.default_config().set(
                optimizer=config_for_function(optimizers.sgd_optimizer).set(
                    learning_rate=0.1, decouple_weight_decay=True
                ),
            ),
            max_step=8,
            checkpointer=Checkpointer.default_config().set(
                save_policy=config_for_function(every_n_steps_and_last_policy).set(
                    n=3,
                    max_step=8,
                ),
            ),
        )

        # Run trainer.
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        trainer.run(prng_key=jax.random.PRNGKey(123))

        assert os.path.exists(os.path.join(cfg.dir, "trainer_state_tree.txt"))
        assert os.path.exists(os.path.join(cfg.dir, "model_analysis.txt"))
        trainer2: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer2.mesh():
            # We should have checkpointed at the last step.
            self.assertEqual(8, trainer2.restore_checkpoint())

    def test_composite_learner(self):
        """Tests composite learner with two sub learners for weight/bias respectively."""
        cfg = SpmdTrainer.default_config().set(name="test_trainer")
        cfg.mesh_axis_names = ("data", "model")
        cfg.mesh_shape = (1, 1)
        cfg.dir = tempfile.mkdtemp()
        cfg.model = DummyModel.default_config().set(dtype=jnp.float32)
        cfg.input = DummyInput.default_config()
        opt1_cfg = config_for_function(optimizers.sgd_optimizer).set(
            learning_rate=0.1,
            decouple_weight_decay=True,
            momentum=0.9,
            weight_decay=0,
        )
        opt2_cfg = config_for_function(optimizers.adamw_optimizer).set(
            learning_rate=1.0, b1=0.9, b2=0.95, eps=1e-6
        )
        learner_rules = [(".*weight.*", "weight"), (".*bias.*", "bias")]

        cfg.learner = learner.CompositeLearner.default_config().set(
            rules=learner_rules,
            learners={
                "weight": learner.Learner.default_config().set(optimizer=opt1_cfg),
                "bias": learner.Learner.default_config().set(optimizer=opt2_cfg),
            },
        )
        evaler_cfg = SpmdEvaler.default_config().set(
            input=DummyInput.default_config().set(total_num_batches=2),
        )
        evaler_cfg.summary_writer.vlog = 5
        cfg.evalers = dict(eval_dummy=evaler_cfg)
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_policy).set(n=5)
        cfg.summary_writer.vlog = 5
        cfg.max_step = 12
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer.mesh():
            trainer.init(prng_key=jax.random.PRNGKey(123))
            init_params = trainer.trainer_state.model
            trainer.run(prng_key=jax.random.PRNGKey(123))
            updated_params = trainer.trainer_state.model
            for (path, init_p), (_, updated_p) in zip(
                flatten_items(init_params), flatten_items(updated_params)
            ):
                self.assertGreater(np.max(np.abs(updated_p - init_p)), 1e-3, msg=path)

    def _dummy_input_checking_model(
        self, global_logical_batch_size: int, partition_spec: PartitionSpec
    ) -> DummyModel.Config:
        """Builds a model that sanity checks its input batch."""

        def check_shape(x: Tensor):
            self.assertEqual(x.shape[0], global_logical_batch_size)

        def check_sharding(path: str, x: Tensor):
            # It's useful to compare normalized PartitionSpecs with `_normalized_spec_for_aval`,
            # e.g. ("data",) vs "data"; so we disable the lint.
            # pylint: disable=protected-access
            if x.shape[0] > 1:
                jax.debug.inspect_array_sharding(
                    x,
                    callback=lambda sharding: self.assertEqual(
                        partition_spec._normalized_spec_for_aval(x.ndim),
                        sharding.spec._normalized_spec_for_aval(x.ndim),
                        msg=f"{path=}, {sharding=}",
                    ),
                )

        class DummyCheckingModel(DummyModel):
            """A dummy model that checks inputs."""

            def __init__(self, cfg, *, parent):
                super().__init__(cfg, parent=parent)
                self.forward_called = False

            def forward(self, input_batch: Nested[Tensor]):
                if self.is_training:
                    # Check that input batch has the right shape and sharding.
                    jax.tree.map(check_shape, input_batch)
                    jax.tree.map(check_sharding, tree_paths(input_batch), input_batch)
                    self.forward_called = True
                return super().forward(input_batch)

        return DummyCheckingModel.default_config().set(dtype=jnp.float32)

    def _dummy_input_checking_input(self, global_logical_batch_size: int) -> DummyInput.Config:
        """Builds a dummy input that checks whether dispatch has been called."""

        class DummyCheckingDispatcher(SpmdInputDispatcher):
            """A dummy dispatcher that checks calls."""

            def __init__(self, cfg, *, parent):
                super().__init__(cfg, parent=parent)
                self.logical_to_physical = False
                self.physical_to_logical = False

            def logical_to_physical_batch(self, *args, **kwargs):
                self.logical_to_physical = True
                return super().logical_to_physical_batch(*args, **kwargs)

            def physical_to_logical_batch(self, *args, **kwargs):
                self.physical_to_logical = True
                return super().physical_to_logical_batch(*args, **kwargs)

        dispatch_cfg = DummyCheckingDispatcher.default_config().set(
            global_logical_batch_size=global_logical_batch_size,
        )
        return DummyInput.default_config().set(input_dispatcher=dispatch_cfg)

    def _test_input_dispatch(self, multiple: float, backend: Optional[str] = None):
        """Tests input dispatch under a few scenarios:
        1. global_logical_batch_size == process_count.
            In this case each process produces one example.
        2. global_logical_batch_size > process_count.
            In this case each process produces more than one example.
        3. global_logical_batch_size < process_count.
            In this case some processes are padding feeds.

        In all scenarios we should be able to entirely bypass logical dispatch by constructing
        per-feed logical batches and forming the global array directly, as long as the
        global_logical_batch_size divides batch_axis_names uniformly.
        """
        if backend is not None:
            utils_spmd.setup(jax_backend=backend)

        device_count = jax.device_count()
        process_count = jax.process_count()
        print(f"{device_count=}, {process_count=}")
        assert device_count > 1

        batch_axis_size = int(process_count * multiple)
        if batch_axis_size < 1:
            self.skipTest(f"Incompatible {process_count=} and {multiple=}")

        mesh_shape = (batch_axis_size, device_count // batch_axis_size)
        global_logical_batch_size = mesh_shape[0]
        batch_axis_names = ("data",)

        input_cfg = self._dummy_input_checking_input(global_logical_batch_size)
        cfg = self._trainer_config(input_cfg)
        cfg.batch_axis_names = batch_axis_names
        cfg.max_step = 3
        cfg.mesh_shape = mesh_shape
        cfg.model = self._dummy_input_checking_model(
            global_logical_batch_size, partition_spec=PartitionSpec(batch_axis_names)
        )
        trainer: SpmdTrainer = cfg.instantiate(parent=None)

        dispatcher = trainer.input.input_dispatcher

        # Check that dispatcher has right partition specs.
        self.assertEqual(dispatcher.config.partition_spec, PartitionSpec(batch_axis_names))

        # Validate that input partition spec is expected.
        with trainer.mesh():
            self.assertEqual(
                PartitionSpec(batch_axis_names),
                # pylint: disable-next=protected-access
                trainer._train_step_input_partition_specs(),
            )

        # Sanity check testing configs. We don't really use global_physical_batch_size without the
        # dispatch steps.
        # 1. Logical batch should divide batch axes.
        self.assertEqual(dispatcher.config.global_logical_batch_size % batch_axis_size, 0)
        # 2. Logical batch and num feeds should be what we requested above.
        self.assertEqual(dispatcher.config.global_logical_batch_size, process_count * multiple)
        # In the case of at least one-per-process (multiple >= 1), we have process_count feeds.
        # Otherwise, in the case of multiple < 1, a subset of processes act as logical feeds.
        self.assertEqual(dispatcher.num_logical_feeds, min(process_count, process_count * multiple))

        trainer.run(jax.random.PRNGKey(0))
        self.assertTrue(trainer.model.forward_called)
        self.assertTrue(trainer.input.input_dispatcher.logical_to_physical)
        self.assertTrue(trainer.input.input_dispatcher.physical_to_logical)

    @parameterized.parameters([1, 2])
    @pytest.mark.for_8_devices
    def test_input_dispatch_basic(self, multiple: float):
        """Tests input dispatch with at least 1 per process."""
        self._test_input_dispatch(multiple)

    @parameterized.parameters([1 / 2, 1 / 4])
    @pytest.mark.tpu
    def test_input_dispatch_every_other_process(self, multiple: float):
        """Tests input dispatch with some padding feeds. Requires process_count > 1."""
        self._test_input_dispatch(multiple, backend="tpu")

    def test_optional_batch_axes(self):
        """Tests that we can omit batch_axis_names."""
        mesh_shape = (jax.device_count(), 1)
        global_logical_batch_size = mesh_shape[0]
        partition_spec = PartitionSpec("model")  # Something other than "data".

        # Explicitly set a partition spec on input.
        input_cfg = self._dummy_input_checking_input(global_logical_batch_size)
        input_cfg.partition_spec = partition_spec

        cfg = self._trainer_config(input_cfg)
        cfg.batch_axis_names = None
        cfg.max_step = 3
        cfg.mesh_shape = mesh_shape
        cfg.model = self._dummy_input_checking_model(
            global_logical_batch_size, partition_spec=partition_spec
        )
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        self.assertEqual(partition_spec, trainer.input.partition_spec)


class SelectMeshConfigTest(test_utils.TestCase):
    def test_select_mesh_config(self):
        cfg = SpmdTrainer.default_config()
        self.assertIs(cfg.mesh_shape, REQUIRED)

        # When mesh_rules=None.
        self.assertIsNone(cfg.mesh_rules)
        select_mesh_config(cfg, mesh_selector="tpu-v4-128")
        # cfg.mesh_shape remains unchanged.
        self.assertIs(cfg.mesh_shape, REQUIRED)

        # When no mesh rule matches the selector.
        cfg.mesh_rules = (("tpu-v4-64", (4, 1, 8, 1)),)
        select_mesh_config(cfg, mesh_selector="tpu-v4-128")
        # cfg.mesh_shape still remains unchanged.
        self.assertIs(cfg.mesh_shape, REQUIRED)

        # When there is a match.
        select_mesh_config(cfg, mesh_selector="tpu-v4-64")
        # cfg.mesh_shape is overridden.
        self.assertEqual(cfg.mesh_shape, (4, 1, 8, 1))

        # When there is a match.
        cfg.mesh_rules = (
            ("gpu-(p5.48xlarge|p4de.24xlarge)-32", (4, 1, 8, 1)),
            ("gpu.*", None),
        )
        select_mesh_config(cfg, mesh_selector="gpu-p5.48xlarge-32")
        self.assertEqual(cfg.mesh_shape, (4, 1, 8, 1))
        select_mesh_config(cfg, mesh_selector="gpu-p4d.24xlarge-128")
        self.assertIsNone(cfg.mesh_shape)


class SelectExtendedMeshConfigTest(test_utils.TestCase):
    def test_select_mesh_config(self):
        cfg = SpmdTrainer.default_config().set(model=DummyModel.default_config())
        self.assertIs(cfg.mesh_shape, REQUIRED)

        # When mesh_rules=None.
        self.assertIsNone(cfg.mesh_rules)
        select_mesh_config(cfg, mesh_selector="tpu-v4-128")
        # cfg.mesh_shape remains unchanged.
        self.assertIs(cfg.mesh_shape, REQUIRED)

        # When no mesh rule matches the selector.
        cfg.mesh_rules = (
            (
                "tpu-v4-64",
                ChainConfigModifier.default_config().set(
                    config_modifiers=[
                        MeshShapeModifier.default_config().set(mesh_shape=(4, 1, 8, 1)),
                        RematSpecModifier.default_config().set(
                            remat_policies={
                                "model.linear": RematSpec(
                                    prevent_cse=True,
                                    policy=jax.ad_checkpoint.checkpoint_policies.dots_saveable,
                                ),
                            }
                        ),
                        GradientAccumulationModifier.default_config().set(grad_acc_steps=4),
                    ],
                ),
            ),
        )
        select_mesh_config(cfg, mesh_selector="tpu-v4-128")
        # cfg.mesh_shape still remains unchanged.
        self.assertIs(cfg.mesh_shape, REQUIRED)
        # When there is a match.
        select_mesh_config(cfg, mesh_selector="tpu-v4-64")
        # cfg.mesh_shape is overridden.
        self.assertEqual(cfg.mesh_shape, (4, 1, 8, 1))
        # Check if gradient accumulation is set up.
        self.assertRegex(str(cfg.learner.forward_fn_transformation), "steps: 4")
        # Check if remat policy is set up.
        self.assertRegex(str(cfg.model.linear), "dots_saveable")


class CompatibilityTest(test_utils.TestCase):
    def test_chex_serialization_compatibility(self):
        """Tests that a chex.dataclass that has been serialized as part of an AXLearn checkpoint
        can be read back in as a struct.PyTreeNode.
        """

        class Model(BaseModel):
            """Model that has struct params for testing."""

            @config_class
            class Config(BaseModel.Config):
                kind: Required[Literal["chex", "struct"]] = REQUIRED

            def initialize_parameters_recursively(
                self,
                prng_key: Tensor,
                *,
                prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None,
            ) -> NestedTensor:
                del prng_key
                del prebuilt
                cfg = self.config
                if cfg.kind == "chex":
                    param = flax_struct_test.Chex(
                        field_d=jnp.array(4),
                        field_b=jnp.array(1),
                        field_a=jnp.array(2),
                        field_c=jnp.array(3),
                    )
                elif cfg.kind == "struct":
                    param = flax_struct_test.Struct(
                        field_d=jnp.array(5),
                        field_b=jnp.array(6),
                        field_a=jnp.array(7),
                        field_c=jnp.array(8),
                    )
                else:
                    raise NotImplementedError
                return {"param": param}

            def create_parameter_specs_recursively(self) -> NestedParameterSpec:
                shape_dtype = jax.eval_shape(
                    lambda: self.initialize_parameters_recursively(jax.random.PRNGKey(0))
                )
                return jax.tree.map(
                    lambda x: ParameterSpec(shape=x.shape, dtype=x.dtype), shape_dtype
                )

        with tempfile.TemporaryDirectory() as tempdir:
            trainer_cfg = SpmdTrainer.default_config().set(
                name="tmp",
                dir=tempdir,
                model=Model.default_config().set(dtype=jnp.float32),
                input=DummyInput.default_config(),
                learner=learner.Learner.default_config().set(
                    optimizer=config_for_function(optimizers.sgd_optimizer).set(
                        learning_rate=0.1, decouple_weight_decay=True
                    )
                ),
                mesh_axis_names=("data",),
                mesh_shape=(1,),
            )
            trainer_cfg.checkpointer.save_policy = config_for_function(
                lambda: lambda *args, **kwargs: True
            )

            chex_trainer_cfg = trainer_cfg.clone()
            chex_trainer_cfg.model.kind = "chex"
            chex_trainer = chex_trainer_cfg.instantiate(parent=None)
            chex_trainer.init(prng_key=jax.random.PRNGKey(0))
            # pylint: disable-next=protected-access
            chex_trainer._step = 0
            chex_trainer.save_checkpoint({})
            chex_data = chex_trainer.trainer_state.model["param"]

            struct_trainer_cfg = trainer_cfg.clone()
            struct_trainer_cfg.model.kind = "struct"
            struct_trainer = struct_trainer_cfg.instantiate(parent=None)
            struct_trainer.init(prng_key=jax.random.PRNGKey(0))
            # Avoid unrelated errors about a prng key mismatch and input iterator mismatch.
            with unittest.mock.patch("axlearn.common.checkpointer.check_state_structure"):
                struct_trainer.restore_checkpoint()
            struct_data = struct_trainer.trainer_state.model["param"]

            self.assertIsInstance(chex_data, flax_struct_test.Chex)
            self.assertIsInstance(struct_data, flax_struct_test.Struct)
            chex.assert_trees_all_equal(
                dataclasses.asdict(chex_data), dataclasses.asdict(struct_data)
            )


class NanForwardModel(BaseModel):
    """A model that returns NaN in the forward pass."""

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        # Ensure we trigger a checkify error.
        return math.nan + jnp.asarray(0), {}


class NanInitModel(BaseModel):
    """A model that initializes its parameter to NaN."""

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        # Ensure we trigger a checkify error.
        return dict(
            a=math.nan + jnp.asarray(0),
        )

    def create_parameter_specs_recursively(self) -> NestedParameterSpec:
        return dict(a=ParameterSpec(shape=()))


class DebuggingTest(test_utils.TestCase):
    @parameterized.parameters(NanInitModel.default_config(), NanForwardModel.default_config())
    def test_context_manager(self, model: BaseModel.Config):
        """Tests the `context_manager` config field of `SpmdTrainer`."""
        with tempfile.TemporaryDirectory() as tempdir:
            trainer_cfg = SpmdTrainer.default_config().set(
                name="tmp",
                dir=tempdir,
                model=model.set(dtype=jnp.float32),
                input=DummyInput.default_config(),
                learner=learner.Learner.default_config().set(
                    optimizer=config_for_function(optimizers.sgd_optimizer).set(
                        learning_rate=0.1, decouple_weight_decay=True
                    )
                ),
                mesh_axis_names=("data",),
                mesh_shape=(1,),
                max_step=1,
                context_manager=config_for_function(
                    lambda target, new: lambda: unittest.mock.patch(target, new)
                ).set(
                    target="axlearn.common.trainer.pjit",
                    new=debug_utils.checkify_pjit(checkify.float_checks),
                ),
            )
            trainer = trainer_cfg.instantiate(parent=None)
            with self.assertRaisesRegex(
                checkify.JaxRuntimeError, "nan generated by primitive: add"
            ):
                trainer.run(prng_key=jax.random.PRNGKey(0))


if __name__ == "__main__":
    absltest.main()
