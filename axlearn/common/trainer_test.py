# Copyright © 2023 Apple Inc.

"""Tests SpmdTrainer."""
# pylint: disable=no-self-use
import copy
import dataclasses
import os.path
import shutil
import tempfile
from typing import Any, Callable, Dict, Optional, Sequence

import jax
import jax.random
import numpy as np
import tensorflow as tf
from absl import flags, logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import layers, learner, optimizers, param_init, test_utils, utils
from axlearn.common.base_layer import NestedParameterSpec
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
from axlearn.common.learner import UpdateType, should_update_with_optimizers
from axlearn.common.module import Module
from axlearn.common.state_builder import Builder as TrainerStateBuilder
from axlearn.common.trainer import SpmdTrainer, _create_device_mesh, _prune_empty, _TrainerState
from axlearn.common.utils import NestedTensor, Tensor, as_tensor, flatten_items, match_regex_rules

FLAGS = flags.FLAGS

NUM_CLASSES = 16


class DummyInput(Module):
    """A dummy input."""

    @config_class
    class Config(Module.Config):
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
                    shape=[cfg.batch_size, 224, 224, 3],
                    minval=0,
                    maxval=256,
                    dtype=np.int32,
                ).astype(np.float32),
            )
            if cfg.include_labels:
                batch["label"] = jax.random.randint(
                    label_key,
                    shape=[cfg.batch_size],
                    minval=0,
                    maxval=NUM_CLASSES,
                    dtype=np.int32,
                )
            yield batch

    def dataset(self) -> tf.data.Dataset:
        cfg = self.config
        features = {
            "image": tf.io.FixedLenFeature(shape=(cfg.batch_size, 224, 224, 3), dtype=tf.float32),
        }
        if cfg.include_labels:
            features["label"] = tf.io.FixedLenFeature(shape=(cfg.batch_size,), dtype=tf.int64)
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
        for input_batch in self.dataset():
            yield input_batch


class DummyModel(BaseModel):
    """A dummy model."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyModel."""

        # Whether to explicitly init dummy state to test pruning backwards compat.
        init_dummy_state: bool = False

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "fc",
            layers.Linear.default_config().set(
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
        self, prng_key: jax.random.KeyArray, *, prebuilt: Optional[NestedTensor]
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
        built_keys = set(
            f"model/{key}" for key, _ in flatten_items(built_state.trainer_state.model)
        )
        logging.info("built_keys=%s", built_keys)
        return TrainerStateBuilder.State(
            step=self.config.step,
            trainer_state=_TrainerState(
                prng_key=built_state.trainer_state.prng_key,
                model=self.config.model_state(),
                learner=built_state.trainer_state.learner,
            ),
            built_keys=built_keys,
        )


class TrainerTest(test_utils.TestCase):
    def test_prune_empty_state(self):
        state = {
            "state": {
                "tensor": jnp.array(0),
                "nested": {
                    "empty": {},
                    "not_empty": jnp.array([]),
                },
            },
            "removed": {
                "nested": {
                    "deep_nested": {},
                },
                "sibling": {
                    "deep_nested": {},
                },
            },
        }
        expected = {
            "state": {
                "tensor": jnp.array(0),
                "nested": {
                    "not_empty": jnp.array([]),
                },
            },
        }
        actual = _prune_empty(state)
        self.assertNestedAllClose(expected, actual)

    def test_prune_backwards_compat(self):
        """Test that pruning is backwards compatible with no pruning."""
        # Construct a base trainer config.
        cfg = SpmdTrainer.default_config().set(
            name="base_trainer",
            vlog=3,
            dir=tempfile.mkdtemp(),
            mesh_axis_names=("data", "model"),
            mesh_shape=(1, 1),
            model=DummyModel.default_config().set(
                dtype=jnp.float32, param_init=param_init.DefaultInitializer.default_config()
            ),
            input=DummyInput.default_config(),
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
                    input=DummyInput.default_config().set(total_num_batches=2),
                ),
            ),
        )
        # Instantiate without pruning. We need to explicitly init dummy state in this case for
        # trainer to run.
        base_cfg = cfg.set(prune_empty_state_updates=False)
        base_cfg.model.set(init_dummy_state=True)
        base_trainer: SpmdTrainer = base_cfg.instantiate(parent=None)

        # Run until first checkpoint.
        base_trainer.run(prng_key=jax.random.PRNGKey(123))
        base_state = base_trainer.trainer_state

        # Make sure checkpoint exists.
        assert os.path.exists(os.path.join(cfg.dir, "trainer_state_tree.txt"))
        trainer2: SpmdTrainer = cfg.instantiate(parent=None)
        with trainer2.mesh():
            self.assertEqual(5, trainer2.restore_checkpoint())

        # Instantiate with pruning and more steps.
        pruned_cfg = cfg.set(
            name="pruned_trainer",
            max_step=12,
            prune_empty_state_updates=True,
        )
        pruned_cfg.model.set(init_dummy_state=False)
        pruned_trainer = pruned_cfg.instantiate(parent=None)

        # Load initial checkpoint and run until next checkpoint.
        pruned_trainer.run(prng_key=jax.random.PRNGKey(123))
        pruned_state = pruned_trainer.trainer_state

        # Model states should have same shapes only after pruning.
        self.assertNotEqual(
            utils.shapes(base_state.model),
            utils.shapes(pruned_state.model),
        )
        self.assertEqual(
            utils.shapes(_prune_empty(base_state.model)),
            utils.shapes(pruned_state.model),
        )

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
            self.assertStartsWith(f.read(), "PyTreeDef(CustomNode(namedtuple[_TrainerState], [*, ")

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
        trainer.init(jax.random.PRNGKey(0))
        # Sanity check that the initial model state is not already equal.
        with self.assertRaises(AssertionError):
            # pylint: disable-next=protected-access
            self.assertNestedAllClose(trainer._trainer_state.model, new_model_state)

        # Restore from builder, and check model state and step are now updated.
        cfg.set(init_state_builder=state_builder)
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
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
    )
    def test_checkpoint_policy(self, *, save_input_iterator: bool, restore_input_iterator: bool):
        """Test checkpoint policy when evaler and checkpointer run at different cadences."""
        model_cfg = DummyModel.default_config().set(dtype=jnp.float32)

        def checkpoint_if_all_evalers_run(evaler_names: Sequence[str]) -> CheckpointPolicy:
            def fn(*, step: int, evaler_summaries: Dict[str, Any]):
                del step
                for evaler_name in evaler_names:
                    if evaler_summaries.get(evaler_name) is None:
                        return False
                return True

            return fn

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

        # Run trainer.
        trainer: SpmdTrainer = cfg.instantiate(parent=None)
        trainer.run(prng_key=jax.random.PRNGKey(123))

        assert os.path.exists(os.path.join(cfg.dir, "trainer_state_tree.txt"))
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


@dataclasses.dataclass(frozen=True)
class DummyDevice:
    """Mock device for testing."""

    platform: str
    device_kind: str
    process_index: int


@dataclasses.dataclass(frozen=True)
class DummyTpuDevice(DummyDevice):
    """Mock TPU device for testing."""

    coords: Sequence[int]
    core_on_chip: int = 0


@dataclasses.dataclass(frozen=True)
class DummyMultiSliceTpuDevice(DummyTpuDevice):
    """Mock multi-slice TPU device for testing."""

    slice_index: int = 0


class DeviceMeshTest(test_utils.TestCase):
    @parameterized.parameters(
        {"logical_mesh": (2, 8)},
        {"logical_mesh": (4, 4)},
        {"logical_mesh": (1, 2, 8)},
    )
    def test_create_device_mesh_tpuv4(self, logical_mesh: Sequence[int]):
        physical_mesh = (4, 4, 1)
        coords = [
            (x, y, z)
            for x in range(physical_mesh[0])
            for y in range(physical_mesh[1])
            for z in range(physical_mesh[2])
        ]
        devices = [
            DummyTpuDevice(
                platform="tpu",
                device_kind="TPU v4",
                process_index=ix // 4,
                coords=coord,
            )
            for ix, coord in enumerate(coords)
        ]
        # Check that the constructed mesh has the expected shape.
        self.assertEqual(
            _create_device_mesh(mesh_shape=logical_mesh, devices=devices).shape, logical_mesh
        )

    @parameterized.parameters(
        {"logical_mesh": (2, 16)},
        {"logical_mesh": (2, 4, 4)},
    )
    def test_create_device_mesh_multi_slice_tpuv4(self, logical_mesh: Sequence[int]):
        slice_physical_mesh = (4, 4, 1)
        num_slices = 2
        coords = [
            (x, y, z)
            for x in range(slice_physical_mesh[0])
            for y in range(slice_physical_mesh[1])
            for z in range(slice_physical_mesh[2])
        ]
        devices = [
            DummyMultiSliceTpuDevice(
                platform="tpu",
                device_kind="TPU v4",
                process_index=(len(coords) * slice_index + ix) // 4,
                coords=coord,
                slice_index=slice_index,
            )
            for ix, coord in enumerate(coords)
            for slice_index in range(num_slices)
        ]
        # Check that the constructed mesh has the expected shape.
        device_mesh = _create_device_mesh(mesh_shape=logical_mesh, devices=devices)
        self.assertEqual(device_mesh.shape, logical_mesh)
        # Check that the sub_mesh along the first axis only contains devices from one of the slices.
        for ix, sub_mesh in enumerate(device_mesh):
            self.assertTrue(all(el.slice_index == ix for el in sub_mesh.flatten()))

    @parameterized.parameters(
        {"logical_mesh": (8, 2, 4)},
        {"logical_mesh": (16, 4)},
        {"logical_mesh": (2, 32)},
    )
    def test_create_device_mesh_gpu(self, logical_mesh: Sequence[int] = (8, 2, 4)):
        num_gpus_per_process = 8
        num_granules = 8
        devices = [
            DummyDevice(
                platform="gpu",
                device_kind="gpu",
                process_index=(num_gpus_per_process * granule_index + ix) // num_gpus_per_process,
            )
            for ix in range(num_gpus_per_process)
            for granule_index in range(num_granules)
        ]
        # Check that the constructed mesh has the expected shape.
        device_mesh = _create_device_mesh(mesh_shape=logical_mesh, devices=devices)
        self.assertEqual(device_mesh.shape, logical_mesh)


if __name__ == "__main__":
    absltest.main()
