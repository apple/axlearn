# Copyright © 2025 Apple Inc.

"""Simple logistic regression tutorial using AxLearn.

This tutorial demonstrates how to train a basic logistic regression model
on synthetic 2D data for binary classification.
"""

# ============================================================================
# Step 1: Create the Tutorial File
# ============================================================================
#
# Install AXLearn following the documentation before training logistic regression model locally
# using CPU and synthetic data.
#
# mkdir -p /tmp/logistic_regression_test
# rm -rf /tmp/logistic_regression_test/*
# python -m axlearn.common.launch_trainer_main \
#     --module=axlearn.experiments.logistic_regression_tutorial \
#     --config=LogisticRegression \
#     --trainer_dir=/tmp/logistic_regression_test \
#     --data_dir=FAKE \
#     --jax_backend=cpu \
#     --status_port=7337
# tree /tmp/logistic_regression_test/

from typing import Optional

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from axlearn.common import config, evaler, input_tf_data, learner, optimizers, schedule, trainer
from axlearn.common.base_model import BaseModel
from axlearn.common.checkpointer import every_n_steps_policy
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps_policy
from axlearn.common.input_tf_data import BuildDatasetFn, Input
from axlearn.common.layers import ClassificationMetric, Linear
from axlearn.common.module import Module
from axlearn.common.utils import NestedTensor, Tensor
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

# ============================================================================
# Step 2: Data Generation
# ============================================================================


def _synthesize_data(
    num_examples: int = 1000, num_features: int = 2, noise_std: float = 0.1, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_seed)
    cluster_0 = rng.normal(loc=[-1.0, -1.0], scale=0.5, size=(num_examples // 2, num_features))
    cluster_1 = rng.normal(loc=[1.0, 1.0], scale=0.5, size=(num_examples // 2, num_features))
    features = np.vstack([cluster_0, cluster_1])
    features += rng.normal(0, noise_std, features.shape)
    labels = np.hstack(
        [np.zeros(num_examples // 2, dtype=np.int32), np.ones(num_examples // 2, dtype=np.int32)]
    )
    return features.astype(np.float32), labels


def synthetic_dataset_fn(
    num_examples: int = 1000, random_seed: int = 42, shuffle_and_repeat: bool = True
) -> BuildDatasetFn:
    def dataset_fn() -> tf.data.Dataset:
        features, labels = _synthesize_data(num_examples=num_examples, random_seed=random_seed)

        def data_generator():
            for i in range(len(features)):
                yield {"features": features[i], "label": labels[i]}

        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature={
                "features": tf.TensorSpec(shape=(2,), dtype=tf.float32),
                "label": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        )

        if shuffle_and_repeat:
            dataset = dataset.shuffle(buffer_size=num_examples)
            dataset = dataset.repeat()

        return dataset

    return dataset_fn


# ============================================================================
# Step 3: Model Definition
# ============================================================================


class LogisticRegressionModel(BaseModel):
    """Logistic regression model."""

    @config_class
    class Config(BaseModel.Config):
        backbone: Required[InstantiableConfig] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child("backbone", cfg.backbone)
        self._add_child("metric", ClassificationMetric.default_config().set(num_classes=2))

    def predict(self, input_batch: NestedTensor) -> NestedTensor:
        logits = self.backbone(input_batch["features"])
        return {"logits": logits}

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        outputs = self.predict(input_batch)
        logits = outputs["logits"]

        binary_logits = jnp.concatenate([jnp.zeros_like(logits), logits], axis=-1)

        labels = input_batch["label"]
        loss = self.metric(binary_logits, labels=labels)

        outputs["binary_logits"] = binary_logits
        outputs["predictions"] = jnp.argmax(binary_logits, axis=-1)

        return loss, outputs


# ============================================================================
# Step 4: Input Pipeline Configuration
# ============================================================================


def build_input_config(is_training: bool, global_batch_size: int = 32) -> Input.Config:
    dataset_fn = (
        synthetic_dataset_fn(num_examples=1000, random_seed=42, shuffle_and_repeat=True)
        if is_training
        else synthetic_dataset_fn(num_examples=200, random_seed=123, shuffle_and_repeat=False)
    )

    return Input.default_config().set(
        is_training=is_training,
        source=config.config_for_function(lambda: dataset_fn).set(),
        processor=config.config_for_function(input_tf_data.identity).set(),
        batcher=config.config_for_function(input_tf_data.batch).set(
            global_batch_size=global_batch_size,
            prefetch_buffer_size=tf.data.AUTOTUNE,
            pad_example_fn=input_tf_data.default_pad_example_fn,
        ),
    )


# ============================================================================
# Step 5: Training Configuration
# ============================================================================


def logistic_regression_trainer() -> trainer.SpmdTrainer.Config:
    model_cfg = LogisticRegressionModel.default_config().set(
        backbone=Linear.default_config().set(
            input_dim=2,
            output_dim=1,
            bias=True,
        ),
        dtype=jnp.float32,
    )

    train_input_cfg = build_input_config(is_training=True, global_batch_size=32)

    eval_input_cfg = build_input_config(is_training=False, global_batch_size=32)

    evaler_cfg = evaler.SpmdEvaler.default_config().set(
        input=eval_input_cfg,
        eval_dtype=jnp.float32,
        eval_policy=config.config_for_function(eval_every_n_steps_policy).set(n=100),
    )

    learning_rate_schedule = config.config_for_function(schedule.polynomial).set(
        begin_value=0.1,
        end_value=0.01,
        power=1.0,
        begin_step=0,
        end_step=1000,
    )

    optimizer_cfg = config.config_for_function(optimizers.sgd_optimizer).set(
        learning_rate=learning_rate_schedule,
        momentum=0.9,
        weight_decay=0.01,
        decouple_weight_decay=True,
    )

    learner_cfg = learner.Learner.default_config().set(optimizer=optimizer_cfg)

    trainer_cfg = trainer.SpmdTrainer.default_config().set(
        name="logistic_regression_trainer",
        model=model_cfg,
        input=train_input_cfg,
        learner=learner_cfg,
        evalers={"eval": evaler_cfg},
        max_step=1000,
    )

    trainer_cfg.checkpointer.save_policy = config.config_for_function(every_n_steps_policy).set(
        n=200
    )

    trainer_cfg.summary_writer.write_every_n_steps = 10

    return trainer_cfg


# ============================================================================
# Step 6: Expose Configuration to AxLearn CLI
# ============================================================================


def named_trainer_configs() -> dict[str, TrainerConfigFn]:
    return {
        "LogisticRegression": logistic_regression_trainer,
    }
