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
#     --module=axlearn.experiments.logistic_regression.tutorial \
#     --config=LogisticRegression \
#     --trainer_dir=/tmp/logistic_regression_test \
#     --data_dir=FAKE \
#     --jax_backend=cpu \
#     --status_port=7337
# tree /tmp/logistic_regression_test/

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl import app, flags

from axlearn.experiments.logistic_regression import ax

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


def synthetic_grain_source(
    num_examples: int = 1000,
    random_seed: int = 42,
    shuffle_and_repeat: bool = True,
    global_batch_size: int = 32,
):
    def dataset_fn(dispatch_config: ax.input_grain.DispatchConfig):
        features, labels = _synthesize_data(num_examples=num_examples, random_seed=random_seed)
        # Convert to list of examples for PyGrain
        examples = [{"features": features[i], "label": labels[i]} for i in range(len(features))]
        # Use fake_grain_source to create the dataset
        ds = ax.input_fake.fake_grain_source(
            examples,
            repeat=None if shuffle_and_repeat else 1,
            shuffle_seed=random_seed if shuffle_and_repeat else None,
        )
        # Add batching for the local batch size (divided by number of shards)
        local_batch_size = global_batch_size // dispatch_config.num_shards[0]
        ds = ds.batch(local_batch_size)
        # Convert to IterDataset as expected by the Input class
        return ax.input_grain.maybe_to_iter_dataset(ds)

    return dataset_fn


# ============================================================================
# Step 3: Model Definition
# ============================================================================


class LogisticRegressionModel(ax.BaseModel):
    """Logistic regression model."""

    @ax.config_class
    class Config(ax.BaseModel.Config):
        backbone: ax.config.Required[ax.config.InstantiableConfig] = ax.config.REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[ax.Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child("backbone", cfg.backbone)
        self._add_child(
            "metric", ax.layers.ClassificationMetric.default_config().set(num_classes=2)
        )

    def predict(self, input_batch: ax.Nested[ax.Tensor]) -> ax.Nested[ax.Tensor]:
        logits = self.backbone(input_batch["features"])
        return {"logits": logits}

    def forward(self, input_batch: ax.Nested[ax.Tensor]) -> tuple[ax.Tensor, ax.Nested[ax.Tensor]]:
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


def config_input(is_training: bool, global_batch_size: int = 32) -> ax.input_grain.Input.Config:
    def build_source():
        return synthetic_grain_source(
            num_examples=1000 if is_training else 200,
            random_seed=42 if is_training else 123,
            shuffle_and_repeat=is_training,
            global_batch_size=global_batch_size,
        )

    return ax.input_grain.Input.default_config().set(
        source=ax.config_for_function(build_source),
    )


# ============================================================================
# Step 5: Training Configuration
# ============================================================================


def config_logistic_regression_trainer() -> ax.SpmdTrainer.Config:
    model_cfg = LogisticRegressionModel.default_config().set(
        backbone=ax.layers.Linear.default_config().set(
            input_dim=2,
            output_dim=1,
            bias=True,
        ),
        dtype=jnp.float32,
    )
    train_input_cfg = config_input(is_training=True, global_batch_size=32)
    eval_input_cfg = config_input(is_training=False, global_batch_size=32)
    evaler_cfg = ax.evaler.SpmdEvaler.default_config().set(
        input=eval_input_cfg,
        eval_dtype=jnp.float32,
        eval_policy=ax.config_for_function(ax.evaler.every_n_steps_policy).set(n=100),
    )
    learning_rate_schedule = ax.config_for_function(ax.schedule.polynomial).set(
        begin_value=0.1,
        end_value=0.01,
        power=1.0,
        begin_step=0,
        end_step=1000,
    )
    optimizer_cfg = ax.config_for_function(ax.optimizers.sgd_optimizer).set(
        learning_rate=learning_rate_schedule,
        momentum=0.9,
        weight_decay=0.01,
        decouple_weight_decay=True,
    )
    learner_cfg = ax.learner.Learner.default_config().set(optimizer=optimizer_cfg)
    trainer_cfg = ax.SpmdTrainer.default_config().set(
        name="logistic_regression_trainer",
        mesh_axis_names=("data",),
        mesh_shape=(1,),
        model=model_cfg,
        input=train_input_cfg,
        learner=learner_cfg,
        evalers={"eval": evaler_cfg},
        max_step=1000,
    )
    trainer_cfg.checkpointer.save_policy = ax.config_for_function(
        ax.checkpointer.every_n_steps_policy
    ).set(n=200)
    trainer_cfg.summary_writer.write_every_n_steps = 10
    return trainer_cfg


# ============================================================================
# Step 6: The entrypoint for axlearn.common.launch_trainer_main.
# ============================================================================


def named_trainer_configs() -> dict[str, ax.config.TrainerConfigFn]:
    return {
        "LogisticRegression": config_logistic_regression_trainer,
    }


# ============================================================================
# Step 7: The entrypoint as a standalone script.
# ============================================================================


if __name__ == "__main__":
    flags.DEFINE_string("trainer_dir", None, "Output directory.", required=True)
    app.run(
        lambda _: config_logistic_regression_trainer()
        .set(dir=flags.FLAGS.trainer_dir)
        .instantiate(parent=None)
        .run(jax.random.PRNGKey(42))
    )
