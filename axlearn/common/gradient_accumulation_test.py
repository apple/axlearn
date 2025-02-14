# Copyright © 2024 Apple Inc.
"""Test module for gradient_accumulation.py"""
from typing import Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax.experimental.pjit import pjit

from axlearn.common import gradient_accumulation, test_utils
from axlearn.common.config import config_for_function
from axlearn.common.input_base import partition_by_path_rank
from axlearn.common.metrics import MetricAccumulator, WeightedScalar
from axlearn.common.module import new_output_collection
from axlearn.common.update_transformation import ForwardOutputs
from axlearn.common.utils import Nested, PartitionSpec, Tensor, tree_paths


class TestMinibatchPartitioner(test_utils.TestCase):
    """Test `with_minibatch_steps` decorator argument minibatch_partitioner."""

    def create_dummy_inputs(self, steps):
        # Multiply by accumulation steps
        self.batch_size = 4 * steps
        self.seq_len = 8
        self.params = dict(
            w=jnp.asarray([0.0, 2.0, 2.0, -3.0]),
            b=jnp.asarray([0.0, -1.0, 0.0, 0.0]),
        )

        self.input_batch = {
            "input_ids": jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((self.batch_size, self.seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((self.batch_size,), dtype=jnp.int32),
        }
        forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 2)
        self.inputs = dict(
            input_batch=self.input_batch,
            forward_key=forward_key,
            param_noise_key=param_noise_key,
        )

    def create_loss_fn(self, expected_minibatch_sharding):
        """Simple ForwardFn with a check for minibatch sharding."""

        def _check_equal_sharding(input_batch: Nested[Tensor], expected: dict):
            """Checks if sharding for input_batch matches expected."""

            def callback_sharding(
                *,
                input_batch: Nested[Tensor],
                callback: Callable[[str, jax.sharding.Sharding], None],
            ):
                """Invokes callback with the sharding.

                The callback is invoked with (path: str, sharding: Sharding).
                """

                def check_sharding(path, value):
                    jax.debug.inspect_array_sharding(
                        value, callback=lambda sharding: callback(path, sharding)
                    )

                jax.tree_map(check_sharding, tree_paths(input_batch), input_batch)
                return input_batch

            callback = lambda path, sharding: self.assertEqual(expected[path], sharding.spec)

            callback_sharding(
                input_batch=input_batch,
                callback=callback,
            )

        def loss_fn(*, model_params, inputs) -> ForwardOutputs:
            """Simple ForwardFn."""
            _check_equal_sharding(
                input_batch=inputs["input_batch"],
                expected=expected_minibatch_sharding,
            )
            loss = -jax.nn.log_softmax(model_params["w"] + model_params["b"])[1]
            output_collection = new_output_collection()
            output_collection.state_updates["w"] = model_params["w"] + 1
            output_collection.state_updates["loss"] = WeightedScalar(loss, 1)
            return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)

        return loss_fn

    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_minibatch_partitioner_default(self, steps):
        """Tests grad accumulation with minibatch steps and default minibatch partitioner."""

        # pylint: disable=too-many-function-args
        with jax.sharding.Mesh(
            devices=np.array(jax.devices()).reshape(1, 2, 1, 2)[..., None],
            axis_names=("expert", "data", "fsdp", "seq", "model"),
        ):
            self.create_dummy_inputs(steps)
            loss_fn = self.create_loss_fn(
                expected_minibatch_sharding={
                    "input_ids": PartitionSpec(("data"), "seq"),
                    "target_labels": PartitionSpec(("data"), "seq"),
                    "target_num_bytes": PartitionSpec(("data")),
                },
            )

            loss_fn = gradient_accumulation.with_minibatch_steps(
                steps=steps,
                metric_accumulator=MetricAccumulator.default_config(),
                minibatch_partitioner=None,
            )(loss_fn)

            pjit(loss_fn, in_shardings=None).lower(
                model_params=self.params, inputs=self.inputs
            ).compile()

    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_minibatch_partitioner_non_default(self, steps):
        """Tests grad accumulation with minibatch steps and a custom minibatch partitioner."""

        with jax.sharding.Mesh(
            devices=np.array(jax.devices()).reshape(2, 2)[..., None],
            axis_names=("data", "seq", "model"),
        ):
            self.create_dummy_inputs(steps)
            loss_fn = self.create_loss_fn(
                expected_minibatch_sharding={
                    "input_ids": PartitionSpec(("data", "seq")),
                    "target_labels": PartitionSpec(("data", "seq")),
                    "target_num_bytes": PartitionSpec(("data", "seq")),
                },
            )

            loss_fn = gradient_accumulation.with_minibatch_steps(
                steps=steps,
                metric_accumulator=MetricAccumulator.default_config(),
                minibatch_partitioner=config_for_function(partition_by_path_rank).set(
                    path_rank_to_partition={
                        # Shard batch dim on all available axis
                        (None, 1): PartitionSpec(("data", "seq")),
                        (None, 2): PartitionSpec(("data", "seq"), None),
                    }
                ),
            )(loss_fn)

            pjit(loss_fn, in_shardings=None).lower(
                model_params=self.params, inputs=self.inputs
            ).compile()


class TestMinibatchSteps(test_utils.TestCase):
    """Test `with_minibatch_steps` decorator."""

    @parameterized.named_parameters(
        ("one_step", 1),  # no accumulation
        ("two_steps", 2),
        ("four_steps", 4),
    )
    def test_minibatch_steps_grads_and_loss(self, steps):
        """Tests grad accumulation with minibatch steps."""
        params = dict(
            w=jnp.asarray([0.0, 2.0, 2.0, -3.0]),
            b=jnp.asarray([0.0, -1.0, 0.0, 0.0]),
        )

        def loss_fn(*, model_params, inputs) -> ForwardOutputs:
            """Simple ForwardFn."""
            del inputs
            loss = -jax.nn.log_softmax(model_params["w"] + model_params["b"])[1]
            output_collection = new_output_collection()
            output_collection.state_updates["w"] = model_params["w"] + 1
            output_collection.state_updates["loss"] = WeightedScalar(loss, 1)
            return ForwardOutputs(loss=loss, aux={}, output_collection=output_collection)

        batch_key, forward_key, param_noise_key = jax.random.split(jax.random.PRNGKey(0), 3)
        inputs = dict(
            input_batch=jax.random.randint(batch_key, (32, 4096), 1, 100),
            forward_key=forward_key,
            param_noise_key=param_noise_key,
        )
        # Compute grads and loss without the minibatch decorator.
        loss_expected, grads_expected = jax.value_and_grad(
            lambda x: loss_fn(model_params=x, inputs=inputs).loss
        )(params)
        # Compute grads and loss with the minibatch decorator.
        loss_fn = gradient_accumulation.with_minibatch_steps(
            steps=steps, metric_accumulator=MetricAccumulator.default_config()
        )(loss_fn)
        loss_minibatch, grads_minibatch = jax.value_and_grad(
            lambda x: loss_fn(model_params=x, inputs=inputs).loss
        )(params)

        chex.assert_trees_all_close(
            (loss_expected, grads_expected),
            (loss_minibatch, grads_minibatch),
        )


if __name__ == "__main__":
    absltest.main()
