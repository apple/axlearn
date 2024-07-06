# Copyright © 2024 Apple Inc.
"""Test module for gradient_accumulation.py"""
import chex
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.common import gradient_accumulation, test_utils
from axlearn.common.metrics import MetricAccumulator, WeightedScalar
from axlearn.common.module import new_output_collection
from axlearn.common.update_transformation import ForwardOutputs


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
