# Copyright © 2024 Apple Inc.
"""Tests for update_transformation.py."""
import dataclasses
from collections.abc import Sequence

import chex
import jax
import optax
from absl.testing import absltest, parameterized
from jax import numpy as jnp

import axlearn.common
from axlearn.common import optimizers, test_utils
from axlearn.common.base_layer import FactorizationSpec, ParameterSpec
from axlearn.common.config import config_for_function, maybe_instantiate
from axlearn.common.module import (
    InvocationContext,
    functional,
    new_output_collection,
    set_current_context,
)
from axlearn.common.optimizer_base import OptParam, PartitionedGradientTransformation
from axlearn.common.update_transformation import (
    Updates,
    UpdateTransformation,
    WrappedPartitionedGradientTransformation,
)
from axlearn.common.utils import Nested, Tensor, VDict


class UpdateTransformationTest(test_utils.TestCase):
    """Tests related to the `UpdateTransformation` API."""

    @parameterized.parameters(
        dict(
            optimizer=optimizers.sgd_optimizer(
                learning_rate=0.5,
                decouple_weight_decay=True,
            ),
            use_config=True,
        ),
        dict(
            optimizer=optimizers.sgd_optimizer(
                learning_rate=0.5,
                decouple_weight_decay=True,
            ),
            use_config=False,
        ),
        dict(
            optimizer=optimizers.adamw_optimizer(
                learning_rate=0.9,
                b1=0.9,
                b2=0.99,
                eps=1e-8,
                weight_decay=0.4,
            )
        ),
        dict(
            optimizer=optimizers.adamw_decoupled_optimizer(
                learning_rate=0.85,
                b1=0.95,
                b2=0.99,
                eps=1e-8,
                update_schedule=1.0,
                weight_decay=0.3,
            )
        ),
        dict(
            optimizer=optimizers.adastar_optimizer(
                learning_rate=0.1,
                gradient_ema_decay=0.8,
                gradient_ema_debias=False,
                gradient_square_ema_decay=0.75,
                gradient_square_ema_debias=True,
                eps=1e-9,
                eps_square=0,
                update_ema_decay=None,
                raw_update_clipping_threshold=0.01,
                update_ema_debias=False,
                update_schedule=1.0,
            )
        ),
        dict(
            optimizer=optimizers.chain(
                optimizers.sgd_optimizer(
                    learning_rate=0.5,
                    decouple_weight_decay=True,
                ),
                optimizers.sgd_optimizer(
                    learning_rate=0.5,
                    decouple_weight_decay=True,
                ),
            )
        ),
    )
    def test_wrapped_partitioned_gradient_transformation(
        self, *, use_config: bool = False, optimizer: PartitionedGradientTransformation
    ):
        if use_config:
            optimizer = config_for_function(lambda t=optimizer: t)
        cfg = WrappedPartitionedGradientTransformation.default_config().set(
            name="tmp", transformation=optimizer
        )
        update_transformation: UpdateTransformation = cfg.instantiate(parent=None)
        optimizer: PartitionedGradientTransformation = maybe_instantiate(optimizer)

        # Setup state and call args.
        updates = mock_updates()
        opt_params = updates.opt_params
        grads = updates.delta_updates
        param_specs = updates.param_specs()
        inplace_updates = updates.inplace_updates

        # For backwards compatibility, we make sure the state tree structure is the same.
        actual_init = update_transformation.init(opt_params)
        expected_init = optimizer.init(opt_params)
        self.assertNestedAllClose(actual_init, expected_init)

        actual_specs = update_transformation.create_state_partition_specs(param_specs)
        expected_specs = optimizer.partition(param_specs)
        self.assertNestedAllClose(actual_specs, expected_specs)

        # Check that the final update and optimizer state updates are the same.
        actual_update, actual_output_collection = functional(
            module=update_transformation,
            prng_key=None,
            state=actual_init,
            inputs=[updates],
            method="__call__",
            is_training=True,
        )
        ctx = InvocationContext(
            name="root",
            parent=None,
            module=None,
            state={},
            is_training=True,
            prng_key=None,
            output_collection=new_output_collection(),
        )
        with set_current_context(ctx):
            (
                expected_param_update,
                expected_state_update,
            ) = optimizer.update(updates=grads, state=expected_init, params=opt_params)
        expected_update = Updates(
            opt_params=opt_params,
            delta_updates=expected_param_update,
            inplace_updates=inplace_updates,
        )
        self.assertNestedAllClose(actual_update, expected_update)
        self.assertNestedAllClose(actual_output_collection.state_updates, expected_state_update)

        # Check that the summaries are the same.
        self.assertNestedAllClose(actual_output_collection.summaries, ctx.get_summaries())


def mock_params() -> Nested[Tensor]:
    """Returns mock model params."""
    return dict(
        weight=jnp.ones((3, 5)),  # An ordinary param.
        vdict=VDict(bias=jnp.ones((7, 5))),  # Param in a custom pytreee node.
        state=jnp.ones(2, dtype=jnp.int32),  # Simulates a nograd param.
        more_state=jnp.ones(3, dtype=jnp.int32),  # A param with both grad and state update.
        do_not_update=optax.MaskedNode(),  # A param masked out by CompositeLearner.
    )


def mock_updates() -> axlearn.common.update_transformation.Updates:
    """Create an updates object with various semi-reasonable values."""
    model_params = mock_params()
    opt_params = jax.tree.map(
        lambda p: OptParam(
            value=p,
            factorization_spec=FactorizationSpec([None] * p.ndim),
            weight_decay_scale=0.1,
        ),
        model_params,
    )
    delta_updates = jax.tree.map(lambda p: p * -0.1, model_params)
    delta_updates["state"] = None
    inplace_updates = dict(
        state=jnp.arange(2, dtype=jnp.int32),
        more_state=jnp.arange(3, dtype=jnp.int32),
        do_not_update=optax.MaskedNode(),
    )
    updates = axlearn.common.update_transformation.Updates(
        opt_params=opt_params, delta_updates=delta_updates, inplace_updates=inplace_updates
    )
    return updates


class UpdatesTest(test_utils.TestCase):
    """Tests related to the `Updates` interface."""

    def test_param_values(self):
        updates = mock_updates()
        actual = updates.param_values()
        expected = mock_params()
        chex.assert_trees_all_equal_structs(actual, expected)
        self.assertNestedAllClose(actual, expected)

    def test_param_specs(self):
        updates = mock_updates()
        actual = updates.param_specs()
        expected = dict(
            weight=ParameterSpec(
                shape=(3, 5),
                dtype=jnp.float32,
                factorization=FactorizationSpec([None, None]),
                weight_decay_scale=0.1,
            ),
            vdict=VDict(
                bias=ParameterSpec(
                    shape=(7, 5),
                    dtype=jnp.float32,
                    factorization=FactorizationSpec([None, None]),
                    weight_decay_scale=0.1,
                )
            ),
            state=ParameterSpec(
                shape=(2,),
                dtype=jnp.int32,
                factorization=FactorizationSpec([None]),
                weight_decay_scale=0.1,
            ),
            more_state=ParameterSpec(
                shape=(3,),
                dtype=jnp.int32,
                factorization=FactorizationSpec([None]),
                weight_decay_scale=0.1,
            ),
            do_not_update=optax.MaskedNode(),
        )
        chex.assert_trees_all_equal_structs(actual, expected)
        self.assertEqual(actual, expected)

    @parameterized.product(
        fields=[[], ["opt_params"], ["delta_updates"], ["inplace_updates"], None],
        masked_names=[["weight", "more_state"], ["weight"]],
    )
    def test_mask(self, fields: Sequence[str], masked_names):
        updates = mock_updates()
        kwargs = dict(keep=lambda tree: {k: k in masked_names for k in tree})
        if fields is not None:
            kwargs.update(fields=fields)
        else:
            fields = ("opt_params", "delta_updates", "inplace_updates")
        actual = updates.mask(**kwargs)

        def expected_result(tree: Nested) -> Nested:
            return tree | {k: optax.MaskedNode() for k in tree if k not in masked_names}

        opt_params = expected_result(updates.opt_params)
        delta_updates = expected_result(updates.delta_updates)
        inplace_updates = expected_result(updates.inplace_updates)

        # Expected value of `masked` with default fields.
        expected_default = axlearn.common.update_transformation.Updates(
            opt_params=opt_params, delta_updates=delta_updates, inplace_updates=inplace_updates
        )
        expected = dataclasses.replace(
            updates, **{field: getattr(expected_default, field) for field in fields}
        )

        chex.assert_trees_all_equal_structs(actual, expected)
        self.assertNestedAllClose(actual, expected)


if __name__ == "__main__":
    absltest.main()
