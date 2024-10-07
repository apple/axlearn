# Copyright © 2023 Apple Inc.

"""Tests factorized RMS."""
from typing import Optional

import jax.nn
import optax
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp

from axlearn.common import factorized_rms
from axlearn.common.base_layer import FactorizationSpec, ParameterSpec
from axlearn.common.optimizer_base import (
    NestedOptStateSpec,
    OptParam,
    PartitionedGradientTransformation,
)
from axlearn.common.optimizers import OptStateSpec, with_partition_fn
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import PartitionSpec, flatten_items


class FactorizedRMSTest(TestCase):
    @parameterized.product(
        factored=(False, True),
        dtype=(jnp.float32, jnp.bfloat16),
    )
    def testParity(self, factored, dtype):
        ref: PartitionedGradientTransformation = with_partition_fn(
            optax.scale_by_factored_rms(factored=factored),
            partition_fn=lambda _: None,
        )
        exp: PartitionedGradientTransformation = factorized_rms.scale_by_factored_rms(
            factored=factored
        )

        # Factorize 'w' but not 'b'.
        # By convention the largest dim is the "row", and the second largest is the "col".
        w_factorization = FactorizationSpec(axes=(None, "col", "row"))
        b_factorization = FactorizationSpec(axes=(None, None))

        # Partition on the 'expert' and 'model' axes.
        num_experts, model_dim, hidden_dim = 16, 150, 512
        param_specs = dict(
            w=ParameterSpec(
                dtype=dtype,
                shape=[num_experts, model_dim, hidden_dim],
                mesh_axes=PartitionSpec("expert", None, "model"),
                factorization=w_factorization,
            ),
            b=ParameterSpec(
                dtype=dtype,
                shape=[num_experts, hidden_dim],
                mesh_axes=PartitionSpec("expert", "model"),
                factorization=b_factorization,
            ),
        )

        # The 'exp' optimizer is partitioned according to the mesh_axes of parameters and
        # factorization spec.
        exp_partition: NestedOptStateSpec = exp.partition(param_specs)
        # Used for `count`.
        count_spec = OptStateSpec(
            dtype=jnp.int32,
            shape=[],
            mesh_axes=PartitionSpec(),
        )
        # Used for disabled states.
        dummy_spec = OptStateSpec(
            dtype=jnp.float32,
            shape=(1,),
            mesh_axes=PartitionSpec(
                None,
            ),
        )
        if factored:
            self.assertSequenceEqual(
                optax.FactoredState(
                    count=count_spec,
                    v_row=dict(
                        b=dummy_spec,
                        # 'v_row' does not have the 'row' dimension.
                        w=OptStateSpec(
                            dtype=dtype,
                            shape=[num_experts, model_dim],
                            mesh_axes=PartitionSpec("expert", None),
                        ),
                    ),
                    v_col=dict(
                        b=dummy_spec,
                        # 'v_col' does not have the 'col' dimension.
                        w=OptStateSpec(
                            dtype=dtype,
                            shape=[num_experts, hidden_dim],
                            mesh_axes=PartitionSpec("expert", "model"),
                        ),
                    ),
                    v=dict(
                        b=OptStateSpec(
                            dtype=dtype,
                            shape=[num_experts, hidden_dim],
                            mesh_axes=PartitionSpec("expert", "model"),
                        ),
                        w=dummy_spec,
                    ),
                ),
                exp_partition,
            )
        else:
            self.assertSequenceEqual(
                optax.FactoredState(
                    count=count_spec,
                    v_row=dict(w=dummy_spec, b=dummy_spec),
                    v_col=dict(w=dummy_spec, b=dummy_spec),
                    v=jax.tree.map(
                        lambda param_spec: OptStateSpec(
                            dtype=dtype, shape=param_spec.shape, mesh_axes=param_spec.mesh_axes
                        ),
                        param_specs,
                    ),
                ),
                exp_partition,
            )

        # init() behaves the same between ref and exp.
        opt_params = dict(
            w=OptParam(
                value=jax.random.normal(
                    jax.random.PRNGKey(1), [num_experts, model_dim, hidden_dim]
                ),
                factorization_spec=w_factorization,
                weight_decay_scale=1.0,
            ),
            b=OptParam(
                value=jnp.zeros([num_experts, hidden_dim]),
                factorization_spec=b_factorization,
                weight_decay_scale=1.0,
            ),
        )
        ref_opt_state = ref.init(opt_params)
        exp_opt_state = exp.init(opt_params)
        self.assertNestedAllClose(ref_opt_state, exp_opt_state)

        # Check exp_partition against exp_opt_state.
        state_spec_map = dict(flatten_items(exp_partition))
        for path, value in flatten_items(exp_opt_state):
            state_spec: Optional[OptStateSpec] = state_spec_map.get(path)
            logging.info(
                "State: %s=%s(%s) state_spec=%s", path, value.dtype, value.shape, state_spec
            )
            if state_spec is None:
                self.assertEqual(value.size, 1, msg=f"{path}: {value.shape}")
                continue
            self.assertIsNotNone(state_spec, msg=f"{path}: {value.shape}")
            self.assertLen(state_spec.mesh_axes, len(value.shape))
            self.assertSequenceEqual(value.shape, state_spec.shape)
            for dim_size, dim_partition in zip(value.shape, state_spec.mesh_axes):
                if dim_partition is not None:
                    self.assertEqual(
                        dim_size % 8,
                        0,
                        msg=f"{path}: {dim_size} "
                        f"for {dim_partition} in {value.shape} vs. {state_spec}",
                    )

        # update() behaves the same between ref and exp.
        for step in range(10):
            updates = jax.tree.map(
                lambda x, seed=100 + step: jax.random.normal(jax.random.PRNGKey(seed), x.shape),
                opt_params,
            )
            ref_scaled_updates, ref_opt_state = ref.update(updates, ref_opt_state, opt_params)
            exp_scaled_updates, exp_opt_state = exp.update(updates, exp_opt_state, opt_params)
            self.assertNestedAllClose(ref_opt_state, exp_opt_state)
            self.assertNestedAllClose(ref_scaled_updates, exp_scaled_updates)


if __name__ == "__main__":
    absltest.main()
