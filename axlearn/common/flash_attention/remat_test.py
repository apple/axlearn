# Copyright © 2025 Apple Inc.

"""Tests FlashAttention remat policy."""
# pylint: disable=ungrouped-imports
import os

# Due to reference layer using XLA,
# set the following environment variables to avoid OOM in GPU tests.
# pylint: disable=wrong-import-position
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# pylint: enable=wrong-import-position
import jax
import jax.numpy as jnp
from absl.testing import parameterized
from jax.ad_checkpoint import checkpoint_policies
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.config import config_for_function
from axlearn.common.flash_attention.layer import (
    FlashAttention,
    default_mha_dim_to_partition_spec,
    default_output_dim_to_partition_spec,
)
from axlearn.common.flash_attention.layer_test import DummyModel, _fake_inputs
from axlearn.common.flash_attention.remat import save_or_offload_flash_attention_policy
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape
from axlearn.common.utils import (
    Offloadable,
    Saveable,
    combine_remat_policies,
    default_remat_combine_fn,
    offload_dots_saveable,
)


class TestFlashAttentionRemat(TestCase):
    """Tests FlashAttention remat policy."""

    def _get_remat_test_data(self, use_segment_ids):
        if jax.default_backend() not in ("gpu", "tpu"):
            self.skipTest("Requires TPU or GPU to run.")
        batch = 8
        seq_len = 128
        num_heads = 1
        per_head_dim = 128
        # Mesh shape doesn't matter. Find any supported mesh shape.
        mesh = (8, 1)
        if not is_supported_mesh_shape(mesh):
            mesh = (4, 1)
        if not is_supported_mesh_shape(mesh):
            mesh = (1, 1)
        mesh_axis_names = ("data", "model")
        mesh = Mesh(mesh_utils.create_device_mesh(mesh), mesh_axis_names)
        with mesh:
            hidden_dim = num_heads * per_head_dim

            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
                causal=True,
                mask=None,
            )
            test_cfg = DummyModel.default_config().set(
                layer=FlashAttention.default_config().set(
                    tpu_block_size=128,
                    mha_dim_to_partition_spec=default_mha_dim_to_partition_spec(mesh_axis_names),
                    output_dim_to_partition_spec=default_output_dim_to_partition_spec(
                        mesh_axis_names
                    ),
                    **kwargs,
                )
            )
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Use the same params for both. Only attention implementation differs.
            params = test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                kv_len=seq_len,
                query_len=seq_len,
                hidden_dim=hidden_dim,
                use_bias=False,
                use_segment_ids=use_segment_ids,
            )

            def loss(params, inputs):
                loss, _ = F(
                    test_layer,
                    inputs=inputs,
                    state=params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )
                return loss

        return mesh, params, inputs, loss

    @parameterized.product(
        use_segment_ids=[True, False],
        remat_type=[Saveable, Offloadable(src="device", dst="pinned_host")],
    )
    def test_flash_remat(self, use_segment_ids, remat_type):
        if jax.default_backend() not in ("gpu", "tpu"):
            self.skipTest("Remat test requires either GPU or TPU.")

        mesh, params, inputs, loss = self._get_remat_test_data(use_segment_ids)
        with mesh:
            no_remat = jax.value_and_grad(loss)
            full_remat = jax.value_and_grad(jax.remat(loss))
            save_flash = jax.value_and_grad(
                jax.remat(loss, policy=save_or_offload_flash_attention_policy(remat_type))
            )
            fn_expected_fw_count = [
                (no_remat, 1),
                (full_remat, 2),
                (save_flash, 1),
            ]

            if jax.default_backend() == "gpu":
                if use_segment_ids:
                    # Pallas kernel case.
                    for fn, count in fn_expected_fw_count:
                        self.assertEqual(
                            jax.jit(fn)
                            .lower(params, inputs)
                            .as_text("hlo")
                            .count('custom_call_target="__gpu$xla.gpu.triton"'),
                            # +1 because this custom call also matches the backward call.
                            # also +1 since the backward kernels are not fused. I.e. there are two
                            # calls, one for dkdv and one for dq.
                            count + 2,
                        )
                else:
                    if isinstance(remat_type, Offloadable):
                        with self.assertRaises(NotImplementedError):
                            jax.jit(save_flash).lower(params, inputs).as_text("hlo")
                        return
                    # cuDNN case.
                    # Note: the backward kernel is called "__cudnn$fmhaSoftmaxBackward".
                    # Use " to distinguish forward and backward kernel.
                    for fn, count in fn_expected_fw_count:
                        self.assertEqual(
                            jax.jit(fn)
                            .lower(params, inputs)
                            .as_text("hlo")
                            .count('"__cudnn$fmhaSoftmax"'),
                            count,
                        )
            elif jax.default_backend() == "tpu":
                for fn, count in fn_expected_fw_count:
                    self.assertEqual(
                        jax.jit(fn)
                        .lower(params, inputs)
                        .as_text("hlo")
                        .count('custom_call_target="tpu_custom_call"'),
                        # +1 because this custom call also matches the backward call.
                        # Also +1 for legacy code path since the backward kernels are
                        # not fused. I.e. there are two calls, one for dkdv and one for dq.
                        count + 1 + int(use_segment_ids),
                    )

    def test_remat_combine_policy(self):
        if jax.default_backend() != "gpu":
            self.skipTest("Need GPU for this test.")
        mesh, params, inputs, loss = self._get_remat_test_data(True)
        with mesh:
            no_remat = jax.value_and_grad(loss)
            no_remat_dots_count = str(jax.jit(no_remat).lower(params, inputs).as_text("hlo")).count(
                " dot("
            )
            offload = Offloadable(src="device", dst="pinned_host")
            remat = jax.value_and_grad(
                jax.remat(
                    loss,
                    policy=combine_remat_policies(
                        checkpoint_policies.dots_saveable,
                        save_or_offload_flash_attention_policy(),
                    ),
                )
            )

            remat_hlo = str(jax.jit(remat).lower(params, inputs).as_text("hlo"))
            self.assertEqual(
                remat_hlo.count('custom_call_target="__gpu$xla.gpu.triton"'),
                3,
            )
            self.assertEqual(
                remat_hlo.count(" dot("),
                no_remat_dots_count,
            )

            with self.assertRaises(RuntimeError):
                # Tests conflicting save policy for dots.
                remat = jax.value_and_grad(
                    jax.remat(
                        loss,
                        policy=combine_remat_policies(
                            checkpoint_policies.everything_saveable,
                            offload_dots_saveable("device", "pinned_host"),
                        ),
                    )
                )
                jax.jit(remat).lower(params, inputs).as_text("hlo")

            # Tests conflicting save policy should works if preferred type is specified.
            for preferred in [Saveable, offload]:
                remat = jax.value_and_grad(
                    jax.remat(
                        loss,
                        policy=combine_remat_policies(
                            checkpoint_policies.everything_saveable,
                            config_for_function(offload_dots_saveable).set(
                                offload_src="device", offload_dst="pinned_host"
                            ),
                            combine_fn=config_for_function(default_remat_combine_fn).set(
                                preferred_remat_type=preferred
                            ),
                        ),
                    )
                )
                self.assertEqual(
                    str(jax.jit(remat).lower(params, inputs).as_text("hlo")).count(" dot("),
                    no_remat_dots_count,
                )
