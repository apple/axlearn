# Copyright © 2023 Apple Inc.

"""Tests FlashAttention layers."""
import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized
from jax.experimental import mesh_utils
from jax.sharding import Mesh

try:
    import jax_triton as jt  # pytype: disable=import-error  # pylint: disable=import-error

    from axlearn.common.flash_attention.layer import FlashAttention

    if jt.get_compute_capability(0) < 80:
        pytest.skip(reason="Incompatible hardware.", allow_module_level=True)
except ModuleNotFoundError as e:
    # Some libraries can only be installed on GPU, so we'll skip on CI.
    pytest.skip(
        reason=f"Skipping flash_attention tests due to missing deps: {e}",
        allow_module_level=True,
    )

from axlearn.common.attention import (
    MultiheadAttention,
    apply_attention_logit_biases,
    make_causal_mask,
)
from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import config_class
from axlearn.common.layers import set_bias_recursively
from axlearn.common.module import Module
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase, is_supported_mesh_shape


def _fake_inputs(*, batch: int, num_heads: int, seq_len: int, hidden_dim: int):
    query = jax.random.normal(
        jax.random.PRNGKey(0),
        [batch, seq_len, hidden_dim],
        dtype=jnp.bfloat16,
    )
    key = jax.random.normal(
        jax.random.PRNGKey(1),
        [batch, seq_len, hidden_dim],
        dtype=jnp.bfloat16,
    )
    value = jax.random.normal(
        jax.random.PRNGKey(2),
        [batch, seq_len, hidden_dim],
        dtype=jnp.bfloat16,
    )
    bias = jax.random.normal(
        jax.random.PRNGKey(3),
        [batch, num_heads, seq_len, seq_len],
        dtype=jnp.bfloat16,
    )
    return dict(query=query, key=key, value=value, attention_logit_biases=bias)


class TestFlashAttention(TestCase):
    """Tests FlashAttention layer."""

    @parameterized.product(
        [
            dict(batch=2, seq_len=384, num_heads=4, per_head_dim=32, mesh=(1, 1)),
            dict(batch=2, seq_len=2048, num_heads=4, per_head_dim=64, mesh=(1, 1)),
            dict(batch=8, seq_len=2048, num_heads=4, per_head_dim=64, mesh=(8, 1)),
        ],
        causal=[False, True],
    )
    def test_forward(self, batch, seq_len, num_heads, per_head_dim, mesh, causal):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

        with Mesh(mesh_utils.create_device_mesh(mesh), ("data", "model")):
            hidden_dim = num_heads * per_head_dim
            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
            )
            ref_cfg = MultiheadAttention.default_config().set(**kwargs)
            test_cfg = FlashAttention.default_config().set(**kwargs)
            test_cfg.set(causal=causal)
            set_bias_recursively(ref_cfg, False)
            set_bias_recursively(test_cfg, False)

            ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Use the same params for both. Only attention implementation differs.
            params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
            )
            ref_inputs = inputs
            if causal:
                # Apply causal mask to ref_inputs.
                ref_inputs["attention_logit_biases"] = apply_attention_logit_biases(
                    inputs["attention_logit_biases"], make_causal_mask(seq_len)
                )

            ref_out, _ = F(
                ref_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=ref_inputs,
                is_training=True,
            )
            test_out, _ = F(
                test_layer,
                prng_key=jax.random.PRNGKey(5),
                state=params,
                inputs=inputs,
                is_training=True,
            )
            # TODO(markblee): Test probs.
            self.assertNestedAllClose(ref_out.data, test_out.data, atol=0.05)

    @parameterized.product(
        [
            dict(batch=2, seq_len=384, num_heads=4, per_head_dim=32, mesh=(1, 1)),
            dict(batch=2, seq_len=2048, num_heads=4, per_head_dim=64, mesh=(1, 1)),
            dict(batch=8, seq_len=2048, num_heads=4, per_head_dim=64, mesh=(8, 1)),
        ],
        causal=[False, True],
    )
    def test_backward(self, batch, seq_len, num_heads, per_head_dim, mesh, causal):
        if not is_supported_mesh_shape(mesh):
            pytest.skip(reason=f"Unsupported mesh {mesh}.")

        with Mesh(mesh_utils.create_device_mesh(mesh), ("data", "model")):

            class DummyModel(BaseLayer):
                """A dummy model."""

                @config_class
                class Config(BaseLayer.Config):
                    layer: MultiheadAttention.Config = MultiheadAttention.default_config()

                def __init__(self, cfg: Config, *, parent: Module):
                    super().__init__(cfg, parent=parent)
                    cfg = self.config
                    self._add_child("layer", cfg.layer)

                def forward(self, *, query, key, value, attention_logit_biases):
                    # [batch, target_length, target_dim].
                    x = self.layer(
                        query,
                        key=key,
                        value=value,
                        attention_logit_biases=attention_logit_biases,
                    )
                    # TODO(markblee,zhaoyi-zhang): The atol needs to increase significantly if using
                    # jnp.sum, as we no longer scale by the size of the data dims.
                    return jnp.mean(x.data, dtype=query.dtype)

            hidden_dim = num_heads * per_head_dim
            kwargs = dict(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                num_heads=num_heads,
                dtype=jnp.bfloat16,
            )
            ref_cfg = DummyModel.default_config().set(
                layer=MultiheadAttention.default_config().set(**kwargs),
            )
            test_cfg = DummyModel.default_config().set(
                layer=FlashAttention.default_config().set(**kwargs).set(causal=causal),
            )
            set_bias_recursively(ref_cfg, False)
            set_bias_recursively(test_cfg, False)
            ref_layer = ref_cfg.set(name="ref").instantiate(parent=None)
            test_layer = test_cfg.set(name="test").instantiate(parent=None)

            # Use the same params for both. Only attention implementation differs.
            params = ref_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
            inputs = _fake_inputs(
                batch=batch,
                num_heads=num_heads,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
            )
            ref_inputs = inputs
            if causal:
                # Apply causal mask to ref_inputs.
                ref_inputs["attention_logit_biases"] = apply_attention_logit_biases(
                    inputs["attention_logit_biases"], make_causal_mask(seq_len)
                )

            def loss(params, inputs, layer):
                loss, _ = F(
                    layer,
                    inputs=inputs,
                    state=params,
                    is_training=True,
                    prng_key=jax.random.PRNGKey(0),
                )
                return loss

            ref_value, ref_grads = jax.value_and_grad(loss)(params, ref_inputs, ref_layer)
            test_value, test_grads = jax.value_and_grad(loss)(params, inputs, test_layer)

            self.assertNestedAllClose(ref_value, test_value, atol=1e-5)
            self.assertNestedAllClose(ref_grads, test_grads, atol=1e-5)
