"""Tests for RAttention and ResidualLinearAttention."""
import copy

import jax
import jax.numpy as jnp
import jax.random
from absl.testing import parameterized
from jax._src.mesh import ResourceEnv, thread_resources
from jax.experimental import mesh_utils
from jax.sharding import Mesh

from axlearn.common.attention import (
    BaseQKVLinear,
    FusedGroupedQKVLinear,
    QLinear,
    RoFormerQKVLinear,
    RoFormerSinusoidalPositionalEmbedding,
)
from axlearn.common.attention_bias import SlidingWindowAttentionBias
from axlearn.common.flash_attention.layer import FlashAttention
from axlearn.common.module import functional as F
from axlearn.common.rattention.kernels.utils import FeatureMap
from axlearn.common.rattention.rattention import RAttention, ResidualLinearAttention
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import TensorSpec


class ResidualLinearAttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        num_devices = jax.device_count()
        devices = mesh_utils.create_device_mesh((1, 1, 1, 1, num_devices, 1))
        global_mesh = Mesh(devices, axis_names=("data", "expert", "fsdp", "seq", "track", "model"))
        new_env = ResourceEnv(physical_mesh=global_mesh)
        self.orig_env = thread_resources.env
        thread_resources.env = new_env

    def tearDown(self):
        super().tearDown()
        thread_resources.env = self.orig_env

    @parameterized.product(
        num_heads=[1, 2],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
    )
    def test_basic(self, num_heads: int, feat_fn: str, use_learned_init: bool, use_qk_scale: bool):
        with jax.checking_leaks():
            batch, seq_len, per_head_dim = 2, 32, 8
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, num_heads * per_head_dim))
            q_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            k_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            v_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            qkv_proj = BaseQKVLinear.Output(query=q_proj, key=k_proj, value=v_proj)

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                input_dim=model_dim,
                hidden_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                sliding_window_size=7,
                feat_fn=FeatureMap(feat_fn),
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                chunk_size=8,
            )

            res_lattn = res_lattn_cfg.set(name="test").instantiate(parent=None)
            res_lattn_params = res_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )

            outputs, _ = F(
                res_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=res_lattn_params,
                is_training=False,
                inputs={"query": query, "qkv_proj": qkv_proj},
            )
            self.assertEqual(outputs.shape, (batch, seq_len, num_heads, per_head_dim))

    # pylint: disable=no-self-use
    @parameterized.product(
        num_heads=[1, 2],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
    )
    def test_extend(self, num_heads: int, feat_fn: str, use_learned_init: bool, use_qk_scale: bool):
        with jax.checking_leaks():
            dtype = jnp.float32
            batch, seq_len, per_head_dim, sliding_window_size = 2, 8, 4, 5
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, num_heads * per_head_dim))
            q_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            k_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            v_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            qkv_proj = BaseQKVLinear.Output(query=q_proj, key=k_proj, value=v_proj)

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                input_dim=model_dim,
                hidden_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                sliding_window_size=sliding_window_size,
                feat_fn=FeatureMap(feat_fn),
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                chunk_size=8,
            )

            res_lattn = res_lattn_cfg.set(name="test").instantiate(parent=None)
            res_lattn_params = res_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )
            forward_outputs, _ = F(
                res_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=res_lattn_params,
                is_training=False,
                inputs={"query": query, "qkv_proj": qkv_proj},
            )

            query_spec = TensorSpec([batch, seq_len, model_dim])
            init_state, _ = F(
                res_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=res_lattn_params,
                is_training=False,
                inputs={"query": query_spec, "time_step": None},
                method="init_states",
            )
            init_state = init_state[0]

            decoder_output_data = jnp.zeros((seq_len, batch, num_heads, per_head_dim), dtype=dtype)

            # `cached_state` contains `time_step`, `hidden states` and optional `gate`.
            extend_state = {"cached_states": init_state}
            for t in range(seq_len):
                extend_state["query"] = query[:, t : t + 1, :]
                extend_state["qkv_proj"] = BaseQKVLinear.Output(
                    query=q_proj[:, t : t + 1, :, :], key=k_proj, value=v_proj
                )
                (next_state, extend_step_output), _ = F(
                    res_lattn,
                    state=res_lattn_params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    inputs=extend_state,
                    method="extend_step",
                )
                extend_state["cached_states"] = next_state
                decoder_output_data = decoder_output_data.at[t].set(
                    jnp.squeeze(extend_step_output, axis=1)
                )

            decoder_output_data = jnp.transpose(decoder_output_data, [1, 0, 2, 3])
            assert_allclose(decoder_output_data, forward_outputs, atol=2e-1)

    @parameterized.product(
        num_heads=[1, 2],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
    )
    def test_prefill(
        self, num_heads: int, feat_fn: str, use_learned_init: bool, use_qk_scale: bool
    ):
        del self
        with jax.checking_leaks():
            dtype = jnp.float32
            batch, seq_len, per_head_dim, sliding_window_size = 2, 8, 4, 5
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, num_heads * per_head_dim))
            q_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            k_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            v_proj = jax.random.normal(rng, (batch, seq_len, num_heads, per_head_dim))
            qkv_proj = BaseQKVLinear.Output(query=q_proj, key=k_proj, value=v_proj)
            time_step = jnp.arange(batch, dtype=jnp.int32) + 6

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                input_dim=model_dim,
                hidden_dim=model_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                sliding_window_size=sliding_window_size,
                feat_fn=FeatureMap(feat_fn),
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                chunk_size=8,
            )

            res_lattn = res_lattn_cfg.set(name="test").instantiate(parent=None)
            res_lattn_params = res_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )
            forward_outputs, _ = F(
                res_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=res_lattn_params,
                is_training=False,
                inputs={"query": query, "qkv_proj": qkv_proj},
            )

            # Prefill + Extend outputs.
            (prefill_state, prefill_output), _ = F(
                res_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=res_lattn_params,
                is_training=False,
                inputs={"query": query, "qkv_proj": qkv_proj, "time_step": time_step},
                method="init_states",
            )

            time_step_mask = (jnp.arange(seq_len)[None, :] < time_step[:, None]).astype(dtype)
            decoder_output = prefill_output * time_step_mask[..., None, None]

            extend_inputs = {"cached_states": prefill_state}

            for _ in range(seq_len):
                extend_inputs["query"] = jnp.take_along_axis(
                    query, time_step[:, None, None], axis=1, mode="clip"
                )
                extend_inputs["qkv_proj"] = BaseQKVLinear.Output(
                    query=jnp.take_along_axis(
                        q_proj, time_step[:, None, None, None], axis=1, mode="clip"
                    ),
                    key=k_proj,
                    value=v_proj,
                )
                (next_state, extend_step_output), _ = F(
                    res_lattn,
                    state=res_lattn_params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    inputs=extend_inputs,
                    method="extend_step",
                )
                # `time_step` within cached_states are updated automatically
                extend_inputs["cached_states"] = next_state

                # [batch, 1, num_heads, per_head_dim]
                cur_query_output = extend_step_output
                # [batch, seq_len, 1, 1]
                oh_indices = jax.nn.one_hot(time_step, seq_len, dtype=dtype)[..., None, None]
                # [batch, seq_len, 1, 1]
                decoder_output = decoder_output + cur_query_output * oh_indices

                # Update time_step to retrieve inputs.
                time_step = time_step + 1

            assert_allclose(decoder_output, forward_outputs, atol=2e-1)


class RAttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        num_devices = jax.device_count()
        devices = mesh_utils.create_device_mesh((1, 1, 1, 1, num_devices, 1))
        global_mesh = Mesh(devices, axis_names=("data", "expert", "fsdp", "seq", "track", "model"))
        new_env = ResourceEnv(physical_mesh=global_mesh)
        self.orig_env = thread_resources.env
        thread_resources.env = new_env

    def tearDown(self):
        super().tearDown()
        thread_resources.env = self.orig_env

    @parameterized.product(
        num_heads=[8, 16],
        sliding_window_size=[
            31,
        ],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
    )
    def test_basic(
        self,
        num_heads: int,
        feat_fn: str,
        sliding_window_size: int,
        use_learned_init: bool,
        use_qk_scale: bool,
    ):
        with jax.checking_leaks():
            batch, seq_len, per_head_dim = 2, 32, 8
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, model_dim))

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                feat_fn=FeatureMap(feat_fn),
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                num_kv_heads=num_heads,
                chunk_size=8,
            )

            rnn_lattn_cfg = RAttention.default_config().set(
                hidden_dim=model_dim,
                num_heads=num_heads,
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                sliding_window_size=sliding_window_size,
                residual_la=res_lattn_cfg,
            )

            rnn_lattn = rnn_lattn_cfg.set(name="test").instantiate(parent=None)
            rnn_lattn_params = rnn_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )

            outputs, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query},
            )

            self.assertEqual(outputs.data.shape, (batch, seq_len, model_dim))

    @parameterized.product(
        num_heads=[2, 4],
        num_kv_heads=[1, 2],
        sliding_window_size=[
            31,
        ],
    )
    def test_against_attention(
        self,
        num_heads: int,
        num_kv_heads: int,
        sliding_window_size: int,
    ):
        """Disabling residual linear attention to see if it matches FlashAttention."""
        with jax.checking_leaks():
            batch, seq_len, per_head_dim = 2, 32, 8
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, model_dim))

            # 1. Test Forward pass.
            rnn_lattn_cfg = RAttention.default_config().set(
                hidden_dim=model_dim,
                num_heads=num_heads,
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                sliding_window_size=sliding_window_size,
                residual_la=None,
                input_linear=FusedGroupedQKVLinear.default_config().set(
                    num_kv_heads=num_kv_heads,
                ),
            )

            rnn_lattn = rnn_lattn_cfg.set(name="test").instantiate(parent=None)
            rnn_lattn_params = rnn_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )

            rnn_lattn_outputs, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query, "return_aux": {"kv_state"}},
            )

            rope_linear_cfg = RoFormerQKVLinear.default_config().set(
                input_linear=FusedGroupedQKVLinear.default_config().set(
                    num_kv_heads=num_kv_heads,
                ),
                rope_pos_emb_layer=(
                    RoFormerSinusoidalPositionalEmbedding.default_config().set(
                        theta=rnn_lattn_cfg.rope_theta,
                    )
                ),
                rotary_value=False,
            )
            flash_attn_cfg = FlashAttention.default_config().set(
                hidden_dim=model_dim,
                num_heads=num_heads,
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                input_linear=rope_linear_cfg,
                mask=SlidingWindowAttentionBias.default_config(
                    sliding_window_size=sliding_window_size
                ),
            )

            flash_attn = flash_attn_cfg.set(name="test").instantiate(parent=None)
            # Copy the RAttention parameters to FlashAttention.
            flash_attn_params = copy.deepcopy(rnn_lattn_params)
            flash_attn_params["i_proj"] = {"i_proj": rnn_lattn_params["i_proj"]}
            flash_attn_outputs, _ = F(
                flash_attn,
                prng_key=jax.random.PRNGKey(123),
                state=flash_attn_params,
                is_training=False,
                inputs={"query": query, "return_aux": {"kv_state"}},
            )

            self.assertEqual(rnn_lattn_outputs.data.shape, (batch, seq_len, model_dim))
            self.assertEqual(flash_attn_outputs.data.shape, (batch, seq_len, model_dim))

            # keys are not the same because RAttention outputs key without RoPE.
            # values and key_positions are the same.
            assert_allclose(
                rnn_lattn_outputs.kv_state[1], flash_attn_outputs.kv_state[1], atol=1e-3
            )
            assert_allclose(
                rnn_lattn_outputs.kv_state[2], flash_attn_outputs.kv_state[2], atol=1e-3
            )

            # 2. Test Extend.
            query_spec = TensorSpec([batch, seq_len, model_dim], dtype=jnp.float32)
            rattn_init_state, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query_spec, "time_step": None},
                method="init_states",
            )[0]
            flash_attn_init_state, _ = F(
                flash_attn,
                prng_key=jax.random.PRNGKey(123),
                state=flash_attn_params,
                is_training=False,
                inputs={"query": query_spec, "time_step": None},
                method="init_states",
            )[0]

            for t in range(3):
                rattn_extend_state = {"cached_states": rattn_init_state, "return_aux": {"kv_state"}}
                rattn_extend_state["query"] = query[:, t : t + 1, :]
                (rattn_next_state, rattn_extend_step_output), _ = F(
                    rnn_lattn,
                    state=rnn_lattn_params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    inputs=rattn_extend_state,
                    method="extend_step",
                )
                rattn_extend_state["cached_states"] = rattn_next_state

                flash_attn_extend_state = {
                    "cached_states": flash_attn_init_state,
                    "return_aux": {"kv_state"},
                }
                flash_attn_extend_state["query"] = query[:, t : t + 1, :]
                (flash_attn_next_state, flash_attn_extend_step_output), _ = F(
                    flash_attn,
                    state=flash_attn_params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    inputs=flash_attn_extend_state,
                    method="extend_step",
                )
                flash_attn_extend_state["cached_states"] = flash_attn_next_state
                assert_allclose(
                    rattn_extend_step_output.data,
                    flash_attn_extend_step_output.data,
                    atol=1e-3,
                )

                # Again, keys are not the same because RAttention outputs key without RoPE.
                assert_allclose(
                    rattn_extend_state["cached_states"]["swa_state"]["value"],
                    flash_attn_extend_state["cached_states"]["kv_cache"]["value"],
                    atol=1e-3,
                )
                assert_allclose(
                    rattn_extend_step_output.kv_state[1],
                    flash_attn_extend_step_output.kv_state[1],
                    atol=1e-3,
                )
                assert_allclose(
                    rattn_extend_step_output.kv_state[2],
                    flash_attn_extend_step_output.kv_state[2],
                    atol=1e-3,
                )

        # Test Prefill.
        time_step = jnp.arange(batch, dtype=jnp.int32) + 6
        rattn_prefill_state, rattn_prefill_output = F(
            rnn_lattn,
            prng_key=jax.random.PRNGKey(123),
            state=rnn_lattn_params,
            is_training=False,
            inputs={
                "query": query,
                "time_step": time_step,
                "return_aux": {"kv_state"},
            },
            method="init_states",
        )[0]

        flash_attn_prefill_state, flash_attn_prefill_output = F(
            flash_attn,
            prng_key=jax.random.PRNGKey(123),
            state=flash_attn_params,
            is_training=False,
            inputs={
                "query": query,
                "time_step": time_step,
                "return_aux": {"kv_state"},
            },
            method="init_states",
        )[0]
        assert_allclose(rattn_prefill_output.data, flash_attn_prefill_output.data, atol=1e-3)

        # Again, keys are not the same because RAttention outputs key without RoPE.
        assert_allclose(
            rattn_prefill_state["swa_state"]["value"],
            flash_attn_prefill_state["kv_cache"]["value"],
            atol=1e-3,
        )
        assert_allclose(
            rattn_prefill_output.kv_state[1], flash_attn_prefill_output.kv_state[1], atol=1e-3
        )
        assert_allclose(
            rattn_prefill_output.kv_state[2], flash_attn_prefill_output.kv_state[2], atol=1e-3
        )

    # pylint: disable=no-self-use
    @parameterized.product(
        num_heads=[1, 2],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
        sliding_window_size=[11],
    )
    def test_extend(
        self,
        num_heads: int,
        feat_fn: str,
        sliding_window_size: int,
        use_learned_init: bool,
        use_qk_scale: bool,
    ):
        with jax.checking_leaks():
            batch, seq_len, per_head_dim = 2, 8, 4
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, model_dim))

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                feat_fn=FeatureMap(feat_fn),
                num_kv_heads=num_heads,
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                chunk_size=8,
            )

            rnn_lattn_cfg = RAttention.default_config().set(
                hidden_dim=model_dim,
                num_heads=num_heads,
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                sliding_window_size=sliding_window_size,
                residual_la=res_lattn_cfg,
            )

            rnn_lattn = rnn_lattn_cfg.set(name="test").instantiate(parent=None)
            rnn_lattn_params = rnn_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )

            forward_outputs, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query},
            )

            query_spec = TensorSpec([batch, seq_len, model_dim], dtype=jnp.float32)
            init_state, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query_spec, "time_step": None},
                method="init_states",
            )
            init_state = init_state[0]

            decoder_output_data = jnp.zeros((seq_len, batch, model_dim), dtype=jnp.float32)
            extend_state = {"cached_states": init_state}
            for t in range(seq_len):
                extend_state["query"] = query[:, t : t + 1, :]
                (next_state, extend_step_output), _ = F(
                    rnn_lattn,
                    state=rnn_lattn_params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    inputs=extend_state,
                    method="extend_step",
                )
                extend_state["cached_states"] = next_state
                decoder_output_data = decoder_output_data.at[t].set(
                    jnp.squeeze(extend_step_output.data, axis=1)
                )

            decoder_output_data = jnp.transpose(decoder_output_data, [1, 0, 2])
            assert_allclose(decoder_output_data, forward_outputs.data, atol=2e-1)

    # pylint: disable=no-self-use
    @parameterized.product(
        num_heads=[1, 2],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
        sliding_window_size=[11],
    )
    def test_prefill(
        self,
        num_heads: int,
        feat_fn: str,
        sliding_window_size: int,
        use_learned_init: bool,
        use_qk_scale: bool,
    ):
        with jax.checking_leaks():
            dtype = jnp.float32
            batch, seq_len, per_head_dim = 1, 8, 4
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, model_dim))
            time_step = jnp.arange(batch, dtype=jnp.int32)

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                feat_fn=FeatureMap(feat_fn),
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                num_kv_heads=num_heads,
                chunk_size=8,
            )

            rnn_lattn_cfg = RAttention.default_config().set(
                hidden_dim=model_dim,
                num_heads=num_heads,
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                sliding_window_size=sliding_window_size,
                residual_la=res_lattn_cfg,
            )

            rnn_lattn = rnn_lattn_cfg.set(name="test").instantiate(parent=None)
            rnn_lattn_params = rnn_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )

            forward_outputs, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query},
            )

            # Prefill + Extend outputs.
            (prefill_state, prefill_output), _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={
                    "query": query,
                    "time_step": time_step,
                },
                method="init_states",
            )

            time_step_mask = (jnp.arange(seq_len)[None, :] < time_step[:, None]).astype(dtype)
            decoder_output = prefill_output.data * time_step_mask[..., None]

            extend_inputs = {"cached_states": prefill_state}

            for _ in range(seq_len):
                extend_inputs["query"] = jnp.take_along_axis(
                    query, time_step[:, None, None], axis=1, mode="clip"
                )
                (next_state, extend_step_output), _ = F(
                    rnn_lattn,
                    state=rnn_lattn_params,
                    is_training=False,
                    prng_key=jax.random.PRNGKey(123),
                    inputs=extend_inputs,
                    method="extend_step",
                )
                # `time_step` within cached_states are updated automatically
                extend_inputs["cached_states"] = next_state

                # [batch, 1, model_dim]
                cur_query_output = extend_step_output.data
                # [batch, seq_len, 1]
                oh_indices = jax.nn.one_hot(time_step, seq_len, dtype=dtype)[..., None]
                # [batch, seq_len, 1]
                decoder_output = decoder_output + cur_query_output * oh_indices

                # Update time_step to retrieve inputs.
                time_step = time_step + 1

            assert_allclose(decoder_output, forward_outputs.data, atol=2e-1)

    @parameterized.product(
        num_heads=[1, 2],
        sliding_window_size=[
            31,
        ],
        feat_fn=["softmax", "relu"],
        use_learned_init=[True, False],
        use_qk_scale=[True, False],
    )
    def test_external_state(
        self,
        num_heads: int,
        feat_fn: str,
        sliding_window_size: int,
        use_learned_init: bool,
        use_qk_scale: bool,
    ):
        with jax.checking_leaks():
            batch, seq_len, per_head_dim = 2, 16, 8
            model_dim = per_head_dim * num_heads
            rng = jax.random.PRNGKey(345)
            query = jax.random.normal(rng, (batch, seq_len, model_dim))

            res_lattn_cfg = ResidualLinearAttention.default_config().set(
                feat_fn=FeatureMap(feat_fn),
                use_learned_init=use_learned_init,
                use_qk_scale=use_qk_scale,
                num_kv_heads=num_heads,
                chunk_size=8,
            )

            rnn_lattn_cfg = RAttention.default_config().set(
                hidden_dim=model_dim,
                num_heads=num_heads,
                query_dim=model_dim,
                key_dim=model_dim,
                value_dim=model_dim,
                sliding_window_size=sliding_window_size,
                residual_la=res_lattn_cfg,
            )

            rnn_lattn = rnn_lattn_cfg.set(name="test").instantiate(parent=None)
            rnn_lattn_params = rnn_lattn.initialize_parameters_recursively(
                prng_key=jax.random.PRNGKey(678)
            )

            outputs_1, _ = F(
                rnn_lattn,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query, "return_aux": {"kv_state"}},
            )

            rnn_lattn_cfg_2 = rnn_lattn_cfg.clone()
            rnn_lattn_cfg_2.input_linear = QLinear.default_config()
            rnn_lattn_2 = rnn_lattn_cfg_2.set(name="test").instantiate(parent=None)
            for k in ["k_proj", "v_proj"]:
                rnn_lattn_params["i_proj"].pop(k)
            outputs_2, _ = F(
                rnn_lattn_2,
                prng_key=jax.random.PRNGKey(123),
                state=rnn_lattn_params,
                is_training=False,
                inputs={"query": query, "kv_state": outputs_1.kv_state, "return_aux": {"kv_state"}},
            )

            assert_allclose(outputs_1.data, outputs_2.data, atol=1e-3)
            # Make sure that the external KV state is passed along unchanged.
            self.assertNestedAllClose(outputs_1.kv_state, outputs_2.kv_state, atol=1e-3)
