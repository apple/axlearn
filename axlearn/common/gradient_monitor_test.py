# Copyright © 2024 Apple Inc.

"""Tests for gradient_monitor.py"""
from functools import partial
from unittest.mock import patch

import chex
import jax
import jax.numpy as jnp
from absl.testing import parameterized

from axlearn.common import optimizers, schedule
from axlearn.common.attention import MultiheadAttention, TransformerFeedForwardLayer
from axlearn.common.base_layer import BaseLayer, NestedTensor, ParameterSpec
from axlearn.common.config import config_class, config_for_function
from axlearn.common.gradient_monitor import (
    GRADIENT_CLIPPING_PATHS,
    GradientMonitorAndClipLayer,
    compute_grad_percentile_no_clip_fn,
    convert_to_monitored_layer_config,
    create_monitored_layer_class,
    gradient_clipping_impl,
    gradient_monitoring_learner_cfg_modifier,
    top_k_clip_fn,
)
from axlearn.common.kv_cache.base_kv_cache import KVState
from axlearn.common.learner import ForwardOutputs, Learner
from axlearn.common.module import child_context
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection
from axlearn.common.optimizer_base import OptParam
from axlearn.common.test_utils import TestCase


class DummyLayer(BaseLayer):
    """Simple layer for testing: y = x @ params"""

    @config_class
    class Config(BaseLayer.Config):
        indim: int = 2
        outdim: int = 2

    def _create_layer_parameter_specs(self):
        cfg = self.config
        return dict(weight=ParameterSpec(shape=(cfg.indim, cfg.outdim)))

    def forward(self, x):
        return x @ self.parameters["weight"]


def dummy_clipping(norm_threshold=5):
    """Set activation gradient to 0 if per token norm is large."""

    def fn(g):
        per_token_norm = jnp.linalg.norm(g, axis=-1)
        d = g.shape[-1]
        is_valid = jnp.expand_dims(per_token_norm < norm_threshold * jnp.sqrt(d), -1)
        return jnp.where(is_valid, g, 0), jnp.array((0, 0, 0, 0))  # dummy stats

    return fn


def _create_base_and_montored_layers(
    layer_type="ffn", input_dim=4, hidden_dim=16, num_heads=2, clip_fn=None
):
    if layer_type == "attn":
        base_cfg = MultiheadAttention.default_config().set(
            name="base_attn",
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            output_dim=input_dim,
            num_heads=num_heads,
        )
    elif layer_type == "ffn":
        base_cfg = TransformerFeedForwardLayer.default_config().set(
            name="base_ffn",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        base_cfg.linear1.param_partition_spec = (None, None)  # Make sure layer works without mesh.
        base_cfg.linear2.param_partition_spec = (None, None)
    else:
        raise ValueError(f"Unsupport layer_type {layer_type}. Available types: attn, ffn.")
    base_layer = base_cfg.instantiate(parent=None)

    gradient_monitor_cfg = None
    if clip_fn is not None:
        gradient_monitor_cfg = GradientMonitorAndClipLayer.default_config().set(clip_fn=clip_fn)
    monitored_cfg = convert_to_monitored_layer_config(base_cfg, gradient_monitor_cfg)
    monitored_cfg.set(name="monitored_layer")
    monitored_layer = monitored_cfg.instantiate(parent=None)
    return base_layer, monitored_layer


def _setup_learner_and_model(
    layer_type="ffn", input_dim=4, ema_decay=None, use_monitoring=True, clip_fn=None
):
    """Helper function to setup model, learner and states for testing."""
    # Init a model with or without gradient monitoring.
    base, monitored = _create_base_and_montored_layers(
        layer_type=layer_type, input_dim=input_dim, clip_fn=clip_fn
    )
    model = monitored if use_monitoring else base
    model_params = model.initialize_parameters_recursively(jax.random.PRNGKey(123))

    # Init a learner.
    learner_cfg = Learner.default_config().set(
        optimizer=config_for_function(optimizers.adam_optimizer).set(
            learning_rate=config_for_function(schedule.constant_schedule).set(value=0.1),
            b1=0.9,
            b2=0.99,
            eps=1e-5,
            l2_regularizer_weight=1.0,
        ),
    )
    learner_cfg.ema.decay = ema_decay
    if use_monitoring:
        learner_cfg = gradient_monitoring_learner_cfg_modifier(learner_cfg)
    learner_cfg = learner_cfg.set(name="test_learner")
    learner: Learner = learner_cfg.instantiate(parent=None)

    # Create opt_params for learner initialization.
    model_param_specs = model.create_parameter_specs_recursively()
    opt_params = _opt_params_from_model_params(model_params, model_param_specs)
    learner_state = learner.init(model_params=opt_params)

    # Define forward function.
    def _forward(*, model_params: NestedTensor, inputs: NestedTensor) -> ForwardOutputs:
        model_output_collection = new_output_collection()
        with child_context(
            "model",
            module=model,
            state=model_params,
            prng_key=inputs["forward_key"],
            output_collection=model_output_collection,
        ):
            if layer_type == "attn":
                output = model(query=inputs["query"], key=inputs["key"], value=inputs["value"])
                aux = {}
            else:
                output, aux = model(inputs=inputs["input_batch"])
            loss = (
                output.sum()
                if not isinstance(output, MultiheadAttention.Output)
                else output.data.sum()
            )
        return ForwardOutputs(loss=loss, aux=aux, output_collection=model_output_collection)

    return {
        "model": model,
        "model_params": model_params,
        "model_param_specs": model_param_specs,
        "learner": learner,
        "learner_state": learner_state,
        "forward_fn": _forward,
    }


def _opt_params_from_model_params(model_params, model_param_specs):
    return jax.tree.map(
        lambda param, spec: OptParam(
            value=param,
            factorization_spec=spec.factorization if spec else None,
            weight_decay_scale=spec.weight_decay_scale if spec else 1.0,
        ),
        model_params,
        model_param_specs,
    )


class GradientMonitorTest(TestCase):
    """Tests for gradient monitoring functionality."""

    def test_create_monitored_layer_class(self):
        """Test that create_monitored_layer_class creates a valid monitored layer class."""
        MonitoredAttention = create_monitored_layer_class(MultiheadAttention)
        self.assertEqual(MonitoredAttention.__name__, "MonitoredMultiheadAttention")

        cfg = MonitoredAttention.default_config()
        self.assertTrue(hasattr(cfg, "gradient_monitor"))

    @parameterized.parameters(None, compute_grad_percentile_no_clip_fn, top_k_clip_fn)
    def test_convert_to_monitored_layer_config(self, clip_fn):
        """Test that convert_to_monitored_layer_config preserves original config settings."""
        # Create original attention config
        original_cfg = MultiheadAttention.default_config().set(
            num_heads=8,
            query_dim=512,
        )
        gradient_monitor_cfg = None
        if clip_fn is not None:
            gradient_monitor_cfg = GradientMonitorAndClipLayer.default_config().set(
                clip_fn=config_for_function(clip_fn)
            )
        monitored_cfg = convert_to_monitored_layer_config(
            original_cfg, gradient_monitor_cfg=gradient_monitor_cfg
        )
        self.assertEqual(monitored_cfg.num_heads, 8)
        self.assertEqual(monitored_cfg.query_dim, 512)
        self.assertIsNotNone(monitored_cfg.gradient_monitor)
        self.assertEqual(monitored_cfg.gradient_monitor.clip_fn.fn, clip_fn or top_k_clip_fn)

    def test_gradient_monitor_forward_jit(self):
        """Test that monitored TransformerFeedForwardLayer produces same output as non-monitored.

        This test verifies:
        1. Adding gradient monitoring doesn't change the forward pass output
        2. The monitored layer works under JIT compilation
        3. Gradient stats are properly tracked
        """
        input_dim = 4
        hidden_dim = 16
        batch_size = 2
        seq_len = 8
        base_layer, monitored_layer = _create_base_and_montored_layers(
            input_dim=input_dim, hidden_dim=hidden_dim
        )

        # Initialize parameters for both layers with the same PRNG key
        prng_key = jax.random.PRNGKey(123)
        base_params = base_layer.initialize_parameters_recursively(prng_key=prng_key)
        monitored_params = monitored_layer.initialize_parameters_recursively(prng_key=prng_key)
        x = jax.random.normal(jax.random.PRNGKey(42), (batch_size, seq_len, input_dim))

        # Define forward function for base layer
        @jax.jit
        def base_forward(x):
            output, _ = F(
                base_layer,
                prng_key=jax.random.PRNGKey(0),
                state=base_params,
                inputs=dict(inputs=x),
                is_training=True,
            )
            return output

        # Define forward function for monitored layer
        @jax.jit
        def monitored_forward(x):
            output, output_collection = F(
                monitored_layer,
                prng_key=jax.random.PRNGKey(0),
                state=monitored_params,
                inputs=dict(inputs=x),
                is_training=True,
            )
            return output, output_collection

        # Compare results of both forward passes
        base_output = base_forward(x)
        monitored_output, output_collection = monitored_forward(x)
        self.assertNestedAllClose(base_output, monitored_output)

        # Verify stats summary was added to monitored layer
        self.assertIn("top_norms", output_collection.summaries["gradient_monitor"])

    @parameterized.parameters(None, 0.999)
    def test_stats_collection(self, ema_decay):
        """Test that gradient stats are correctly accumulated over multiple iterations."""
        batch_size = 2
        seq_len = 6
        input_dim = 4

        # Setup model, learner, and states
        setup = _setup_learner_and_model(
            input_dim=input_dim, ema_decay=ema_decay, use_monitoring=True
        )
        learner = setup["learner"]
        model_params = setup["model_params"]
        model_param_specs = setup["model_param_specs"]
        learner_state = setup["learner_state"]
        forward_fn = setup["forward_fn"]

        def mock_gradient_clipping(
            stats,
            g,
            *,
            clip_fn=config_for_function(top_k_clip_fn),
            supported_paths=GRADIENT_CLIPPING_PATHS,
            prev_stats=None,
        ):
            del clip_fn, supported_paths, prev_stats
            return g, jnp.array([stats, stats + 1, stats + 2, stats + 3])

        # Pass stats as a static argument so that patching can be correctly applied.
        # Note that this will cause recompilation.
        @partial(jax.jit, static_argnames=("mock_stats",))
        def step_fn(x, model_params, learner_state, forward_key, learner_key, mock_stats):
            opt_params = _opt_params_from_model_params(
                model_params, model_param_specs=model_param_specs
            )
            with patch(
                "axlearn.common.gradient_monitor.gradient_clipping_impl",
                side_effect=partial(mock_gradient_clipping, mock_stats),
            ):
                fwd_bwd_outputs, learner_output_collection = F(
                    learner,
                    method="forward_and_backward",
                    state=learner_state,
                    is_training=True,
                    prng_key=learner_key,
                    inputs=dict(
                        fn=forward_fn,
                        opt_params=opt_params,
                        inputs=dict(
                            input_batch=x,
                            forward_key=forward_key,
                        ),
                    ),
                )

            forward_outputs: ForwardOutputs = fwd_bwd_outputs.forward_outputs
            updated_model_params = fwd_bwd_outputs.backward_outputs.updated_params
            updated_learner_state = learner_output_collection.state_updates
            summaries = forward_outputs.output_collection.summaries

            return updated_model_params, updated_learner_state, summaries

        # Mock stats for each iteration
        mock_grad_stats = [10.0, 20.0, 30.0, 40.0]

        for iteration, stats in enumerate(mock_grad_stats):
            data_key, fwd_key, learner_key = jax.random.split(jax.random.PRNGKey(iteration), num=3)
            x = jax.random.normal(data_key, (batch_size, seq_len, input_dim))

            model_params, learner_state, summaries = step_fn(
                x, model_params, learner_state, fwd_key, learner_key, mock_stats=stats
            )

            # Check stats correctly updated.
            grad_stats_summary = summaries["gradient_monitor"]["top_norms"]
            grad_stats_param = model_params["gradient_monitor"]["stats"]
            assert (
                float(grad_stats_param[0]) == mock_grad_stats[iteration]
            ), "Monitoring layer param should be updated to gradient_stats of current batch."
            if iteration > 0:
                assert (
                    float(grad_stats_summary[0]) == mock_grad_stats[iteration - 1]
                ), "Previous batch gradient_stats should be written to summaries."

    def _assert_pure_states_all_close(self, base_states, monitored_states):
        # Verify initial model params are the same (excluding gradient_monitor path)
        # monitored_model_params has an extra 'gradient_monitor' key that base doesn't have
        monitored_states_pure = {
            k: v for k, v in monitored_states.items() if k != "gradient_monitor"
        }
        self.assertNestedAllClose(base_states, monitored_states_pure)

    @parameterized.product(layer_type=["attn", "ffn"], ema_decay=[None, 0.999])
    def test_training_parity(self, layer_type, ema_decay):
        """Test that adding gradient monitoring doesn't change model params or learner states."""
        batch_size = 2
        seq_len = 6
        input_dim = 4
        num_iterations = 2

        # Setup base model (without monitoring)
        base_setup = _setup_learner_and_model(
            layer_type=layer_type, input_dim=input_dim, ema_decay=ema_decay, use_monitoring=False
        )
        base_learner = base_setup["learner"]
        base_model_params = base_setup["model_params"]
        base_model_param_specs = base_setup["model_param_specs"]
        base_learner_state = base_setup["learner_state"]
        base_forward_fn = base_setup["forward_fn"]

        # Setup monitored model, use no-clipping function.
        monitored_setup = _setup_learner_and_model(
            layer_type=layer_type,
            input_dim=input_dim,
            ema_decay=ema_decay,
            use_monitoring=True,
            clip_fn=config_for_function(compute_grad_percentile_no_clip_fn),
        )
        monitored_learner = monitored_setup["learner"]
        monitored_model_params = monitored_setup["model_params"]
        monitored_model_param_specs = monitored_setup["model_param_specs"]
        monitored_learner_state = monitored_setup["learner_state"]
        monitored_forward_fn = monitored_setup["forward_fn"]

        for iteration in range(num_iterations):
            data_key, fwd_key, learner_key = jax.random.split(jax.random.PRNGKey(iteration), num=3)

            if layer_type == "attn":
                key_q, key_k, key_v = jax.random.split(data_key, num=3)
                inputs = dict(
                    query=jax.random.normal(key_q, (batch_size, seq_len, input_dim)),
                    key=jax.random.normal(key_k, (batch_size, seq_len, input_dim)),
                    value=jax.random.normal(key_v, (batch_size, seq_len, input_dim)),
                    forward_key=fwd_key,
                )
            else:
                inputs = dict(
                    input_batch=jax.random.normal(data_key, (batch_size, seq_len, input_dim)),
                    forward_key=fwd_key,
                )

            base_opt_params = _opt_params_from_model_params(
                base_model_params, base_model_param_specs
            )
            monitored_opt_params = _opt_params_from_model_params(
                monitored_model_params, monitored_model_param_specs
            )

            # Run base model
            base_fwd_bwd_outputs, base_learner_output_collection = F(
                base_learner,
                method="forward_and_backward",
                state=base_learner_state,
                is_training=True,
                prng_key=learner_key,
                inputs=dict(
                    fn=base_forward_fn,
                    opt_params=base_opt_params,
                    inputs=inputs,
                ),
            )

            # Run monitored model
            monitored_fwd_bwd_outputs, monitored_learner_output_collection = F(
                monitored_learner,
                method="forward_and_backward",
                state=monitored_learner_state,
                is_training=True,
                prng_key=learner_key,
                inputs=dict(
                    fn=monitored_forward_fn,
                    opt_params=monitored_opt_params,
                    inputs=inputs,
                ),
            )

            # Verify losses are the same
            self.assertAlmostEqual(
                base_fwd_bwd_outputs.forward_outputs.loss,
                monitored_fwd_bwd_outputs.forward_outputs.loss,
            )

            base_learner_state = base_learner_output_collection.state_updates
            base_model_params = base_fwd_bwd_outputs.backward_outputs.updated_params

            monitored_learner_state = monitored_learner_output_collection.state_updates
            monitored_model_params = monitored_fwd_bwd_outputs.backward_outputs.updated_params

            # Verify updated model params are the same after this iteration
            self._assert_pure_states_all_close(base_model_params, monitored_model_params)

            # Verify learner states are the same (optimizer states)
            self.assertNestedAllClose(
                base_learner_state["optimizer"], monitored_learner_state["inner"]["optimizer"]
            )
            if ema_decay is not None:
                self._assert_pure_states_all_close(
                    base_learner_state["ema"].ema, monitored_learner_state["ema"].ema
                )

    @parameterized.product(
        x=[
            jax.random.normal(jax.random.PRNGKey(0), (2, 8, 64)),
            MultiheadAttention.Output(
                data=jax.random.normal(jax.random.PRNGKey(5), (2, 8, 64)),
                probs=None,
                kv_state=None,
            ),
            # transformer with shared kv state
            MultiheadAttention.Output(
                data=jax.random.normal(jax.random.PRNGKey(1), (2, 8, 64)),
                probs=None,
                kv_state=KVState(
                    k_proj=jax.random.normal(jax.random.PRNGKey(2), (2, 8, 2, 16)),
                    v_proj=jax.random.normal(jax.random.PRNGKey(3), (2, 8, 2, 16)),
                    key_positions=jax.random.randint(jax.random.PRNGKey(4), (1, 8), 0, 100),
                    page_indices=None,
                ),
            ),
        ],
        clip_fn=[compute_grad_percentile_no_clip_fn, top_k_clip_fn],
    )
    def test_gradient_clipping_implementation(self, x, clip_fn):
        """Test gradient_clipping_implementation with input types from different layers."""
        prev_stats = jnp.array([1.0, 2.0, 3.0, 4.0])
        clipped, stats = gradient_clipping_impl(x, clip_fn=clip_fn(), prev_stats=prev_stats)

        chex.assert_trees_all_equal_shapes_and_dtypes(clipped, x)
        chex.assert_trees_all_equal_shapes_and_dtypes(stats, prev_stats)

    def test_gradient_clipping_properties(self):
        """This test verifies three key properties:
        1. Forward passes with and without gradient clipping produce the same loss
        2. Jacobians ∂y/∂θ remain the same (clipping only affects gradients in backward)
        3. Parameter gradients differ due to activation gradient clipping
        """
        prng_key = jax.random.PRNGKey(123)
        key1, key2, key3, key4, key5 = jax.random.split(prng_key, 5)

        batch_size, input_dim, model_dim = 4, 6, 3
        inputs = jax.random.normal(key1, (batch_size, input_dim))
        targets = jax.random.normal(key2, (batch_size, model_dim))
        targets = targets.at[-1, :].set(
            targets[-1, :] * 100
        )  # Create outlier tokens at end of batch)

        base_cfg = DummyLayer.default_config().set(name="base", indim=input_dim, outdim=model_dim)
        base_layer = base_cfg.instantiate(parent=None)
        monitored_cfg = convert_to_monitored_layer_config(
            base_cfg,
            GradientMonitorAndClipLayer.default_config().set(
                clip_fn=config_for_function(dummy_clipping)
            ),
        )
        monitored_cfg.set(name="monitored")
        monitored_layer = monitored_cfg.instantiate(parent=None)
        b_params = base_layer.initialize_parameters_recursively(key3)
        m_params = monitored_layer.initialize_parameters_recursively(key3)

        def forward_fn(layer, params, inputs):
            in_dict = dict(x=inputs)
            output, _ = F(layer, prng_key=key4, state=params, inputs=in_dict, is_training=True)
            return output

        def loss_fn(layer, params, inputs, targets):
            output = forward_fn(layer, params, inputs)
            return jnp.mean((output - targets) ** 2)  # Simple MSE loss

        value_and_grad_fn = jax.value_and_grad(loss_fn, argnums=1)
        loss_1, grads_1 = value_and_grad_fn(base_layer, b_params, inputs, targets)
        loss_2, grads_2 = value_and_grad_fn(monitored_layer, m_params, inputs, targets)
        # Check loss parity
        self.assertAlmostEqual(float(loss_1), float(loss_2))

        _, vjp_fn_1 = jax.vjp(lambda p, i: forward_fn(base_layer, p, i), b_params, inputs)
        _, vjp_fn_2 = jax.vjp(lambda p, i: forward_fn(monitored_layer, p, i), m_params, inputs)

        # pre-clip the cotengent so it won't trigger clipping
        v_y, _ = dummy_clipping()(jax.random.normal(key5, targets.shape))
        param_bar_1, inputs_bar_1 = vjp_fn_1(v_y)
        param_bar_2, inputs_bar_2 = vjp_fn_2(v_y)

        # Check Jacobians ∂y/∂θ remains the same
        self.assertNestedAllClose(param_bar_1["weight"], param_bar_2["weight"])
        self.assertNestedAllClose(inputs_bar_1, inputs_bar_2)

        # Check that param gradients are different as a result of clipping
        self.assertLess(
            float(jnp.linalg.norm(grads_2["weight"])), float(jnp.linalg.norm(grads_1["weight"]))
        )
