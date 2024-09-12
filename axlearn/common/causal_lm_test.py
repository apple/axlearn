# Copyright © 2023 Apple Inc.

"""Tests autoregressive models."""
from functools import partial

import jax
import jax.random
import numpy as np
import pytest
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental.pjit import pjit
from transformers.models.gpt2 import modeling_gpt2 as hf_gpt2

from axlearn.common import causal_lm, utils
from axlearn.common.attention import (
    CausalAttentionLogitBiasLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
)
from axlearn.common.loss import cross_entropy
from axlearn.common.metrics import MetricAccumulator
from axlearn.common.module import (
    InvocationContext,
    functional,
    new_output_collection,
    set_current_context,
)
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import Tensor


class Gpt2TransformerTest(TestCase):
    @parameterized.parameters("attention_mask", "causal_attention")
    def test_against_hf_gpt2_lm(self, causal_attention_mode: str):
        hidden_dim = 16
        vocab_size = 24
        num_heads = 4
        num_layers = 2
        source_length = 11
        # Reference implementation.
        ref_cfg = hf_gpt2.GPT2Config(
            n_embd=hidden_dim,
            n_head=num_heads,
            n_layer=num_layers,
            n_positions=source_length,
            vocab_size=vocab_size,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
        )
        ref_layer = hf_gpt2.GPT2LMHeadModel(ref_cfg).eval()
        # Equivalent AXLearn implementation.
        # The config has similarities with some in encoder_test.py.
        # pylint: disable=duplicate-code
        decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.gelu",
            max_position_embeddings=source_length,
            layer_norm_epsilon=ref_cfg.layer_norm_epsilon,
            dropout_rate=ref_cfg.attn_pdrop,
        )
        if causal_attention_mode == "attention_mask":
            decoder_cfg.transformer.layer.self_attention.attention.causal = False
            decoder_cfg.attention_mask = CausalAttentionLogitBiasLayer.default_config()
            require_same_tree_structure = True
        elif causal_attention_mode == "causal_attention":
            decoder_cfg.transformer.layer.self_attention.attention.causal = True
            decoder_cfg.attention_mask = None
            require_same_tree_structure = False
        else:
            raise ValueError(f"Unknown causal_attention_mode: {causal_attention_mode}")

        decoder_cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    fan=None, scale=0.02, distribution="normal"
                )
            }
        )
        layer = (
            causal_lm.Model.default_config()
            .set(
                decoder=decoder_cfg,
                name="layer_test",
            )
            .instantiate(parent=None)
        )
        input_ids = np.random.randint(1, vocab_size, size=(3, source_length))
        (_, test_aux), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref_layer,
            test_inputs=dict(input_batch=dict(input_ids=input_ids), return_aux=True),
            ref_inputs=as_torch_tensor(input_ids),
            parameters_from_ref_layer=parameters_from_torch_layer,
            require_same_tree_structure=require_same_tree_structure,
        )
        test_logits = test_aux["logits"]
        ref_logits = ref_outputs.logits.detach().numpy()
        assert_allclose(test_logits, ref_logits)


class ModelMetricsTest(TestCase):
    def test_metrics(self):
        decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=1,
            hidden_dim=10,
            num_heads=2,
            vocab_size=10,
            activation_function="nn.gelu",
            max_position_embeddings=10,
            layer_norm_epsilon=0.1,
            dropout_rate=0.0,
        )
        model = (
            causal_lm.Model.default_config()
            .set(
                decoder=decoder_cfg,
                name="metrics_test",
            )
            .instantiate(parent=None)
        )

        prng_key, init_key = jax.random.split(jax.random.PRNGKey(123))
        model_params = model.initialize_parameters_recursively(init_key)
        # Compute summaries after forwarding two batches.
        # The second batch is a dummy one - should not affect metrics.
        target_labels = jnp.array([[[1, 3, 0], [2, 3, 1]], [[0, 0, 0], [0, 0, 0]]])
        logits = jnp.array(
            [
                [
                    [
                        [0.1, 0.9, 0.1, 0.1],  # Target 1; pred 1.
                        [0.1, 0.1, 0.9, 0.1],  # Target 3; pred 2.
                        [0.9, 0.1, 0.1, 0.1],  # Target 0; pred 0.
                    ],  # Example 0.
                    [
                        [0.1, 0.1, 0.9, 0.1],  # Target 2; pred 2.
                        [0.1, 0.1, 0.9, 0.1],  # Target 3; pred 2.
                        [0.9, 0.1, 0.1, 0.1],  # Target 1; pred 0.
                    ],  # Example 1.
                ],  # Batch 0.
                [
                    [
                        [0.1, 0.9, 0.1, 0.1],  # Target 0; pred 1.
                        [0.1, 0.1, 0.9, 0.1],  # Target 0; pred 2.
                        [0.9, 0.1, 0.1, 0.1],  # Target 0; pred 0.
                    ],  # Example 0.
                    [
                        [0.1, 0.1, 0.9, 0.1],  # Target 0; pred 2.
                        [0.1, 0.1, 0.9, 0.1],  # Target 0; pred 2.
                        [0.9, 0.1, 0.1, 0.1],  # Target 0; pred 0.
                    ],  # Example 1.
                ],  # Batch 1.
            ]
        )
        target_num_bytes = jnp.array([[3, 7], [0, 0]])
        live_targets = jnp.array([[[1, 1, 0], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]])
        accumulator = MetricAccumulator.default_config().instantiate()
        for i in range(2):
            _, output_collection = functional(
                model,
                inputs=dict(
                    logits=logits[i],
                    target_labels=target_labels[i],
                    target_num_bytes=target_num_bytes[i],
                ),
                is_training=True,
                prng_key=prng_key,
                state=model_params,
                method="_metrics",
            )
            accumulator.update(output_collection.summaries)
        summaries = accumulator.summaries()
        # Only the first batch should affect results.
        loss, loss_dict = cross_entropy(
            logits=logits[0],
            target_labels=target_labels[0],
            live_targets=live_targets[0],
        )
        self.assertEqual(2.0 / 5, summaries["accuracy"].mean)
        self.assertAlmostEqual(loss, summaries["loss"].mean)
        self.assertEqual(5, summaries["loss"].weight)
        self.assertAlmostEqual(jnp.exp(loss), summaries["perplexity"].mean, places=6)
        per_token_loss = loss_dict["per_target_loss"] * live_targets
        total_bytes = target_num_bytes.sum()
        bits_per_byte = per_token_loss.sum() / jnp.maximum(1, total_bytes) / jnp.log(2)
        self.assertAlmostEqual(bits_per_byte, summaries["bits_per_byte"].mean)

    def test_segment_ids(self):
        batch_size, seq_len, vocab_size = 3, 10, 10

        ref_decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=2,
            hidden_dim=10,
            num_heads=2,
            vocab_size=vocab_size,
            activation_function="nn.gelu",
            max_position_embeddings=seq_len,
            layer_norm_epsilon=0.1,
            dropout_rate=0.0,
        )
        ref_model_cfg = causal_lm.Model.default_config().set(
            decoder=ref_decoder_cfg, name="ref_model"
        )
        ref_model = ref_model_cfg.instantiate(parent=None)

        # Enable attention_mask
        decoder_cfg = ref_decoder_cfg.clone()
        decoder_cfg.transformer.layer.self_attention.attention.causal = False
        decoder_cfg.attention_mask = CausalAttentionLogitBiasLayer.default_config()
        model_cfg = causal_lm.Model.default_config().set(decoder=decoder_cfg, name="model")
        model = model_cfg.instantiate(parent=None)

        prng_key, init_key = jax.random.split(jax.random.PRNGKey(123))

        input_ids = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=0, maxval=vocab_size
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=-1, maxval=vocab_size
        )
        input_batch = dict(
            input_ids=input_ids,
            target_labels=target_labels,
        )

        model_params = ref_model.initialize_parameters_recursively(init_key)

        ctx = InvocationContext(
            name="root",
            parent=None,
            module=ref_model,
            state=model_params,
            output_collection=new_output_collection(),
            is_training=True,
            prng_key=prng_key,
        )
        with set_current_context(ctx):
            ref_loss, _ = ref_model.forward(input_batch=input_batch)

        # Results should be the same.
        segment_ids = jnp.ones(shape=[batch_size, seq_len], dtype=jnp.int32)
        positions = jnp.broadcast_to(jnp.arange(seq_len), shape=[batch_size, seq_len])
        input_batch = dict(
            input_ids=input_ids,
            target_labels=target_labels,
            input_segment_ids=segment_ids,
            input_positions=positions,
        )
        ctx = InvocationContext(
            name="root",
            parent=None,
            module=model,
            state=model_params,
            output_collection=new_output_collection(),
            is_training=True,
            prng_key=prng_key,
        )
        with set_current_context(ctx):
            loss, _ = model.forward(input_batch=input_batch)

        self.assertAlmostEqual(ref_loss, loss)

        # Use a different attention mask.
        segment_ids = jnp.broadcast_to(jnp.arange(seq_len), shape=[batch_size, seq_len])
        positions = jnp.broadcast_to(jnp.arange(seq_len), shape=[batch_size, seq_len])
        input_batch = dict(
            input_ids=input_ids,
            target_labels=target_labels,
            input_segment_ids=segment_ids,
            input_positions=positions,
        )
        ctx = InvocationContext(
            name="root",
            parent=None,
            module=model,
            state=model_params,
            output_collection=new_output_collection(),
            is_training=True,
            prng_key=prng_key,
        )
        with set_current_context(ctx):
            loss, _ = model.forward(input_batch=input_batch)
        self.assertNotAlmostEqual(ref_loss, loss)

    def test_forward(self):
        batch_size, seq_len, vocab_size = 3, 10, 10

        decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=StackedTransformerLayer.default_config(),
            num_layers=2,
            hidden_dim=10,
            num_heads=2,
            vocab_size=vocab_size,
            activation_function="nn.gelu",
            max_position_embeddings=seq_len,
            layer_norm_epsilon=0.1,
            dropout_rate=0.0,
        )
        model_cfg = causal_lm.Model.default_config().set(decoder=decoder_cfg, name="metrics_test")
        model = model_cfg.instantiate(parent=None)

        prng_key, init_key = jax.random.split(jax.random.PRNGKey(123))
        model_params = model.initialize_parameters_recursively(init_key)

        input_ids = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=0, maxval=vocab_size
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=-1, maxval=vocab_size
        )
        input_batch = dict(input_ids=input_ids, target_labels=target_labels)

        # Ensure that forward outputs are consistent with metrics output.
        ctx = InvocationContext(
            name="root",
            parent=None,
            module=model,
            state=model_params,
            output_collection=new_output_collection(),
            is_training=True,
            prng_key=prng_key,
        )
        with set_current_context(ctx):
            loss, aux = model.forward(input_batch=input_batch, return_aux=True)
            # pylint: disable-next=protected-access
            ref_outputs = model._metrics(
                logits=aux["logits"], target_labels=target_labels, target_num_bytes=None
            )
            self.assertAlmostEqual(loss, ref_outputs["loss"])
            self.assertTrue(jnp.allclose(aux["per_label_loss"], ref_outputs["per_token_loss"]))

    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason="Incorrect device & process count for mesh.",
    )
    def test_constrain_input_batch(self):
        model = (
            causal_lm.Model.default_config()
            .set(
                decoder=causal_lm.gpt_decoder_config(
                    stack_cfg=StackedTransformerLayer.default_config(),
                    num_layers=1,
                    hidden_dim=10,
                    num_heads=2,
                    vocab_size=10,
                    activation_function="nn.relu",
                    max_position_embeddings=10,
                    layer_norm_epsilon=0.1,
                    dropout_rate=0.0,
                ),
                batch_axis_names=("data", "expert", "fsdp"),
                seq_axis_names=("seq",),
                name="metrics_test",
            )
            .instantiate(parent=None)
        )
        batch_size = 4
        seq_len = 8
        input_batch = {
            "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "prefix": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "target_num_bytes": jnp.ones((batch_size,), dtype=jnp.int32),
            "extra_variable": jnp.ones((batch_size,), dtype=jnp.int32),
            "input_segment_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
            "input_positions": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        }

        with jax.sharding.Mesh(
            np.array(jax.devices()).reshape(2, 2)[:, :, None, None, None],
            axis_names=("data", "seq", "expert", "fsdp", "model"),
        ):
            # Check that no values are dropped when applying the constraint.
            constrained_input_batch = input_batch.copy()
            # pylint: disable-next=protected-access
            model._constrain_input_batch(constrained_input_batch)
            self.assertNestedEqual(constrained_input_batch, input_batch)

            @partial(pjit, in_shardings=None, out_shardings=None)
            def fn(x):
                # pylint: disable-next=protected-access
                model._constrain_input_batch(x)
                return x

            # Get stable-hlo representation.
            hlo_text = fn.lower(input_batch).compiler_ir(dialect="hlo").as_hlo_text()

            # Five (out of six) tensors were sharded.
            self.assertEqual(hlo_text.count('custom_call_target="Sharding"'), 5)
            # For the [batch, seq_len] tensors.
            self.assertEqual(hlo_text.count("sharding={devices=[2,2]<=[4]}"), 4)
            # For the [batch,] tensor.
            self.assertEqual(
                hlo_text.count("sharding={devices=[2,2]<=[4] last_tile_dim_replicate}"), 1
            )


class DummyFeedForwardWithAuxLoss(TransformerFeedForwardLayer):
    """A dummy FFN with aux loss."""

    def forward(self, inputs: Tensor) -> Tensor:
        self.add_module_output("aux_loss", jnp.array(1.0))
        return inputs


class ModelAuxLossTest(parameterized.TestCase):
    @parameterized.product(
        aux_loss_regex=(None, ".*/aux_loss", ".*/apple"),
        stack_cfg=(
            RepeatedTransformerLayer.default_config(),
            StackedTransformerLayer.default_config(),
        ),
        use_aux_layer=(False, True),
    )
    def test_aux_loss(self, aux_loss_regex, stack_cfg, use_aux_layer):
        batch_size, seq_len, vocab_size = 3, 10, 10
        hidden_dim = 8
        num_layers = 6
        decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=stack_cfg,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_heads=4,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
        )
        if isinstance(decoder_cfg.transformer, RepeatedTransformerLayer.Config):
            decoder_cfg.transformer.repeat.drop_output = None
        if use_aux_layer:
            decoder_cfg.transformer.layer.feed_forward = (
                DummyFeedForwardWithAuxLoss.default_config().set(hidden_dim=4 * hidden_dim)
            )
        model_cfg = causal_lm.Model.default_config().set(
            decoder=decoder_cfg, name="metrics_test", aux_loss_regex=aux_loss_regex
        )
        model = model_cfg.instantiate(parent=None)
        prng_key, init_key = jax.random.split(jax.random.PRNGKey(123))
        model_params = model.initialize_parameters_recursively(init_key)

        input_ids = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=0, maxval=vocab_size
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=-1, maxval=vocab_size
        )
        input_batch = dict(input_ids=input_ids, target_labels=target_labels)

        # Ensure that forward outputs are consistent with metrics output.
        ctx = InvocationContext(
            name="root",
            parent=None,
            module=model,
            state=model_params,
            output_collection=new_output_collection(),
            is_training=True,
            prng_key=prng_key,
        )
        with set_current_context(ctx):
            loss, aux = model.forward(input_batch=input_batch, return_aux=True)
            # pylint: disable-next=protected-access
            ref = model._metrics(
                logits=aux["logits"], target_labels=target_labels, target_num_bytes=None
            )
            # `aux_loss` is only collected when `aux_loss_regex` is set.
            if aux_loss_regex is not None:
                self.assertIn("aux_loss", aux)
                if aux_loss_regex == ".*/aux_loss" and use_aux_layer:
                    self.assertEqual(aux["aux_loss"], 1.0)
                else:
                    self.assertEqual(aux["aux_loss"], 0.0)
                self.assertEqual(ref["cross_entropy"] + aux["aux_loss"], loss)
            else:
                self.assertNotIn("aux_loss", aux)
                self.assertEqual(ref["cross_entropy"], loss)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
