# Copyright © 2023 Apple Inc.

"""Tests autoregressive models."""

from functools import partial
from typing import cast

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
    BaseStackedTransformerLayer,
    CausalAttentionLogitBiasLayer,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerFeedForwardLayer,
)
from axlearn.common.config import config_for_function
from axlearn.common.learner import Learner
from axlearn.common.loss import cross_entropy
from axlearn.common.loss_metrics import BaseLossMetrics
from axlearn.common.metrics import MetricAccumulator, WeightedScalar
from axlearn.common.module import (
    InvocationContext,
    OutputCollection,
    child_context,
    functional,
    new_output_collection,
    set_current_context,
)
from axlearn.common.optimizer_base import OptParam
from axlearn.common.optimizers import sgd_optimizer
from axlearn.common.param_converter import as_torch_tensor
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.update_transformation import ForwardBackwardOutputs, ForwardOutputs
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


class ModelTest(TestCase):
    def _model_config(self, vocab_size: int, seq_len: int) -> causal_lm.Model.Config:
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
        return causal_lm.Model.default_config().set(decoder=decoder_cfg)

    def test_metrics(self):
        model = (
            self._model_config(vocab_size=10, seq_len=10)
            .set(name="metrics_test")
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
                    input_batch=dict(
                        target_labels=target_labels[i],
                        target_num_bytes=target_num_bytes[i],
                    ),
                    predict_outputs=dict(
                        logits=logits[i],
                    ),
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

        ref_model_cfg = self._model_config(vocab_size=vocab_size, seq_len=seq_len)
        ref_decoder_cfg = ref_model_cfg.decoder
        ref_model = ref_model_cfg.set(name="ref_model").instantiate(parent=None)

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

        model_cfg = self._model_config(vocab_size=vocab_size, seq_len=seq_len)
        model = model_cfg.set(name="metrics_test").instantiate(parent=None)

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
        common_kwargs = dict(module=model, prng_key=prng_key, state=model_params, is_training=True)
        (loss, aux), _ = functional(
            **common_kwargs,
            inputs=dict(input_batch=input_batch, return_aux=True),
        )
        (ref_loss, metrics), _ = functional(
            **common_kwargs,
            inputs=dict(
                input_batch=dict(target_labels=target_labels),
                predict_outputs=dict(logits=aux["logits"]),
            ),
            method="_metrics",
        )
        self.assertAlmostEqual(loss, ref_loss)
        self.assertNestedAllClose(aux["metrics"], metrics)

        # Check against score.
        score_metrics, _ = functional(
            **common_kwargs, inputs=dict(input_batch=input_batch), method="score"
        )
        for k, v in score_metrics.items():
            self.assertNestedAllClose(metrics[k], v)

    def test_metrics_update(self):
        # pylint: disable=unused-argument

        class DummyMetrics(BaseLossMetrics):
            def forward(self, *args, **kwargs):
                self.add_summary(f"{self.name}_summary", 0)
                self.add_module_output(f"{self.name}_output", 0)
                self.add_state_update(f"{self.name}_state", 0)
                return 0, {f"{self.name}_output": 0}

        class DummyConflictModel(causal_lm.Model):
            def _metrics(self, *args, **kwargs):
                self.add_summary("metrics_summary", 1)
                return super()._metrics(*args, **kwargs)

        def forward(model_cfg: causal_lm.Model.Config, metrics_cfg: BaseLossMetrics.Config):
            batch_size, vocab_size, seq_len = 3, 10, 10
            base_cfg = self._model_config(vocab_size=vocab_size, seq_len=seq_len)
            model_cfg = model_cfg.set(**{k: v for k, v in base_cfg.items() if k not in ("klass",)})
            model_cfg.metrics = metrics_cfg
            model = model_cfg.set(name="test").instantiate(parent=None)
            init_key, forward_key, target_key = jax.random.split(jax.random.PRNGKey(0), num=3)
            target_labels = jax.random.randint(
                target_key, shape=[batch_size, seq_len], minval=-1, maxval=vocab_size
            )
            return functional(
                module=model,
                prng_key=forward_key,
                state=model.initialize_parameters_recursively(init_key),
                inputs=dict(input_batch=dict(target_labels=target_labels), predict_outputs={}),
                method="_metrics",
                is_training=True,
                drop_output_collections=(),
            )

        # Check that flattening summaries do not override base model summaries.
        with self.assertRaisesRegex(KeyError, "Key conflict"):
            forward(DummyConflictModel.default_config(), DummyMetrics.default_config())

        class DummyModel(causal_lm.Model):
            def _metrics(self, *args, **kwargs):
                self.add_summary("parent_summary", 1)
                self.add_module_output("parent_output", 1)
                self.add_state_update("parent_state", 1)
                return super()._metrics(*args, **kwargs)

        def test_no_conflict(
            metrics_cfg: BaseLossMetrics.Config,
            expected_oc: OutputCollection,
            expected_metrics: dict,
        ):
            (_, metrics), output_collection = forward(DummyModel.default_config(), metrics_cfg)
            self.assertNestedEqual(output_collection.summaries, expected_oc.summaries)
            self.assertNestedEqual(output_collection.module_outputs, expected_oc.module_outputs)
            self.assertNestedEqual(output_collection.state_updates, expected_oc.state_updates)
            self.assertNestedEqual(metrics, expected_metrics)

        test_no_conflict(
            DummyMetrics.default_config(),
            OutputCollection(
                summaries={"parent_summary": 1, "metrics_summary": 0},
                module_outputs={"parent_output": 1, "metrics": {"metrics_output": 0}},
                state_updates={"parent_state": 1, "metrics": {"metrics_state": 0}},
            ),
            {"metrics_output": 0},
        )
        for flatten_metrics in (None, True):
            test_no_conflict(
                causal_lm.CompositeLossMetrics.default_config().set(
                    metrics={
                        "child1": DummyMetrics.default_config(),
                        "child2": DummyMetrics.default_config(),
                    },
                    flatten_metrics=flatten_metrics,
                ),
                OutputCollection(
                    summaries={"parent_summary": 1, "child1_summary": 0, "child2_summary": 0},
                    module_outputs={
                        "parent_output": 1,
                        "metrics": {"child1": {"child1_output": 0}, "child2": {"child2_output": 0}},
                    },
                    state_updates={
                        "parent_state": 1,
                        "metrics": {"child1": {"child1_state": 0}, "child2": {"child2_state": 0}},
                    },
                ),
                {"child1_output": 0, "child2_output": 0},
            )

        # Test without flattening.
        test_no_conflict(
            causal_lm.CompositeLossMetrics.default_config().set(
                metrics={
                    "child1": DummyMetrics.default_config(),
                    "child2": DummyMetrics.default_config(),
                },
                flatten_metrics=False,
            ),
            OutputCollection(
                summaries={
                    "parent_summary": 1,
                    "child1": {"child1_summary": 0},
                    "child2": {"child2_summary": 0},
                },
                module_outputs={
                    "parent_output": 1,
                    "metrics": {"child1": {"child1_output": 0}, "child2": {"child2_output": 0}},
                },
                state_updates={
                    "parent_state": 1,
                    "metrics": {"child1": {"child1_state": 0}, "child2": {"child2_state": 0}},
                },
            ),
            {"child1": {"child1_output": 0}, "child2": {"child2_output": 0}},
        )

    # TODO(markblee): Add a pytest marker for multi-device tests.
    @pytest.mark.skipif(
        jax.device_count() != 4 or jax.process_count() != 1,
        reason=(
            "Incorrect device & process count for mesh.\n"
            "Use XLA_FLAGS=--xla_force_host_platform_device_count=4 to run locally."
        ),
    )
    def test_constrain_input_batch(self):
        model = (
            self._model_config(vocab_size=10, seq_len=10)
            .set(
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

            # Seven (out of eight) tensors were sharded.
            self.assertEqual(hlo_text.count('custom_call_target="Sharding"'), 7)
            # For the [batch, seq_len] tensors.
            self.assertEqual(hlo_text.count("sharding={devices=[2,2]<=[4]}"), 6)
            # For the [batch,] tensor.
            self.assertEqual(
                hlo_text.count("sharding={devices=[2,2]<=[4] last_tile_dim_replicate}"), 1
            )


class CrossEntropyLossMetricsTest(TestCase):
    """Tests CrossEntropyLossMetrics."""

    def test_live_targets(self):
        batch_size, seq_len, vocab_size = 3, 10, 10
        tgt_key, logit_key, live_tgt_key = jax.random.split(jax.random.PRNGKey(0), num=3)
        target_labels = jax.random.randint(
            tgt_key, shape=[batch_size, seq_len], minval=-1, maxval=vocab_size
        )
        logits = jax.random.uniform(logit_key, shape=[*target_labels.shape, vocab_size])
        layer = (
            causal_lm.CrossEntropyLossMetrics.default_config()
            .set(name="test")
            .instantiate(parent=None)
        )

        # Make sure at least one masked target.
        assert jnp.any(target_labels == -1), target_labels

        def forward(live_targets):
            (loss, metrics), _ = functional(
                layer,
                prng_key=None,
                state={},
                inputs=dict(
                    input_batch=dict(target_labels=target_labels, live_targets=live_targets),
                    predict_outputs=dict(logits=logits),
                    module_outputs={},
                ),
                is_training=True,
            )
            return loss, metrics

        # Test without live_targets. Should be equivalent to target_labels >= 0.
        test_loss, metrics = forward(live_targets=None)
        ref_loss, _ = cross_entropy(logits, target_labels, live_targets=target_labels >= 0)
        self.assertAlmostEqual(test_loss, ref_loss)
        self.assertEqual(metrics["num_targets"], (target_labels >= 0).sum())

        # Test with live_targets.
        live_targets = jax.random.randint(
            live_tgt_key, shape=target_labels.shape, minval=0, maxval=2
        )
        test_loss, metrics = forward(live_targets=live_targets)
        ref_loss, _ = cross_entropy(logits, target_labels, live_targets=live_targets)
        self.assertAlmostEqual(test_loss, ref_loss)
        self.assertEqual(metrics["num_targets"], live_targets.sum())


class CompositeLossMetricsTest(TestCase):
    """Tests CompositeLossMetrics."""

    def test_loss_weights(self):
        class DummyMetrics(BaseLossMetrics):
            def forward(self, input_batch, **kwargs):
                del kwargs
                return input_batch[self.name], {}

        class FixedLossWeights(causal_lm.CompositeLossWeights):
            def forward(self, child_metrics):
                del child_metrics
                return {"test0": 0.5, "test1": 1.0}

        cfg = causal_lm.CompositeLossMetrics.default_config().set(
            name="test",
            metrics={
                "test0": DummyMetrics.default_config(),
                "test1": DummyMetrics.default_config(),
            },
            loss_weights=FixedLossWeights.default_config(),
        )

        metrics = cfg.instantiate(parent=None)

        (loss, _), _ = functional(
            metrics,
            prng_key=jax.random.PRNGKey(123),
            state={},
            inputs=dict(
                input_batch={"test0": 1.23, "test1": 3.45}, predict_outputs={}, module_outputs={}
            ),
            is_training=True,
        )
        self.assertAlmostEqual(loss, 1.23 * 0.5 + 3.45)


class DummyFeedForwardWithAuxLoss(TransformerFeedForwardLayer):
    """A dummy FFN with aux loss."""

    def forward(self, inputs: Tensor) -> Tensor:
        self.add_module_output("aux_loss", jnp.array(1.0))
        return inputs


class ModelAuxLossTest(TestCase):
    def _model_config(
        self,
        *,
        stack_cfg: BaseStackedTransformerLayer.Config,
        hidden_dim: int,
        vocab_size: int,
        seq_len: int,
        aux_loss_regex: str,
        use_aux_layer: bool,
    ) -> causal_lm.Model.Config:
        decoder_cfg = causal_lm.gpt_decoder_config(
            stack_cfg=stack_cfg,
            num_layers=6,
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
        return causal_lm.Model.default_config().set(
            decoder=decoder_cfg,
            name="metrics_test",
            metrics=causal_lm.metrics_config(aux_loss_regex=aux_loss_regex),
        )

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
        model_cfg = self._model_config(
            stack_cfg=stack_cfg,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            seq_len=seq_len,
            aux_loss_regex=aux_loss_regex,
            use_aux_layer=use_aux_layer,
        )
        model: causal_lm.Model = model_cfg.instantiate(parent=None)
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
        common_kwargs = dict(module=model, prng_key=prng_key, state=model_params, is_training=True)
        (loss, aux), output_collection = functional(
            **common_kwargs,
            inputs=dict(input_batch=input_batch, return_aux=True),
            drop_output_collections=(),
        )
        oc = new_output_collection()
        oc.module_outputs.update(output_collection.module_outputs)
        metrics_fn = InvocationContext(
            name="metrics", parent=None, output_collection=oc, **common_kwargs
        ).functional(
            model._metrics  # pylint: disable=protected-access
        )
        (ref_loss, metrics), _ = metrics_fn(
            input_batch=dict(target_labels=target_labels),
            predict_outputs=dict(logits=aux["logits"]),
        )
        self.assertAlmostEqual(loss, ref_loss)
        self.assertNestedAllClose(aux["metrics"], metrics)

        # `aux_loss` is only collected when `aux_loss_regex` is set.
        if aux_loss_regex is not None:
            self.assertIn("aux_loss", aux["metrics"])
            if aux_loss_regex == ".*/aux_loss" and use_aux_layer:
                self.assertEqual(aux["metrics"]["aux_loss"], 1.0)
            else:
                self.assertEqual(aux["metrics"]["aux_loss"], 0.0)
            self.assertEqual(aux["metrics"]["cross_entropy"] + aux["metrics"]["aux_loss"], loss)
        else:
            self.assertNotIn("aux_loss", aux)
            self.assertEqual(aux["metrics"]["cross_entropy"], loss)

    @parameterized.product(
        stack_cfg=(
            RepeatedTransformerLayer.default_config(),
            StackedTransformerLayer.default_config(),
        ),
    )
    def test_aux_loss_learner(self, stack_cfg):
        batch_size, seq_len, vocab_size = 3, 10, 10
        hidden_dim = 8
        model_cfg = self._model_config(
            stack_cfg=stack_cfg,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
            seq_len=seq_len,
            aux_loss_regex=".*/aux_loss",
            use_aux_layer=True,
        )
        model: causal_lm.Model = model_cfg.set(name="model").instantiate(parent=None)
        learner_cfg = Learner.default_config().set(
            optimizer=config_for_function(sgd_optimizer).set(
                learning_rate=0.1, decouple_weight_decay=True, weight_decay=1.0
            )
        )
        learner = learner_cfg.set(name="learner").instantiate(parent=None)
        init_key, forward_key = jax.random.split(jax.random.PRNGKey(123), num=2)

        model_cfg = causal_lm.Model.default_config()
        params = model.initialize_parameters_recursively(init_key)
        opt_params = jax.tree.map(
            lambda v: OptParam(value=v, factorization_spec=None, weight_decay_scale=None), params
        )
        state = learner.init(model_params=opt_params)

        input_ids = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=0, maxval=vocab_size
        )
        target_labels = jax.random.randint(
            jax.random.PRNGKey(123), shape=[batch_size, seq_len], minval=-1, maxval=vocab_size
        )
        input_batch = dict(input_ids=input_ids, target_labels=target_labels)

        def loss_fn(model_params, inputs):
            model_output_collection = new_output_collection()
            with child_context(
                "model",
                module=model,
                state=model_params,
                prng_key=inputs["forward_key"],
                output_collection=model_output_collection,
            ):
                loss, aux = model(input_batch=inputs["input_batch"])
            return ForwardOutputs(loss=loss, aux=aux, output_collection=model_output_collection)

        inputs = dict(
            fn=loss_fn,
            inputs=dict(forward_key=forward_key, input_batch=input_batch),
            opt_params=opt_params,
        )
        outputs, _ = functional(
            learner,
            method="forward_and_backward",
            is_training=True,
            prng_key=forward_key,
            state=state,
            inputs=inputs,
        )
        outputs = cast(ForwardBackwardOutputs, outputs)
        output_collection: OutputCollection = outputs.forward_outputs.output_collection
        summaries: dict[str, WeightedScalar] = output_collection.summaries
        self.assertIn("aux_loss", summaries)
        self.assertEqual(summaries["aux_loss"].mean, 1.0)
        self.assertEqual(
            summaries["cross_entropy_loss"].mean + summaries["aux_loss"].mean,
            outputs.forward_outputs.loss,
        )


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
