# Copyright © 2023 Apple Inc.

"""Tests T5 layers."""

import os

# pylint: disable=no-self-use
from typing import Dict

import jax
import numpy as np
import pytest
import torch
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from transformers.models.t5 import modeling_t5 as hf_t5

from axlearn.common import t5, utils
from axlearn.common.attention import TransformerLayer
from axlearn.common.layers import RMSNorm
from axlearn.common.module import functional as F
from axlearn.common.t5 import (
    T5Encoder,
    T5EncoderDecoderModel,
    t5_encoder_config,
    t5_encoder_decoder_config,
    t5_transformer_stack_config,
)
from axlearn.common.test_utils import TestCase
from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.common.utils import Tensor, as_tensor

testdata_dir = os.path.join(os.path.dirname(__file__), "../experiments/testdata")


def random_inputs_for_t5(
    *,
    source_length: int,
    target_length: int,
    source_vocab_size: int,
    target_vocab_size: int,
    batch_size: int = 2,
) -> Dict[str, jax.Array]:
    """Generate random inputs for AXLearn T5.

    Args:
        source_length: Encoder input length.
        target_length: Decoder input length.
        source_vocab_size: Encoder vocab size.
        target_vocab_size: Decoder vocab size.
        batch_size: Batch size.

    Returns:
        A dict containing:
            source: A dict containing:
                input_ids: An int Tensor of shape [batch_size, source_length].
            target: A dict containing:
                input_ids: An int Tensor of shape [batch_size, target_length].
            target_labels: An int Tensor of shape [batch_size, target_length].
    """
    source_ids = jax.random.randint(
        jax.random.PRNGKey(123),
        [batch_size, source_length],
        minval=2,
        maxval=source_vocab_size,
    )
    source_lengths = jax.random.randint(
        jax.random.PRNGKey(100),
        [batch_size],
        minval=1,
        maxval=source_length,
    )
    # Set source_ids[i, j] = 0 if j >= source_lengths[i].
    source_paddings = jnp.arange(source_length)[None, :] >= source_lengths[:, None]
    source_ids *= 1 - source_paddings
    target_labels = jax.random.randint(
        jax.random.PRNGKey(456),
        [batch_size, target_length + 1],
        minval=2,
        maxval=target_vocab_size,
    )
    target_ids = target_labels[:, :-1]
    target_labels = target_labels[:, 1:]
    target_lengths = jax.random.randint(
        jax.random.PRNGKey(200),
        [batch_size],
        minval=target_length // 2,
        maxval=target_length,
    )
    target_paddings = jnp.arange(target_length)[None, :] >= target_lengths[:, None]
    target_ids *= 1 - target_paddings
    # Set target_labels to -1 for padding.
    target_labels = target_labels * (1 - target_paddings) - target_paddings
    return dict(
        source=dict(input_ids=source_ids),
        target=dict(input_ids=target_ids),
        target_labels=target_labels,
    )


def prepare_hf_t5_inputs(
    source_ids: Tensor, target_ids: Tensor, target_labels: Tensor, pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    input_ids = torch.as_tensor(np.asarray(source_ids).copy(), dtype=torch.int32)
    attention_mask = (input_ids != pad_token_id).int()
    labels = torch.as_tensor(np.asarray(target_labels).copy(), dtype=torch.long)
    decoder_input_ids = torch.as_tensor(np.asarray(target_ids).copy(), dtype=torch.int32)
    labels[labels <= 0] = -100
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )


class RelativePositionTest(TestCase):
    def test_mtf_buckets(self):
        seq_len = 20
        # When number of buckets are limited, multiple relative positions share the same bucket.
        # fmt: off
        np.testing.assert_array_equal(
            [
                7, 7, 7, 7, 7, 7, 7,
                6, 6, 6, 6, 6,
                5, 5, 5,
                4, 4,
                3,
                2,
                1,
                0,
                9,
                10,
                11,
                12, 12,
                13, 13, 13,
                14, 14, 14, 14, 14,
                15, 15, 15, 15, 15, 15, 15,
            ],
            t5.t5_relative_position_bucket(
                jnp.arange(-seq_len, seq_len + 1, dtype=jnp.int32),
                num_buckets=16,
                max_distance=seq_len,
            ),
        )
        # When max_distance is limited, relative distances with magnitude >= max_distance share two
        # buckets.
        np.testing.assert_array_equal(
            [
                13, 13, 13, 13, 13, 13, 13,
                12,
                11, 11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1,
                0,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25, 25,
                26,
                27, 27, 27, 27, 27, 27, 27,
            ],
            t5.t5_relative_position_bucket(
                jnp.arange(-seq_len, seq_len + 1, dtype=jnp.int32),
                num_buckets=28,
                max_distance=15,
            ),
        )
        # fmt: on

    @parameterized.product(bidirectional=[True, False], seq_len=[20, 100, 256])
    def test_buckets_against_t5x(self, bidirectional, seq_len):
        testcase = jnp.load(
            os.path.join(
                testdata_dir, __name__, f"test_buckets_against_t5x_{bidirectional}_{seq_len}.npy"
            ),
            allow_pickle=True,
        ).item()
        actual = t5.t5_relative_position_bucket(testcase["inputs"], bidirectional=bidirectional)
        self.assertNestedAllClose(testcase["outputs"], actual)


class T5EncoderTest(TestCase):
    def test_against_t5_encoder(self):
        vocab_size = 128
        model_dim = 12
        num_layers = 2
        num_heads = 3
        cfg: T5Encoder.Config = t5_encoder_config(
            vocab_size=vocab_size,
            dim=model_dim,
            num_attention_heads=num_heads,
            num_layers=num_layers,
        ).set(name="test", vlog=5)
        layer_cfg: TransformerLayer.Config = cfg.transformer.layer
        layer_cfg.vlog = 5
        layer_cfg.self_attention.vlog = 5
        layer_cfg.self_attention.attention.vlog = 5
        layer_cfg.feed_forward.vlog = 5
        self.assertFalse(layer_cfg.self_attention.attention.input_linear.layer.bias)
        self.assertFalse(layer_cfg.self_attention.attention.output_linear.bias)
        layer: T5Encoder = cfg.instantiate(parent=None)
        t5_config = hf_t5.T5Config(
            vocab_size=vocab_size,
            d_model=model_dim,
            d_kv=model_dim // num_heads,
            d_ff=model_dim * 8 // 3,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=0,
            layer_norm_epsilon=RMSNorm.default_config().eps,
            feed_forward_proj="gated-gelu",
            use_cache=False,
        )
        ref = hf_t5.T5EncoderModel(t5_config)

        input_ids = [
            [1, 2, 3, 4, 5, 0],
            [1, 7, 8, 9, 0, 0],
        ]
        logging.info("input_ids=%s", input_ids)
        is_padding = jnp.expand_dims(jnp.asarray(input_ids) == 0, -1).astype(jnp.float32)
        test_outputs, ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref,
            test_inputs=dict(input_ids=jnp.asarray(input_ids)),
            ref_inputs=dict(
                input_ids=torch.as_tensor(input_ids, dtype=torch.int32),
                attention_mask=(torch.as_tensor(input_ids) != 0).int(),
            ),
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        self.assertNestedAllClose(
            test_outputs * (1 - is_padding),
            as_tensor(ref_outputs["last_hidden_state"]) * (1 - is_padding),
            atol=5e-6,
        )


class T5EncoderDecoderModelTest(TestCase):
    @parameterized.parameters(["t5-v1-1", "t5-v1", "t5-ul2"])
    def test_against_t5_encoder_decoder_model(self, arch):
        vocab_size = 128
        model_dim = 12
        num_layers = 2
        num_heads = 3
        cfg: T5EncoderDecoderModel.Config = t5_encoder_decoder_config(
            vocab_size=vocab_size,
            dim=model_dim,
            num_attention_heads=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout_rate=0,
            z_loss_scale=0,
            stack_cfg=t5_transformer_stack_config(arch=arch),
        ).set(name="test", vlog=5)
        layer_cfg: TransformerLayer.Config = cfg.decoder.transformer.layer
        layer_cfg.vlog = 5
        layer_cfg.self_attention.vlog = 5
        layer_cfg.self_attention.attention.vlog = 5
        layer_cfg.feed_forward.vlog = 5
        cfg.decoder.relative_pos_emb.vlog = 5
        layer: T5EncoderDecoderModel = cfg.instantiate(parent=None)
        if arch == "t5-v1":
            d_ff = 4 * model_dim
            feed_forward_proj = "relu"
        elif arch == "t5-v1-1":
            d_ff = model_dim * 8 // 3
            feed_forward_proj = "gated-gelu"
        elif arch == "t5-ul2":
            d_ff = model_dim * 8 // 3
            # Ref: https://huggingface.co/google/ul2/blob/main/config.json#L13.
            feed_forward_proj = "gated-silu"
        else:
            raise ValueError(f"unsupported t5 arch {arch}")
        t5_config = hf_t5.T5Config(
            vocab_size=vocab_size,
            d_model=model_dim,
            d_kv=model_dim // num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=0,
            feed_forward_proj=feed_forward_proj,
            use_cache=False,
            tie_word_embeddings=False,
        )
        ref = hf_t5.T5ForConditionalGeneration(t5_config)
        test_inputs = random_inputs_for_t5(
            source_length=64,
            target_length=64,
            source_vocab_size=vocab_size,
            target_vocab_size=vocab_size,
        )
        ref_inputs = prepare_hf_t5_inputs(
            source_ids=test_inputs["source"]["input_ids"],
            target_ids=test_inputs["target"]["input_ids"],
            target_labels=test_inputs["target_labels"],
        )
        (test_loss, test_aux), ref_outputs = self._compute_layer_outputs(
            test_layer=layer,
            ref_layer=ref,
            test_inputs=dict(
                input_batch=test_inputs,
                return_aux=True,
            ),
            ref_inputs=ref_inputs,
            parameters_from_ref_layer=parameters_from_torch_layer,
        )
        # Ignore out-of-class labels, as well as padding (0) labels.
        self.assertEqual(cfg.encoder.pad_token_id, 0)
        self.assertEqual(cfg.decoder.pad_token_id, 0)
        target_label_mask = jnp.logical_and(
            0 < test_inputs["target_labels"],
            test_inputs["target_labels"] < vocab_size,
        )
        self.assertNestedAllClose(
            test_aux["logits"] * target_label_mask[:, :, None],
            as_tensor(ref_outputs.logits) * target_label_mask[:, :, None],
            atol=1e-4,
        )
        self.assertNestedAllClose(test_loss, as_tensor(ref_outputs.loss))

    def test_pjit(self):
        # A simple test to ensure a train step does not leak tracers.
        mesh_shape = (1, 1)
        mesh_axes = ("data", "model")
        devices = mesh_utils.create_device_mesh(mesh_shape)
        with jax.checking_leaks(), jax.sharding.Mesh(devices, mesh_axes):
            vocab_size = 6
            cfg: T5EncoderDecoderModel.Config = t5_encoder_decoder_config(
                vocab_size=vocab_size,
                dim=4,
                num_attention_heads=2,
                num_encoder_layers=2,
                num_decoder_layers=2,
                dropout_rate=0,
                z_loss_scale=0,
            ).set(name="test", vlog=5)
            layer: T5EncoderDecoderModel = cfg.instantiate(parent=None)

            def train_step(state, input_batch):
                input_batch = utils.dispatch_input_batch(input_batch)
                new_prng_key, prng_key = jax.random.split(state["prng_key"])

                def _forward(model_parameters, forward_input_batch):
                    (loss, aux), model_output_collection = F(
                        layer,
                        state=model_parameters,
                        is_training=True,
                        prng_key=prng_key,
                        inputs=dict(input_batch=forward_input_batch),
                    )
                    return loss, (aux, model_output_collection)

                forward_and_grad = jax.value_and_grad(_forward, has_aux=True)
                (loss, _), grads = forward_and_grad(state["model"], input_batch)

                logging.info("loss=%s", loss)
                logging.info("grads=%s", grads)

                return dict(
                    prng_key=new_prng_key,
                    model=jax.tree_map(lambda x, y: x + y, state["model"], grads),
                )

            state_partition_specs = dict(
                prng_key=None,
                model=jax.tree_map(
                    lambda spec: spec.mesh_axes,
                    layer.create_parameter_specs_recursively(),
                ),
            )
            jit_train_step = pjit(
                train_step,
                in_shardings=(state_partition_specs, utils.input_partition_spec()),
                out_shardings=state_partition_specs,
            )
            state = init_state = {
                "prng_key": jax.random.PRNGKey(1),
                "model": layer.initialize_parameters_recursively(jax.random.PRNGKey(2)),
            }
            logging.info("init_state=%s", state["model"]["shared_token_emb"])
            for _ in range(3):
                input_batch = random_inputs_for_t5(
                    source_length=4,
                    target_length=4,
                    source_vocab_size=vocab_size,
                    target_vocab_size=vocab_size,
                )
                state = jit_train_step(state, input_batch)
            logging.info("final_state=%s", state["model"]["shared_token_emb"])

            # Ensure that init and final states do not match.
            with self.assertRaises(AssertionError):
                self.assertNestedAllClose(init_state, state)


class OneHotGatherTest(TestCase):
    """Tests one-hot gather against all-gather.

    This is relevant when comparing against T5X as it uses one-hot gather for embeddings.
    On TPU, we need the precision to be set to float32 in order for it to match the all-gather
    used in AXLearn.

    Ref: https://github.com/google-research/t5x/blob/main/t5x/examples/t5/layers.py#L496
    Explanation: https://github.com/google/jax/issues/7010
    """

    @parameterized.parameters([10, 100, 1000, 10000])
    # TODO(markblee): Re-enable in CI when we have access to a larger instance.
    @pytest.mark.high_cpu
    def test_one_hot_against_all_gather(self, size):
        batch_size = 8

        prng = jax.random.PRNGKey(111)
        arr = jax.random.normal(prng, (size, size))
        indices = jax.random.randint(prng, (batch_size, size // 4), 0, size)

        iota = jax.lax.iota(jnp.int32, size)
        one_hot = jnp.array(indices[..., jnp.newaxis] == iota, dtype=jnp.float32)

        expected = arr[indices]
        with jax.default_matmul_precision("float32"):
            actual = jnp.dot(one_hot, jnp.asarray(arr, jnp.float32))

        self.assertNestedAllClose(expected, actual)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
