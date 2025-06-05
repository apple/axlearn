# Copyright © 2023 Apple Inc.

"""Tests vision transformer layers."""
# pylint: disable=no-member,no-self-use
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import (
    Encoder1D,
    VisionTransformer,
    build_vit_model_config,
    named_model_configs,
    sequence_to_space_with_scaling,
)
from axlearn.vision import mask_generator


class ModelTest(TestCase):
    def test_stride(self):
        cfg = build_vit_model_config(
            num_layers=1, model_dim=8, num_heads=4, patch_size=(16, 16), stride=(8, 8)
        )

        model: VisionTransformer = cfg.set(name="test_stride").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)

        F(
            model,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )

    def test_stride_error(self):
        with self.assertRaises(ValueError):
            build_vit_model_config(
                num_layers=1, model_dim=8, num_heads=4, patch_size=(16, 16), stride=(3, 3)
            )

    @parameterized.product(
        is_training=(False, True),
        peak_rate=(None, 0.2),
    )
    def test_vit_features(self, is_training, peak_rate):
        cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            peak_stochastic_depth_rate=peak_rate,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)
        F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )

    @parameterized.product(
        is_training=(False, True),
        drop_token_rate=(0, 0.5, 0.9),
    )
    def test_drop_token_features(self, is_training, drop_token_rate):
        cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
        )
        cfg.encoder_1d.drop_token.rate = drop_token_rate
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)
        test_output, _ = F(
            model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(inputs)),
        )
        ref_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
        )
        ref_model: VisionTransformer = ref_cfg.set(name="test").instantiate(parent=None)
        ref_state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        ref_output, _ = F(
            ref_model,
            is_training=is_training,
            prng_key=jax.random.PRNGKey(123),
            state=ref_state,
            inputs=dict(image=as_tensor(inputs)),
        )
        if is_training is False:
            # If not training, the drop token should not be activated.
            # We test it against a ref model without any drop tokens.
            np.all(ref_output["embedding"] == test_output["embedding"])
            np.all(ref_output["encoded_features"] == test_output["encoded_features"])
            np.all(ref_output["pooled_features"] == test_output["pooled_features"])
        else:
            if drop_token_rate > 0:
                assert np.all(ref_output["embedding"] != test_output["embedding"])

    @parameterized.parameters("cls_token", "gap", "cls_distill_token")
    def test_vit_global_feature_extraction(self, global_feature_extraction):
        cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            global_feature_extraction=global_feature_extraction,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        image = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(image)),
        )

        self.assertIn("pooled_features", outputs)
        self.assertEqual((batch_size, 1, 8), outputs["pooled_features"].shape)
        if global_feature_extraction == "cls_distill_token":
            self.assertIn("distillation_features", outputs)
            self.assertEqual((batch_size, 1, 8), outputs["distillation_features"].shape)
        self.assertEqual((batch_size, 196, 8), outputs["patch_features"].shape)

    def test_model_with_output_proj(self):
        model_cfg = named_model_configs(extra_settings=dict(output_proj_dim=32))[
            "Test16"
        ]  # type: VisionTransformer.Config
        model = model_cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        predictions, _ = F(
            model,
            inputs=dict(image=jnp.zeros((1, 224, 224, 3))),
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
        )
        emb = predictions["embedding"]
        self.assertEqual(emb.shape, (1, 32))
        self.assertNestedAllClose(jnp.linalg.norm(emb, axis=-1), jnp.array([1.0]))

    @parameterized.parameters(0, 118)
    def test_vit_forward_with_is_masked(self, num_masking_patches):
        cfg = build_vit_model_config(
            num_layers=1,
            model_dim=4,
            num_heads=4,
        )
        cfg.use_mask_tokens = True
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        image = np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32)

        mask_model = mask_generator.MaskingGenerator(
            input_size=(14, 14),
            num_masking_patches=num_masking_patches,
        )
        is_masked = mask_model()
        is_masked = jnp.expand_dims(is_masked, axis=0)
        is_masked = jnp.tile(is_masked, [batch_size, 1, 1])
        is_masked = jnp.reshape(is_masked, (batch_size, 196))

        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(image), is_masked=as_tensor(is_masked)),
        )
        # A reference model in which no mask is used.
        ref, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(image), is_masked=None),
        )
        if num_masking_patches == 0:
            np.testing.assert_array_equal(outputs["embedding"], ref["embedding"])
        else:
            np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                outputs["embedding"],
                ref["embedding"],
            )

    @parameterized.parameters(224, 336, 384)
    def test_scaling_inputs_wrt_pos_embedding(self, pretrain_image_size):
        image_size = 384
        patch_size = 16

        cfg = build_vit_model_config(
            num_layers=1,
            model_dim=4,
            num_heads=4,
            image_size=(pretrain_image_size, pretrain_image_size),
            patch_size=(patch_size, patch_size),
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        image = np.random.uniform(-1, 1, [batch_size, image_size, image_size, 3]).astype(np.float32)

        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(image)),
        )
        ref_shape = (batch_size, (image_size // patch_size) ** 2 + 1, 4)
        self.assertEqual(outputs["encoded_features"].shape, ref_shape)
        ref_shape_2d = (batch_size, image_size // patch_size, image_size // patch_size, 4)
        self.assertEqual(outputs["4"].shape, ref_shape_2d)

    @parameterized.product(
        num_cls_tokens=(0, 1),
        target_space_len=(None, 10, 14, 16),
    )
    def test_sequence_to_space_with_scaling(self, num_cls_tokens, target_space_len):
        batch_size = 2
        output_dim = 16
        inputs_len = 14

        inputs = np.random.uniform(
            -1, 1, [batch_size, inputs_len**2 + num_cls_tokens, output_dim]
        ).astype(np.float32)
        outputs = sequence_to_space_with_scaling(
            inputs,
            num_cls_tokens=num_cls_tokens,
            target_len=target_space_len**2 if target_space_len else None,
        )

        if target_space_len and target_space_len != inputs_len:
            ref_seq_shape = (batch_size, target_space_len**2 + num_cls_tokens, output_dim)
            ref_space_shape = (batch_size, target_space_len, target_space_len, output_dim)
        else:
            ref_seq_shape = (batch_size, inputs_len**2 + num_cls_tokens, output_dim)
            ref_space_shape = (batch_size, inputs_len, inputs_len, output_dim)
            np.testing.assert_array_equal(outputs["sequence_features"], inputs)
        self.assertEqual(outputs["sequence_features"].shape, ref_seq_shape)
        self.assertEqual(outputs["space_features"].shape, ref_space_shape)

    @parameterized.product(num_cls_tokens=(0, 1, 2), image_size=(56, 112))
    def test_encoder1d_scaling(self, num_cls_tokens, image_size):
        patch_size = 16
        input_dim = 64

        source_image_size = 56
        pos_emb_seq_len = (source_image_size // patch_size) ** 2 + num_cls_tokens

        cfg = Encoder1D.default_config()
        cfg.input_dim = input_dim
        cfg.num_cls_tokens = num_cls_tokens
        cfg.pos_emb.shape = (pos_emb_seq_len,)
        cfg.transformer.num_layers = 2
        cfg.transformer.layer.self_attention.attention.num_heads = 4

        model: Encoder1D = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        batch_size = 2
        seq_tokens = np.random.uniform(
            -1, 1, [batch_size, image_size // patch_size, image_size // patch_size, input_dim]
        ).astype(np.float32)
        seq_tokens = seq_tokens.reshape(batch_size, -1, input_dim)
        cls_tokens = np.random.uniform(-1, 1, [batch_size, num_cls_tokens, input_dim])

        inputs = np.concatenate((cls_tokens, seq_tokens), axis=1)

        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(inputs=as_tensor(inputs)),
        )

        self.assertEqual(outputs.shape, inputs.shape)

    @parameterized.parameters(True, False)
    def test_use_pos_emb(self, use_pos_emb):
        image_size = 224
        patch_size = 16

        cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            use_pos_emb=use_pos_emb,
        )
        model: VisionTransformer = cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        image = np.random.uniform(-1, 1, [batch_size, image_size, image_size, 3]).astype(np.float32)
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(image=as_tensor(image)),
        )
        ref_shape = (batch_size, (image_size // patch_size) ** 2 + 1, 8)
        self.assertEqual(outputs["encoded_features"].shape, ref_shape)
        ref_shape_2d = (batch_size, image_size // patch_size, image_size // patch_size, 8)
        self.assertEqual(outputs["4"].shape, ref_shape_2d)


if __name__ == "__main__":
    absltest.main()
