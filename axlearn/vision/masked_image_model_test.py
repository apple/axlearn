# Copyright © 2023 Apple Inc.

"""Tests masked image modeling."""
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.loss import cross_entropy, negative_cosine_similarity_loss
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision import beit_image_tokenizer, feature_tokenizer, mask_generator
from axlearn.vision.clip import set_vision_encoder_config
from axlearn.vision.masked_image_model import MaskedImageModel


class FakeTokenizer(BaseLayer):
    """A dummy tokenizer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures FakeTokenizer."""

        seq_len: Required[int] = REQUIRED
        codebook_size: Required[int] = REQUIRED

    def forward(self, image):
        prng_key = jax.random.PRNGKey(0)
        quantized_codebook_ids = jax.random.randint(
            prng_key,
            shape=[image.shape[0], self.config.seq_len],
            minval=0,
            maxval=self.config.codebook_size,
            dtype=np.int32,
        )
        return quantized_codebook_ids, {}


class ModelTest(TestCase):
    """Tests MaskedImageModel."""

    @parameterized.product(
        tokenizer=("fake", "beitv2", "clip"),
    )
    def test_masked_image_model_forward(self, tokenizer):
        batch_size = 2
        image_size = 48
        patch_size = 16
        output_dim = 4
        codebook_size = 100

        seq_len = (image_size // patch_size) ** 2
        if tokenizer == "fake":
            tokenizer_cfg = FakeTokenizer.default_config().set(
                codebook_size=codebook_size,
                seq_len=seq_len,
            )
            loss_fn = cross_entropy
            head_output_dim = codebook_size
            expected_label_shape = (batch_size, seq_len)
        elif tokenizer == "beitv2":
            tokenizer_cfg = beit_image_tokenizer.set_beit_image_tokenizer_encoder_config(
                num_layers=1,
                model_dim=output_dim,
                codebook_size=codebook_size,
                codebook_dim=8,
                num_heads=4,
                image_size=(image_size, image_size),
                patch_size=(patch_size, patch_size),
            )
            loss_fn = cross_entropy
            head_output_dim = codebook_size
            expected_label_shape = (batch_size, seq_len)
        elif tokenizer == "clip":
            tokenizer_cfg = set_vision_encoder_config(
                num_layers=1,
                model_dim=output_dim,
                num_heads=4,
                projection_dim=output_dim,
                encoder_cls=feature_tokenizer.CLIPFeatureTokenizer,
            )
            loss_fn = negative_cosine_similarity_loss
            head_output_dim = output_dim
            expected_label_shape = (batch_size, seq_len, output_dim)
        else:
            raise ValueError(f"Tokenizer {tokenizer} not implemented.")

        encoder_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=output_dim,
            num_heads=4,
            image_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
        )
        encoder_cfg.use_mask_tokens = True
        model_cfg = MaskedImageModel.default_config().set(
            tokenizer=tokenizer_cfg,
            encoder=encoder_cfg,
            loss_fn=loss_fn,
        )
        model_cfg.head.output_dim = head_output_dim
        model = model_cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Generate inputs.
        image = np.random.uniform(-1, 1, [batch_size, image_size, image_size, 3]).astype(np.float32)

        mask_model = mask_generator.MaskingGenerator(
            input_size=(image_size // patch_size, image_size // patch_size),
            min_mask_patches=1,
            num_masking_patches=5,
        )
        is_masked = mask_model()
        is_masked = jnp.expand_dims(is_masked, axis=0)
        is_masked = jnp.tile(is_masked, [batch_size, 1, 1])

        input_batch = dict(image=as_tensor(image), is_masked=as_tensor(is_masked))
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=input_batch),
        )

        self.assertEqual((batch_size, seq_len, head_output_dim), outputs[1]["logits"].shape)
        self.assertEqual(expected_label_shape, outputs[1]["labels"].shape)
        self.assertEqual((batch_size, seq_len), outputs[1]["is_masked"].shape)

    @parameterized.product(
        tokenizer=("fake", "beitv2", "clip"),
    )
    def test_no_masking_zero_loss(self, tokenizer: str):
        batch_size = 2
        image_size = 16
        patch_size = 4
        output_dim = 4
        codebook_size = 100
        min_mask_patches = 0
        num_masking_patches = 0
        expected_loss = 0

        seq_len = (image_size // patch_size) ** 2
        if tokenizer == "fake":
            tokenizer_cfg = FakeTokenizer.default_config().set(
                codebook_size=codebook_size,
                seq_len=seq_len,
            )
            loss_fn = cross_entropy
            head_output_dim = codebook_size
        elif tokenizer == "beitv2":
            tokenizer_cfg = beit_image_tokenizer.set_beit_image_tokenizer_encoder_config(
                num_layers=1,
                model_dim=output_dim,
                codebook_size=codebook_size,
                codebook_dim=8,
                num_heads=4,
                image_size=(image_size, image_size),
                patch_size=(patch_size, patch_size),
            )
            loss_fn = cross_entropy
            head_output_dim = codebook_size
        elif tokenizer == "clip":
            tokenizer_cfg = set_vision_encoder_config(
                num_layers=1,
                model_dim=output_dim,
                num_heads=4,
                projection_dim=output_dim,
                encoder_cls=feature_tokenizer.CLIPFeatureTokenizer,
            )
            loss_fn = negative_cosine_similarity_loss
            head_output_dim = output_dim
        else:
            raise ValueError(f"Tokenizer {tokenizer} not implemented.")

        encoder_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=output_dim,
            num_heads=4,
            image_size=(image_size, image_size),
            patch_size=(patch_size, patch_size),
        )
        encoder_cfg.use_mask_tokens = True
        model_cfg = MaskedImageModel.default_config().set(
            tokenizer=tokenizer_cfg,
            encoder=encoder_cfg,
            loss_fn=loss_fn,
        )
        model_cfg.head.output_dim = head_output_dim
        model = model_cfg.set(name="test").instantiate(parent=None)
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))

        # Generate inputs.
        image = np.random.uniform(-1, 1, [batch_size, image_size, image_size, 3]).astype(np.float32)

        mask_model = mask_generator.MaskingGenerator(
            input_size=(image_size // patch_size, image_size // patch_size),
            min_mask_patches=min_mask_patches,
            num_masking_patches=num_masking_patches,
        )
        is_masked = mask_model()
        is_masked = jnp.expand_dims(is_masked, axis=0)
        is_masked = jnp.tile(is_masked, [batch_size, 1, 1])

        input_batch = dict(image=as_tensor(image), is_masked=as_tensor(is_masked))
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=input_batch),
        )
        loss = outputs[0]
        self.assertEqual(loss, expected_loss)


if __name__ == "__main__":
    absltest.main()
