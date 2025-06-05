# Copyright © 2023 Apple Inc.

"""Test distillation layers."""
# pylint: disable=no-member,no-self-use
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

from axlearn.common.distillation import (
    DeiTDistillationMetric,
    DeiTDistillationModel,
    DistillationModel,
    KLDivergenceMetric,
    NegativeCosineSimilarityMetric,
)
from axlearn.common.loss import kl_divergence, negative_cosine_similarity_loss
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import as_tensor
from axlearn.common.vision_transformer import build_vit_model_config
from axlearn.vision.image_classification import ImageClassificationModel


class KLDistillationTest(TestCase):
    def test_distillation_model_with_kldivergence(self):
        vit_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            patch_size=(16, 16),
            global_feature_extraction="cls_token",
        )
        num_classes = 1000
        child_cfg = ImageClassificationModel.default_config().set(
            backbone=vit_cfg, num_classes=num_classes
        )
        model = (
            DistillationModel.default_config()
            .set(
                name="test",
                teacher=child_cfg,
                student=child_cfg,
            )
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.random.randint(0, num_classes - 1, [batch_size]).astype(np.int32),
        }
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=as_tensor(inputs)),
        )
        self.assertEqual((batch_size, num_classes), outputs[1]["student_outputs"]["logits"].shape)
        self.assertEqual((batch_size, num_classes), outputs[1]["teacher_outputs"]["logits"].shape)

    def test_distillation_model_with_negative_cosine_similarity(self):
        model_dim = 8
        vit_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=model_dim,
            num_heads=4,
            patch_size=(16, 16),
            global_feature_extraction="cls_token",
        )
        num_classes = 1000
        child_cfg = ImageClassificationModel.default_config().set(
            backbone=vit_cfg, num_classes=num_classes
        )
        model = (
            DistillationModel.default_config()
            .set(
                name="test",
                teacher=child_cfg,
                student=child_cfg,
                metric=NegativeCosineSimilarityMetric.default_config(),
            )
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
        }
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=as_tensor(inputs)),
        )
        self.assertEqual((batch_size, model_dim), outputs[1]["student_outputs"]["embedding"].shape)
        self.assertEqual((batch_size, model_dim), outputs[1]["teacher_outputs"]["embedding"].shape)

    @parameterized.parameters(False, True)
    def test_kl_divergence_metric(self, is_log_targets):
        batch_size = 2
        num_classes = 1000
        logits_s = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        logits_t = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        labels = np.random.randint(0, num_classes - 1, [batch_size]).astype(np.int32)
        layer = (
            KLDivergenceMetric.default_config()
            .set(name="test", is_log_targets=is_log_targets)
            .instantiate(parent=None)
        )
        loss, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=None,
            inputs=dict(
                predictions={"logits": logits_s},
                targets={"logits": logits_t},
                label=labels,
            ),
        )
        ref_loss, _ = kl_divergence(
            log_predictions=jax.nn.log_softmax(logits_s),
            targets=jax.nn.log_softmax(logits_t) if is_log_targets else jax.nn.softmax(logits_t),
            is_log_targets=is_log_targets,
        )
        assert jnp.allclose(loss, ref_loss)

    @parameterized.parameters(False, True)
    def test_negative_cosine_similarity_metric(self, normalize_embedding):
        batch_size = 2
        dim = 256
        emb_s = np.random.uniform(-1, 1, [batch_size, dim]).astype(np.float32)
        emb_t = np.random.uniform(-1, 1, [batch_size, dim]).astype(np.float32)
        layer = (
            NegativeCosineSimilarityMetric.default_config()
            .set(name="test", normalize_embedding=normalize_embedding)
            .instantiate(parent=None)
        )
        loss, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=None,
            inputs=dict(
                predictions={"embedding": emb_s},
                targets={"embedding": emb_t},
            ),
        )
        ref_loss, _ = negative_cosine_similarity_loss(
            predictions=emb_s,
            targets=emb_t,
            normalize_embedding=normalize_embedding,
        )
        assert jnp.allclose(loss, ref_loss)


class DeiTDistillationTest(TestCase):
    def test_deit_distillation_model(self):
        vit_cfg = build_vit_model_config(
            num_layers=1,
            model_dim=8,
            num_heads=4,
            patch_size=(16, 16),
            global_feature_extraction="cls_distill_token",
        )
        num_classes = 1000
        child_cfg = ImageClassificationModel.default_config().set(
            backbone=vit_cfg, num_classes=num_classes
        )
        model = (
            DeiTDistillationModel.default_config()
            .set(name="test", teacher=child_cfg, student=child_cfg, num_classes=num_classes)
            .instantiate(parent=None)
        )
        state = model.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123))
        batch_size = 2
        inputs = {
            "image": np.random.uniform(-1, 1, [batch_size, 224, 224, 3]).astype(np.float32),
            "label": np.random.randint(0, num_classes - 1, [batch_size]).astype(np.int32),
        }
        outputs, _ = F(
            model,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs=dict(input_batch=as_tensor(inputs)),
        )
        self.assertEqual((batch_size, num_classes), outputs[1]["logits"].shape)

    def _compute_loss(self, logits, labels, num_classes):
        num_examples = (labels >= 0).sum()
        denominator = jnp.maximum(1, num_examples)
        labels_onehot = jax.nn.one_hot(labels, num_classes, dtype=logits.dtype)
        per_example_loss = jnp.sum(-1 * labels_onehot * jax.nn.log_softmax(logits), axis=-1)
        loss = per_example_loss.sum() / denominator
        return loss

    @parameterized.parameters((0, True), (0.5, False), (1.0, False))
    def test_deit_distillation_metric(self, alpha, use_soft_labels):
        batch_size = 2
        num_classes = 1000
        logits_teacher = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        logits_classification = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(
            np.float32
        )
        logits_distillation = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        labels = np.random.randint(0, num_classes - 1, [batch_size]).astype(np.int32)
        if use_soft_labels:
            soft_labels = jax.nn.one_hot(labels, num_classes, dtype=np.float32)
        else:
            soft_labels = None

        layer = (
            DeiTDistillationMetric.default_config()
            .set(name="test", num_classes=num_classes, alpha=alpha)
            .instantiate(parent=None)
        )
        loss, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=None,
            inputs=dict(
                logits_teacher=logits_teacher,
                logits_classification=logits_classification,
                logits_distillation=logits_distillation,
                labels=labels,
                soft_labels=soft_labels,
            ),
        )
        ref_loss_s = self._compute_loss(logits_classification, labels, num_classes)
        ref_loss_xd = self._compute_loss(
            logits_distillation, jnp.argmax(logits_teacher, axis=1), num_classes
        )
        self.assertEqual(loss, ref_loss_s * (1 - alpha) + ref_loss_xd * alpha)

    @parameterized.parameters(-1, 1)
    def test_deit_distillation_metric_with_padding(self, padding_label_sign):
        batch_size = 2
        num_classes = 1000
        logits_teacher = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        logits_classification = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(
            np.float32
        )
        logits_distillation = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        labels = np.random.randint(0, num_classes - 1, [batch_size]).astype(np.int32)

        layer = (
            DeiTDistillationMetric.default_config()
            .set(name="test", num_classes=num_classes)
            .instantiate(parent=None)
        )
        loss, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=None,
            inputs=dict(
                logits_teacher=logits_teacher,
                logits_classification=logits_classification,
                logits_distillation=logits_distillation,
                labels=labels,
            ),
        )
        padding_logits = np.random.uniform(-1, 1, [batch_size, num_classes]).astype(np.float32)
        padding_labels = padding_label_sign * np.ones(shape=[batch_size])
        padding_loss, _ = F(
            layer,
            is_training=True,
            prng_key=jax.random.PRNGKey(123),
            state=None,
            inputs=dict(
                logits_teacher=np.concatenate([logits_teacher, padding_logits]),
                logits_classification=np.concatenate([logits_classification, padding_logits]),
                logits_distillation=np.concatenate([logits_distillation, padding_logits]),
                labels=np.concatenate([labels, padding_labels]),
            ),
        )
        if padding_label_sign >= 0:
            self.assertNotEqual(loss, padding_loss)
        else:
            self.assertEqual(loss, padding_loss)


if __name__ == "__main__":
    absltest.main()
