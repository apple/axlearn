# Copyright © 2023 Apple Inc.

"""Tests retrieval evaluation pipeline."""
# pylint: disable=no-self-use
import tempfile
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax.experimental.pjit import pjit

from axlearn.common.attention import NEG_INF
from axlearn.common.eval_retrieval import (
    CLIPRetrievalMetricCalculator,
    CxcImageRetrievalMetricCalculator,
    EmbeddingRetrievalMetricCalculator,
    KnnMetricCalculator,
    clip_generate_labels,
    get_pairwise_distances_sqr,
)
from axlearn.common.evaler_test import DummyModel
from axlearn.common.test_utils import TestCase, assert_allclose
from axlearn.common.utils import NestedTensor, Tensor


def dummy_data_generator(
    valid_sentences: Tensor,
    image_embeddings: Tensor,
    text_embeddings: Tensor,
    batch_size: int = 2,
):
    for batch_idx in range(int(np.ceil(len(image_embeddings) / batch_size))):
        start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
        yield {
            "input": {
                "valid_sentences": jnp.asarray(valid_sentences[start:end]),
            },
            "predicted": {
                "visual_encoder": {"output_features": jnp.asarray(image_embeddings[start:end])},
                "textual_encoder": {"output_features": jnp.asarray(text_embeddings[start:end])},
            },
        }


# pylint: disable-next=abstract-method
class DummyRetrievalModel(DummyModel):
    """A dummy model whose `embed_{image,text}_batch` returns input_batch["emb"])."""

    # pylint: disable-next=no-self-use
    def predict(self, input_batch: NestedTensor) -> Tensor:
        return input_batch["predicted"]


def _compute_metrics(
    *,
    data_generator,
    calculator_cfg,
) -> dict:
    """Computes zero-shot classification metrics.

    Args:
        text_labels: A list of class labels.
        text_embeddings: A list of embeddings, each corresponding to a label in `text_labels`.
            `text_labels` and `text_embeddings` must have the same length.
        image_labels: A list of image labels.
        image_embeddings: A list of embeddings, each corresponding to a label in `image_labels`.
            `image_labels` and `image_embeddings` must have the same length.
        top_ks: a list of K's to compute top-k accuracy.

    Returns:
        A dict containing:
        "class_embeddings": a Tensor of shape [num_classes, emb_dim] computed from text_labels
            and text_embeddings.
        "summaries": a Dict of WeightedScalar values computed by
            ZeroShotImageClassificationMetricCalculator.
    """
    with jax.sharding.Mesh(
        jax.experimental.mesh_utils.create_device_mesh((1, 1)), ("data", "model")
    ):
        model = DummyRetrievalModel.default_config().set(name="model").instantiate(parent=None)
        model_param_partition_specs = jax.tree.map(
            lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
        )
        calculator = calculator_cfg.set(name="calculator").instantiate(
            parent=None, model=model, model_param_partition_specs=model_param_partition_specs
        )
        model_params = pjit(
            model.initialize_parameters_recursively,
            in_shardings=(None,),
            out_shardings=model_param_partition_specs,
        )(jax.random.PRNGKey(0))

        state = calculator.init_state(prng_key=jax.random.PRNGKey(0), model_params=model_params)
        all_forward_outputs = []
        for input_batch in data_generator:
            forward_outputs = calculator.forward(
                input_batch, model_params=model_params, state=state
            )
            state = forward_outputs["state"]
            all_forward_outputs.append(forward_outputs["output"])
        summaries = calculator.get_summaries(
            model_params=model_params, state=state, all_forward_outputs=all_forward_outputs
        )
        return summaries


class CLIPRetrievalMetricCalculatorTest(TestCase, parameterized.TestCase):
    @parameterized.parameters(
        {"top_k": 1, "expected_metrics": {"i2t_top@1": 1 / 3, "t2i_top@1": 3 / 10}},
        {"top_k": 2, "expected_metrics": {"i2t_top@2": 2 / 3, "t2i_top@2": 8 / 10}},
        {
            "top_k": [1, 2],
            "expected_metrics": {
                "i2t_top@1": 1 / 3,
                "t2i_top@1": 3 / 10,
                "i2t_top@2": 2 / 3,
                "t2i_top@2": 8 / 10,
            },
        },
    )
    def test_clip_retrieval_metric_calculator(
        self, top_k: Union[int, list[int]], expected_metrics: dict[str, float]
    ):
        text_embeddings = jnp.asarray(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.2], [100, 100, 100]],
                [[0.5, 0, 0], [0, 0.5, 0], [0, 0.5, 0.7], [100, 100, 100], [100, 100, 100]],
                [[0.6, 0, 0], [0, 0.7, 0], [0, 0.6, 1], [100, 100, 100], [100, 100, 100]],
                [
                    [100, 100, 100],
                    [100, 100, 100],
                    [100, 100, 100],
                    [100, 100, 100],
                    [100, 100, 100],
                ],
            ]
        )
        valid_sentences = [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
        image_embeddings = [
            [[1, 0, 0]],
            [[0, 1, 0]],
            [[0, 0, 1]],
            [[100, 100, 100]],
        ]
        # For Image to Text Retrieval:
        # Top_1 accuracy:
        # Image_0 retrieved Text_0 (matched).
        # Image_1 retrieved Text_1 (missed). Image_1 needs to retrieval Text_[5-9].
        # Image_2 retrieved Text_2 (missed). Image_2 needs to retrieval Text_[10-14].
        # Image_3 is padded.
        # i2t_top@1 = 1/3. Noted Image_3 is padded. Therefore it doens't count.
        #
        # Top_2 accuracy:
        # Image_0 retrieved Text_0 and Text_10 (matched).
        # Image_1 retrieved Text_1 and Text_11 (missed).
        # Image_2 retrieved Text_2 and Text_13 (matched).
        # Image_3 is padded. Therefore, it doesn't count.
        # i2t_top@2 = 2/3

        # For Text to Image Retrieval:
        # Top_1 accuracy:
        # Text_0 retrieved Image_0 (matched).
        # Text_1 retrieved Image_1 (missed).
        # Text_2 retrieved Image_2 (missed).
        # Text_3 retrieved Image_2 (missed).
        # Text_5 retrieved Image_0 (missed).
        # Text_6 retrieved Image_1 (matched).
        # Text_7 retrieved Image_2 (missed).
        # Text_10 retrieved Image_0 (missed).
        # Text_11 retrieved Image_1 (missed).
        # Text_12 retrieved Image_2 (matched).
        # t2i_top@1 = 3/10
        #
        # Top_2 accuracy:
        # Text_0 retrieved Image_0 and Image_1 (matched).
        # Text_1 retrieved Image_1 and Image_0 (matched) (tie-breaker: prefers first image).
        # Text_2 retrieved Image_2 and Image_0 (matched) (tie-breaker).
        # Text_3 retrieved Image_2 and Image_0 (matched) (tie-breaker).
        # Text_5 retrieved Image_0 and Image_1 (matched) (tie-breaker).
        # Text_6 retrieved Image_1 and Image_0 (matched).
        # Text_7 retrieved Image_2 and Image_1 (matched) (tie-breaker).
        # Text_10 retrieved Image_0 and Image_1 (missed).
        # Text_11 retrieved Image_1 and Image_0 (missed).
        # Text_12 retrieved Image_2 and Image_1 (matched).
        # t2i_top@2 = 8/10

        if isinstance(top_k, int):
            top_k = [top_k]

        summaries = _compute_metrics(
            calculator_cfg=CLIPRetrievalMetricCalculator.default_config().set(top_ks=top_k),
            data_generator=dummy_data_generator(
                valid_sentences=valid_sentences,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
            ),
        )
        self.assertNestedAllClose(summaries, expected_metrics)

    def test_clip_generate_labels(self):  # pylint: disable=no-self-use
        sentence_paddings = jnp.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]])
        image_paddings = jnp.array([[0], [0], [0], [1]])
        labels_dict = clip_generate_labels(sentence_paddings)
        assert_allclose(
            labels_dict["image_to_text_label"],
            [[0, 1, 2, 3], [4, 5, 6, -1], [8, 9, -1, -1], [-1, -1, -1, -1]],
        )
        assert_allclose(
            labels_dict["text_to_image_label"],
            [[0], [0], [0], [0], [1], [1], [1], [-1], [2], [2], [-1], [-1], [-1], [-1], [-1], [-1]],
        )
        image_to_text_similarity_bias = np.array(labels_dict["image_to_text_similarity_bias"])
        text_to_image_similarity_bias = np.array(labels_dict["text_to_image_similarity_bias"])
        assert_allclose(image_to_text_similarity_bias, sentence_paddings * NEG_INF)
        assert_allclose(text_to_image_similarity_bias, image_paddings * NEG_INF)


class EmbeddingRetrievalMetricCalculatorTest(TestCase, parameterized.TestCase):
    @parameterized.parameters([1, 2, 3, 100])
    def test_compute_metrics_separate_query_index(self, max_query_chunk_size):
        def data_generator(batch_size=2):
            embedding = jnp.array(
                [[1.0, 0, 0], [0, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0]]
            )
            is_query = jnp.array([True, True, True, False, False, False])
            label = jnp.array([1, 2, 2, 2, 1, 1])
            category = jnp.array([0, 1, 0, 0, 1, 0])

            for start in range(0, embedding.shape[0], batch_size):
                end = start + batch_size
                yield {
                    "input": {
                        "is_query": is_query[start:end],
                        "label": label[start:end],
                        "category": category[start:end],
                    },
                    "predicted": {"embedding": embedding[start:end]},
                }

        calculator_cfg = EmbeddingRetrievalMetricCalculator.default_config().set(
            metrics=["MAP@1", "MAP@2", "accuracy@1", "recall@1"],
            categories_names=("cat", "dog"),
            max_query_chunk_size=max_query_chunk_size,
        )
        summaries = _compute_metrics(
            calculator_cfg=calculator_cfg,
            data_generator=data_generator(),
        )
        expected_metrics = {
            "MAP@1": (1.0 + 0.0 + 0.0) / 3,
            "MAP@1_cat": (1.0 + 0.0) / 2,
            "MAP@1_dog": 0.0,
            "MAP@1_avg_category": ((1.0 + 0.0) / 2 + 0.0) / 2,
            "MAP@2": (0.5 + 0.5 + 0.5) / 3,
            "MAP@2_cat": (0.5 + 0.5) / 2,
            "MAP@2_dog": 0.5,
            "MAP@2_avg_category": ((0.5 + 0.5) / 2 + 0.5) / 2,
            "accuracy@1": (1.0 + 0.0 + 0.0) / 3,
            "accuracy@1_cat": (1.0 + 0.0) / 2,
            "accuracy@1_dog": 0.0,
            "accuracy@1_avg_category": ((1.0 + 0.0) / 2 + 0.0) / 2,
            "recall@1": (1.0 + 0.0 + 0.0) / 3,
            "recall@1_cat": (1.0 + 0.0) / 2,
            "recall@1_dog": 0.0,
            "recall@1_avg_category": ((1.0 + 0.0) / 2 + 0.0) / 2,
            "num_valid": 3,
        }
        self.assertNestedAllClose(summaries, expected_metrics)

    @parameterized.parameters([1, 2, 3, 100])
    def test_compute_metrics_same_query_index(self, max_query_chunk_size):
        def data_generator(batch_size=2):
            embedding = jnp.array(
                [[1.0, 0, 0], [0, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0]]
            )
            label = jnp.array([1, 2, 2, 2, 1, 1])
            category = jnp.array([0, 1, 0, 0, 1, 0])

            for start in range(0, embedding.shape[0], batch_size):
                end = start + batch_size
                yield {
                    "input": {
                        "label": label[start:end],
                        "category": category[start:end],
                    },
                    "predicted": {"embedding": embedding[start:end]},
                }

        calculator_cfg = EmbeddingRetrievalMetricCalculator.default_config().set(
            metrics=["MAP@1", "MAP@2"],
            categories_names=("cat", "dog"),
            max_query_chunk_size=max_query_chunk_size,
        )
        summaries = _compute_metrics(
            calculator_cfg=calculator_cfg,
            data_generator=data_generator(),
        )
        expected_metrics = {
            "MAP@1": (1.0 + 1.0 + 1.0 + 0.0 + 1.0 + 0.0) / 6,
            "MAP@1_cat": (1.0 + 1.0 + 0.0 + 0.0) / 4,
            "MAP@1_dog": (1.0 + 1.0) / 2,
            "MAP@1_avg_category": ((1.0 + 1.0 + 0.0 + 0.0) / 4 + (1.0 + 1.0) / 2) / 2,
            "MAP@2": (0.5 + 0.5 + 0.5 + 0.25 + 0.5 + 0.0) / 6,
            "MAP@2_cat": (0.5 + 0.5 + 0.25 + 0.0) / 4,
            "MAP@2_dog": (0.5 + 0.5) / 2,
            "MAP@2_avg_category": ((0.5 + 0.5 + 0.25 + 0.0) / 4 + (0.5 + 0.5) / 2) / 2,
            "num_valid": 6,
        }
        self.assertNestedAllClose(summaries, expected_metrics)

    def test_compute_metrics_with_prefix(self):
        def data_generator(batch_size=2):
            embedding = jnp.array(
                [[1.0, 0, 0], [0, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0]]
            )
            label = jnp.array([1, 2, 2, 2, 1, 1])
            category = jnp.array([0, 1, 0, 0, 1, 0])

            for start in range(0, embedding.shape[0], batch_size):
                end = start + batch_size
                yield {
                    "input": {
                        "label": label[start:end],
                        "category": category[start:end],
                    },
                    "predicted": {"embedding": embedding[start:end]},
                }

        calculator_cfg = EmbeddingRetrievalMetricCalculator.default_config().set(
            metrics=["MAP@1"], categories_names=("cat", "dog"), prefix="test"
        )
        summaries = _compute_metrics(
            calculator_cfg=calculator_cfg,
            data_generator=data_generator(),
        )
        expected_metrics = {
            "test/MAP@1": (1.0 + 1.0 + 1.0 + 0.0 + 1.0 + 0.0) / 6,
            "test/MAP@1_cat": (1.0 + 1.0 + 0.0 + 0.0) / 4,
            "test/MAP@1_dog": (1.0 + 1.0) / 2,
            "test/MAP@1_avg_category": ((1.0 + 1.0 + 0.0 + 0.0) / 4 + (1.0 + 1.0) / 2) / 2,
            "test/num_valid": 6,
        }
        self.assertNestedAllClose(summaries, expected_metrics)

    def test_get_pairwise_distances_sqr(self):
        prng = jax.random.split(jax.random.PRNGKey(777))
        a = jax.random.uniform(prng[0], (16, 32))
        b = jax.random.uniform(prng[1], (64, 32))
        dists = get_pairwise_distances_sqr(a, b)
        expected_dists = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        self.assertNestedAllClose(expected_dists, dists)


_TEST_IMAGE_RELEVANCE_CSV = """image1,image2,agg_score,sampling_method
COCO_val2014_000000205866.jpg,COCO_val2014_000000039081.jpg,3.6,i2i_csim
COCO_val2014_000000520871.jpg,COCO_val2014_000000358039.jpg,3.99,i2i_csim
COCO_val2014_000000281782.jpg,COCO_val2014_000000447314.jpg,3.9,i2i_csim
COCO_val2014_000000529981.jpg,COCO_val2014_000000074460.jpg,3.61,i2i_csim
COCO_val2014_000000001682.jpg,COCO_val2014_000000409088.jpg,1.72,i2i_csim
"""


class CxcImageRetrievalMetricCalculatorTest(TestCase, parameterized.TestCase):
    @parameterized.parameters([1, 2, 3, 100])
    def test_compute_metrics_same_query_index(self, max_query_chunk_size):
        def data_generator(batch_size=3):
            embedding = jnp.array([[1.0, 0.0], [2.0, 0.0], [0.5, 0.5]])
            # Only 205866 and 39081 are relevant.
            image_id = jnp.array([205866, 39081, 1682])

            for start in range(0, embedding.shape[0], batch_size):
                end = start + batch_size
                yield {
                    "input": {
                        "image_id": image_id[start:end],
                    },
                    "predicted": {"embedding": embedding[start:end]},
                }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
            f.write(_TEST_IMAGE_RELEVANCE_CSV)
            f.flush()
            calculator_cfg = CxcImageRetrievalMetricCalculator.default_config().set(
                image_relevance_path=f.name,
                metrics=["accuracy@1", "accuracy@2"],
                max_query_chunk_size=max_query_chunk_size,
            )
            summaries = _compute_metrics(
                calculator_cfg=calculator_cfg,
                data_generator=data_generator(),
            )

        expected_metrics = {
            "accuracy@1": (0.0 + 1.0 + 0.0) / 3,
            "accuracy@2": (1.0 + 1.0 + 0.0) / 3,
            "num_valid": 3,
        }
        self.assertNestedAllClose(summaries, expected_metrics)


class KnnMetricCalculatorTest(TestCase, parameterized.TestCase):
    def test_compute_metrics(self):
        def data_generator(batch_size=2):
            embedding = jnp.array(
                [[1.0, 0, 0], [0, 1, 0], [0, 1, 0], [0.5, 0.5, 0.5], [1, 0, 0], [0, 1, 0]]
            )
            is_query = jnp.array([True, True, True, False, False, False])
            label = jnp.array([1, 1, 2, 1, 1, 2])
            category = jnp.array([0, 1, 0, 0, 1, 0])

            for start in range(0, embedding.shape[0], batch_size):
                end = start + batch_size
                yield {
                    "input": {
                        "is_query": is_query[start:end],
                        "label": label[start:end],
                        "category": category[start:end],
                    },
                    "predicted": {"embedding": embedding[start:end]},
                }

        calculator_cfg = KnnMetricCalculator.default_config().set(
            categories_names=("cat", "dog"),
            num_labels=3,
            top_ks=(1, 2, 3),
            temps=[2.01],
        )
        summaries = _compute_metrics(
            calculator_cfg=calculator_cfg,
            data_generator=data_generator(),
        )
        expected_metrics = {
            "knn-t2.01@1": (1.0 + 0.0 + 1.0) / 3.0,
            "knn-t2.01@1_cat": (1.0 + 1.0) / 2,
            "knn-t2.01@1_dog": 0.0,
            "knn-t2.01@1_avg_category": 0.5,
            "knn-t2.01@2": (1.0 + 0.0 + 1.0) / 3.0,
            "knn-t2.01@2_cat": (1.0 + 1.0) / 2,
            "knn-t2.01@2_dog": 0.0,
            "knn-t2.01@2_avg_category": 0.5,
            "knn-t2.01@3": (1.0 + 1.0 + 0.0) / 3.0,
            "knn-t2.01@3_cat": (1.0 + 0.0) / 2,
            "knn-t2.01@3_dog": 1.0,
            "knn-t2.01@3_avg_category": 0.75,
            "num_valid": 3,
        }
        self.assertNestedAllClose(summaries, expected_metrics)
