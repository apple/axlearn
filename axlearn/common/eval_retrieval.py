# Copyright © 2023 Apple Inc.

"""Retrieval evaluation pipeline."""

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Callable, Optional

import jax
import numpy as np
from absl import logging
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from axlearn.common import file_system as fs
from axlearn.common import utils
from axlearn.common.attention import NEG_INF
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.evaler import GlobalMetricCalculator, PredictionOutputs
from axlearn.common.loss import contrastive_logits
from axlearn.common.metrics_retrieval import (
    average_precision_at_k,
    calculate_mean_metrics,
    metric_at_k_name,
    top_k_accuracy,
    top_k_recall,
)
from axlearn.common.module import Module
from axlearn.common.utils import NestedPartitionSpec, NestedTensor, Tensor, with_sharding_constraint


def clip_generate_labels(sentence_paddings: Tensor) -> dict[str, Tensor]:
    """Generate the image to text and text to image retrieval labels.

    Assumption:
        Given one batch, the image and text in this batch are aligned.

    Define:
        num_batch_instances = num_eval_iters * eval_batch_size
        total_num_sentences = num_batch_instances * eval_num_sentences_per_image

    Image to text label:
        Shape [num_batch_instances, num_sentences].
        image_to_text_label[i] = [i_1, i_2, i_3, ..., -1, ...] means the labels
            of the i-th image are text i_1, text i_2, ...
            Negative value means padded item.

    Text to image label:
        Shape [max_sentence_index, 1]
        text_to_image_label[i] = [j] means the label of the i-th text is image j.
        Negative value means padded item.

    Args:
        sentence_paddings: Shape [num_batch_instances, num_sentences], dtype=jnp.int32
            0 means valid, 1 means padded.

    Returns:
        A dictionary containing:
            *"image_to_text_label": A Tensor for image to text label with shape
                [num_batch_instances, num_sentences]. dtype=int32
                image_to_text_label[i,j]<0 means the j-th position is padded.
                It will not counted toward calculate the top_k metrics.
                If the value >= 0, it will be in range [0, total_num_sentences).
            *"text_to_image_label": A Tensor for text to image label with shape
                [max_sentence_index, 1]. dtype=int32
                text_to_image_label[i,1]<0 means the i-th text is padded.
                If the value >= 0, it will be in range [0, num_batch_instances).
            *"image_to_text_similarity_bias": A Tensor for similarity biases with
                shape [num_batch_instances, num_sentences]. dtype=float
            *"text_to_image_similarity_bias": A Tensor for similarity biases with
                shape [num_batch_instances, 1]. dtype=float
    """
    num_sentences = sentence_paddings.shape[1]
    num_batch_instances = sentence_paddings.shape[0]
    total_num_sentences = num_batch_instances * num_sentences

    text_to_image_label = jnp.arange(0, num_batch_instances, dtype=jnp.int32)

    # [total_num_sentences, 1].
    text_to_image_label = jnp.repeat(
        jnp.expand_dims(text_to_image_label, 1),
        repeats=num_sentences,
        axis=1,
    ).reshape((-1, 1))

    text_to_image_label = text_to_image_label * (
        1 - sentence_paddings.reshape(-1, 1)
    ) - sentence_paddings.reshape(-1, 1)

    image_to_text_label = jnp.arange(0, total_num_sentences, dtype=jnp.int32).reshape(
        num_batch_instances, num_sentences
    )

    image_to_text_label = image_to_text_label * (1 - sentence_paddings) - sentence_paddings

    image_paddings = 1 - (jnp.sum(1 - sentence_paddings, 1, keepdims=True) > 0)
    text_to_image_similarity_bias = image_paddings * NEG_INF
    image_to_text_similarity_bias = sentence_paddings * NEG_INF

    return dict(
        image_to_text_label=image_to_text_label,
        image_to_text_similarity_bias=image_to_text_similarity_bias,
        text_to_image_label=text_to_image_label,
        text_to_image_similarity_bias=text_to_image_similarity_bias,
    )


def calculate_clip_retrieval_metrics(
    *,
    top_ks: list[int],
    visual_embeddings: Tensor,
    textual_embeddings: Tensor,
    valid_sentences: Tensor,
) -> dict[str, Tensor]:
    """Calculates top-k accuracies.

    Args:
        top_ks: A list of K's for top-k stats.
        visual_embeddings: [num_examples, 1, dim].
        textual_embeddings: [num_examples, num_sentences, dim].
        valid_sentences: [num_examples, num_sentences].

    Returns:
        A dict containing:
        - "i2t_top@{k}": image-to-text accuracy at top-k
        - "t2i_top@{k}": text-to-image accuracy at top-k
    """
    visual_embeddings = with_sharding_constraint(visual_embeddings, PartitionSpec(None, None))
    textual_embeddings = with_sharding_constraint(textual_embeddings, PartitionSpec(None, None))
    valid_sentences = with_sharding_constraint(valid_sentences, PartitionSpec(None, None))
    num_examples, num_sentences = valid_sentences.shape
    dim = visual_embeddings.shape[-1]
    assert textual_embeddings.shape == (
        num_examples,
        num_sentences,
        dim,
    ), f"{textual_embeddings.shape} vs. {valid_sentences.shape}"
    assert visual_embeddings.shape == (
        num_examples,
        1,
        dim,
    ), f"{visual_embeddings.shape} vs. {valid_sentences.shape}"

    # [num_examples].
    valid_examples = (jnp.sum(valid_sentences, axis=1) > 0).astype(jnp.int32)

    dim = visual_embeddings.shape[-1]

    # Reshape embeddings from [num_examples, x, dim] to [num_examples * x, dim].
    # For visual embeddings: x = 1
    # For textual embeddings: x = num_sentences
    flatten_visual_embeddings = visual_embeddings.reshape(-1, dim)
    flatten_textual_embeddings = textual_embeddings.reshape(-1, dim)
    i2t_sim = contrastive_logits(flatten_visual_embeddings, flatten_textual_embeddings)
    t2i_sim = i2t_sim.T

    labels_dict = clip_generate_labels(1 - valid_sentences)

    # Calculate the metrics.
    i2t_topk_per_instance = top_k_accuracy(
        i2t_sim,
        labels_dict["image_to_text_label"],
        top_ks,
        similarity_bias=labels_dict["image_to_text_similarity_bias"],
    )
    t2i_topk_per_instance = top_k_accuracy(
        t2i_sim,
        labels_dict["text_to_image_label"],
        top_ks,
        similarity_bias=labels_dict["text_to_image_similarity_bias"],
    )
    metrics = {}
    flatten_valid_examples = valid_examples.reshape(-1).astype(jnp.float32)
    flatten_valid_sentences = valid_sentences.reshape(-1).astype(jnp.float32)
    for idx, k in enumerate(top_ks):
        i2t_topk_avg = jnp.sum(i2t_topk_per_instance[idx] * flatten_valid_examples) / jnp.sum(
            flatten_valid_examples
        )
        t2i_topk_avg = jnp.sum(t2i_topk_per_instance[idx] * flatten_valid_sentences) / jnp.sum(
            flatten_valid_sentences
        )
        metrics[f"i2t_top@{k}"] = i2t_topk_avg
        metrics[f"t2i_top@{k}"] = t2i_topk_avg
    return metrics


class CLIPRetrievalMetricCalculator(GlobalMetricCalculator):
    """A metric calculator for CLIP retrieval top-k accuracies."""

    @config_class
    class Config(GlobalMetricCalculator.Config):
        # The values for k for which to compute accuracy.
        top_ks: Required[list[int]] = REQUIRED

    def _calculate_metrics(self, outputs: PredictionOutputs) -> dict[str, Tensor]:
        cfg = self.config
        metrics = calculate_clip_retrieval_metrics(
            top_ks=cfg.top_ks,
            visual_embeddings=outputs.predict_outputs["visual_encoder"]["output_features"],
            textual_embeddings=outputs.predict_outputs["textual_encoder"]["output_features"],
            valid_sentences=outputs.input_batch["input"]["valid_sentences"],
        )
        formatted_metrics = {}
        for name, val in metrics.items():
            formatted_metrics[self.formatted_metric_name(name)] = val

        return formatted_metrics


def _named_average_precision_at_k(
    scores: Tensor, *, relevance_labels: Tensor, name: str, top_ks: list[int]
) -> dict[str, str]:
    output = average_precision_at_k(scores=scores, relevance_labels=relevance_labels, top_ks=top_ks)
    return {metric_at_k_name(name, k): v for k, v in output.items()}


def _named_top_k_accuracy(
    scores: Tensor, *, relevance_labels: Tensor, name: str, top_ks: list[int]
) -> dict[str, str]:
    output = top_k_accuracy(
        sim=scores, gt_targets=None, relevance_labels=relevance_labels, top_ks=top_ks
    )
    return {metric_at_k_name(name, k): v for k, v in zip(top_ks, output)}


def _named_top_k_recall(
    scores: Tensor, *, relevance_labels: Tensor, name: str, top_ks: list[int]
) -> dict[str, str]:
    output = top_k_recall(
        sim=scores, gt_targets=None, relevance_labels=relevance_labels, top_ks=top_ks
    )
    return {metric_at_k_name(name, k): v for k, v in zip(top_ks, output)}


def _get_ranking_metrics_fns(metrics: Sequence[str]) -> list[Callable]:
    # Parse metric names.
    metric_to_ks = defaultdict(list)
    for metric in metrics:
        name, k = metric.split("@", 1) if "@" in metric else (metric, -1)
        k = int(k)
        metric_to_ks[name].append(k)
    # Create callbacks.
    metrics_fns = []
    for name, ks in metric_to_ks.items():
        if name == "MAP":
            metrics_fns.append(partial(_named_average_precision_at_k, top_ks=ks, name=name))
        elif name == "accuracy":
            metrics_fns.append(partial(_named_top_k_accuracy, top_ks=ks, name=name))
        elif name == "recall":
            metrics_fns.append(partial(_named_top_k_recall, top_ks=ks, name=name))
        else:
            raise ValueError(f"Unknown metric: {name}")
    return metrics_fns


class EmbeddingRetrievalMetricCalculator(GlobalMetricCalculator):
    """A metric calculator for embedding based retrieval.

    Please see `_calculate_metrics` function's for input/output expectations.
    """

    @config_class
    class Config(GlobalMetricCalculator.Config):
        """Configures EmbeddingRetrievalMetricCalculator."""

        # List of metrics to be computed. Supported:
        #   * MAP - mean average precision
        #   * MAP@k - mean average precision at K
        #   * accuracy@k - accuracy at K
        # TODO(atimofeev): add more supported metrics (accuracy, recall, etc.)
        metrics: Required[list[str]] = REQUIRED

        # Optional tuple of cateogory names - category `i` gets name `categories_names[i]`.
        # Categories are used for providing metrics for a subset of items belonging to a particular
        # category.
        categories_names: Optional[tuple[str, ...]] = None

        # Field inside output which contains embedding for the evaluation.
        # For nested fields use 'root/nested1/nested2' notation.
        embedding_field: Optional[str] = "embedding"

        # If true, allows self-matches (for compatibility with GPR1200).
        allow_self_match: bool = False

        # Max number of queries to be processed at once. Smaller chunk size helps to avoid OOM, but
        # might make evaluation slower.
        max_query_chunk_size: int = 5000

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(
            cfg,
            parent=parent,
            model=model,
            model_param_partition_specs=model_param_partition_specs,
            use_jit_for_metric_calculation=False,
        )
        self._metrics_fns = _get_ranking_metrics_fns(cfg.metrics)

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        """Calls predict method of the model and returns input_batch and per-batch model outputs.

        Will be called repeatedly during an evaluation step, once per evaluation input batch.

        Args:
            input_batch: The evaluation input batch.
            model_params: The model parameters.
            state: As returned by `init_state` or by the previous invocation of `forward`.

        Returns:
            A dict containing:
            - "state": A dict containing prng_key.
            - "output": A dict containing input_batch and per-batch model outputs.
        """
        result = super().forward(input_batch, model_params=model_params, state=state)
        result["output"] = self._select_predictions(result["output"])
        return result

    def _calculate_metrics(self, outputs: PredictionOutputs) -> dict[str, Tensor]:
        """Calculates configured metrics for the embedding queries retrieval from index embeddings.

        An index item is considered relevant to a query if they share the same label.

        Args:
            outputs: PredictionOutputs with input_batch containing the following fields
                * "label" - an int tensor with label ids; index items are considered relevant to
                    a query if they have the same label as the query;
                * "category" - an optional int tensor with examples category ids. Used for
                    breaking down metrics by queries category.
                * "is_query" - an optional bool tensor with True for query embeddings and False for
                    index embeddings; if not provided all items are used as query and index;
                and predict_outputs containing`
                * "embedding" - embedding vectors which will be used for computing distances.

        Returns:
            A mapping from metric name to corresponding value. If categories were provided, then
            the metrics are also broken down by category.
        """
        query_indices_chunked, num_queries = self._get_query_indices_chunked(outputs)
        metrics_chunked = []
        categories_chunked = []
        for query_indices in query_indices_chunked:
            categories_chunked.append(self._get_categories(outputs, query_indices))
            metrics_chunked.append(self._get_chunk_metrics(outputs, query_indices))

        if categories_chunked[0] is not None:
            categories = jnp.concatenate(categories_chunked)
        else:
            categories = None
        per_query_metrics = jax.tree.map(  # pylint: disable=no-value-for-parameter
            lambda *xs: jnp.concatenate(xs), *metrics_chunked
        )

        metrics = {"num_valid": num_queries}
        for name, query_metrics in per_query_metrics.items():
            assert query_metrics.shape[0] == num_queries
            aggregated = calculate_mean_metrics(
                metric_name=name,
                query_metrics=query_metrics,
                query_padding=jnp.zeros(num_queries, dtype="bool"),  # all valid
                query_categories=categories,
                categories_names=self.config.categories_names,
            )
            metrics.update(aggregated)

        formatted_metrics = {self.formatted_metric_name(name): val for name, val in metrics.items()}
        logging.info("Computed: %s", formatted_metrics)
        return formatted_metrics

    def _get_chunk_metrics(
        self, outputs: PredictionOutputs, query_indices: Tensor
    ) -> dict[str, Tensor]:
        """Returns metrics for the given chunk of query indices."""
        scores = self._get_scores(outputs, query_indices)
        relevance_labels = self._get_relevance_labels(outputs, query_indices)
        chunk_metrics = {}
        for metric_fn in self._metrics_fns:
            new_metrics = metric_fn(scores=scores, relevance_labels=relevance_labels)
            chunk_metrics.update(new_metrics)
        return chunk_metrics

    def _select_predictions(self, outputs: PredictionOutputs) -> PredictionOutputs:
        """Filter predictions which are required for the metric computation."""
        embeddings = utils.get_recursively(outputs.predict_outputs, self.config.embedding_field)
        return PredictionOutputs(
            input_batch=_select_keys(
                outputs.input_batch["input"], ["label", "category", "is_query"]
            ),
            predict_outputs={"embedding": embeddings},
        )

    def _get_query_indices_chunked(self, outputs: PredictionOutputs) -> tuple[list[Tensor], int]:
        is_query = outputs.input_batch.get("is_query")
        if is_query is None:
            num = outputs.predict_outputs["embedding"].shape[0]
            is_query = jnp.ones([num], dtype="bool")
        is_query = jnp.logical_and(is_query, self._get_valids_mask(outputs))
        query_indices = jnp.where(is_query)[0]
        num_queries = query_indices.shape[0]
        chunk_size = self.config.max_query_chunk_size
        chunk_splits = jnp.arange(chunk_size, num_queries, chunk_size)
        query_indices_chunked = jnp.split(query_indices, chunk_splits)
        return query_indices_chunked, num_queries

    # pylint: disable-next=no-self-use
    def _get_valids_mask(self, outputs: PredictionOutputs) -> Tensor:
        labels = outputs.input_batch["label"]
        return labels >= 0

    def _get_scores(self, outputs: PredictionOutputs, query_indices: Tensor) -> Tensor:
        embeddings = outputs.predict_outputs["embedding"]
        if len(embeddings.shape) == 3:
            # Multi-stream models have dummy second dimension, remove it.
            embeddings = jnp.squeeze(embeddings, 1)

        index_mask = self._get_index_mask(outputs)
        scores = -get_pairwise_distances_sqr(embeddings[query_indices], embeddings[index_mask])
        if "is_query" not in outputs.input_batch and not self.config.allow_self_match:
            # Mask out self-similarity.
            self_mask = jax.nn.one_hot(query_indices, scores.shape[1])
            scores += self_mask * NEG_INF
        return scores

    def _get_relevance_labels(self, outputs: PredictionOutputs, query_indices: Tensor) -> Tensor:
        labels = outputs.input_batch["label"]
        index_mask = self._get_index_mask(outputs)
        return labels[query_indices][:, None] == labels[index_mask][None, :]

    # pylint: disable-next=no-self-use
    def _get_categories(
        self, outputs: PredictionOutputs, query_indices: Tensor
    ) -> Optional[Tensor]:
        categories = outputs.input_batch.get("category")
        return None if categories is None else categories[query_indices]

    # pylint: disable-next=no-self-use
    def _get_index_mask(self, outputs) -> Tensor:
        index_mask = self._get_valids_mask(outputs)
        is_query = outputs.input_batch.get("is_query")
        if is_query is not None:
            index_mask = jnp.logical_and(index_mask, ~is_query)
        return index_mask


def get_pairwise_distances_sqr(a: Tensor, b: Tensor) -> Tensor:
    # Avoid large broadcasting.
    aa = jnp.sum(a**2, axis=-1)
    bb = jnp.sum(b**2, axis=-1)
    ab = jnp.einsum("i d, j d -> i j", a, b)
    return aa[:, None] + bb[None, :] - 2 * ab


def _select_keys(kv: Mapping[str, Any], keys: Sequence[str]):
    return {k: v for k, v in kv.items() if k in set(keys)}


class CxcImageRetrievalMetricCalculator(EmbeddingRetrievalMetricCalculator):
    """A metric calculator for cxc image retrieval.

    Image data (5000 images) comes from "test" split of
    "gs://axlearn-public/tensorflow_datasets/coco_captions".

    Please see https://arxiv.org/pdf/2004.15020.pdf for more info.
    """

    @config_class
    class Config(EmbeddingRetrievalMetricCalculator.Config):
        image_relevance_path: str = "gs://axlearn-public/generic_embedding/eval/cxc/sis_test.csv"
        threshold: float = 3.0

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(
            cfg,
            parent=parent,
            model=model,
            model_param_partition_specs=model_param_partition_specs,
        )
        self._relevance_scores, self._image_id_to_pos = _load_cxc_relevance(
            cfg.image_relevance_path
        )

    def _select_predictions(self, outputs: PredictionOutputs) -> PredictionOutputs:
        """Filter predictions which are required for the metric computation."""
        embeddings = utils.get_recursively(outputs.predict_outputs, self.config.embedding_field)
        return PredictionOutputs(
            input_batch=_select_keys(outputs.input_batch["input"], ["image_id"]),
            predict_outputs={"embedding": embeddings},
        )

    def _get_permutation(self, outputs: PredictionOutputs) -> Tensor:
        return jnp.array(
            [
                self._image_id_to_pos.get(f"COCO_val2014_{image_id:012}.jpg", -1)
                for image_id in outputs.input_batch["image_id"]
            ]
        )

    def _get_valids_mask(self, outputs: PredictionOutputs) -> Tensor:
        return self._get_permutation(outputs) >= 0

    def _get_relevance_labels(self, outputs: PredictionOutputs, query_indices: Tensor) -> Tensor:
        perm = self._get_permutation(outputs)
        perm = perm[perm >= 0]  # avoid padding items
        relevance_labels = self._relevance_scores[perm, :][:, perm]
        # query_padding = perm < 0
        return relevance_labels[query_indices] >= self.config.threshold


def _load_cxc_relevance(path: str) -> tuple[Tensor, dict[str, int]]:
    with fs.open(path) as fin:
        rows = list(csv.DictReader(fin))
    image_ids = sorted({row[k] for row in rows for k in ["image1", "image2"]})
    image_id_to_pos = {k: i for i, k in enumerate(image_ids)}

    n = len(image_ids)
    relevance_scores = np.zeros((n, n))
    for row in rows:
        i1 = image_id_to_pos[row["image1"]]
        i2 = image_id_to_pos[row["image2"]]
        relevance_scores[i1, i2] = row["agg_score"]
        relevance_scores[i2, i1] = row["agg_score"]

    return jnp.array(relevance_scores), image_id_to_pos


class KnnMetricCalculator(EmbeddingRetrievalMetricCalculator):
    """A metric calculator for K-NN accuracy.

    Uses approach described in https://arxiv.org/pdf/1805.01978.pdf (sec.3.4)
    """

    @config_class
    class Config(EmbeddingRetrievalMetricCalculator.Config):
        top_ks: tuple[int, ...] = (1, 20, 200)
        temps: tuple[float, ...] = (0.01, 0.03, 0.07)
        num_labels: Required[int] = REQUIRED

    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.metrics = []  # knn uses hardcoded metric (accuracy)
        return cfg

    def _get_chunk_metrics(
        self, outputs: PredictionOutputs, query_indices: Tensor
    ) -> dict[str, Tensor]:
        """Returns metrics for the given chunk of query indices."""

        cfg = self.config
        embeddings = outputs.predict_outputs["embedding"]
        if len(embeddings.shape) == 3:
            # Multi-stream models have dummy second dimension, remove it.
            embeddings = jnp.squeeze(embeddings, 1)

        labels = outputs.input_batch["label"]
        query_labels = labels[query_indices]

        index_mask = self._get_index_mask(outputs)
        index_labels = labels[index_mask]
        similarities = jnp.einsum(
            "i d, j d -> i j", embeddings[query_indices], embeddings[index_mask]
        )

        batch_segment_sum = jax.vmap(partial(jax.ops.segment_sum, num_segments=cfg.num_labels))

        chunk_metrics = {}
        for k in cfg.top_ks:
            top_similarities, top_indices = jax.lax.top_k(similarities, k)
            top_labels = jnp.take(index_labels, top_indices)
            for temp in cfg.temps:
                weights = jnp.exp(top_similarities / temp)
                label_scores = batch_segment_sum(weights, top_labels)
                correct = (jnp.argmax(label_scores, axis=-1) == query_labels).astype(
                    label_scores.dtype
                )
                chunk_metrics[f"knn-t{temp:.2f}@{k}"] = correct

        return chunk_metrics
