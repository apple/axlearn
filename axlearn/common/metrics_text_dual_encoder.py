# Copyright © 2023 Apple Inc.

"""Tests dual encoder text metrics."""
from typing import Dict, List, Optional

from jax import numpy as jnp

from axlearn.common.attention import NEG_INF
from axlearn.common.config import REQUIRED, Required, config_class
from axlearn.common.evaler import GlobalMetricCalculator, PredictionOutputs
from axlearn.common.loss import contrastive_logits
from axlearn.common.metrics_retrieval import (
    average_rank,
    calculate_mean_metrics,
    ndcg_at_k,
    top_k_accuracy,
)
from axlearn.common.text_dual_encoder import (
    FLATTENED_LEFT_EMBEDDINGS,
    FLATTENED_RIGHT_EMBEDDINGS,
    NEGATIVE_EMBEDDINGS,
    NEGATIVE_PADDINGS,
    POSITIVE_EMBEDDINGS,
    POSITIVE_PADDINGS,
    RIGHT_PADDINGS,
    flatten_and_concat_embeddings,
)
from axlearn.common.utils import Tensor


def _calculate_retrieval_metrics_from_embeddings(
    *,
    top_ks_for_accuracy: List[int],
    query_embeddings: Tensor,
    query_paddings: Tensor,
    text_positive_embeddings: Tensor,
    text_positive_paddings: Tensor,
    text_negative_embeddings: Optional[Tensor] = None,
    text_negative_paddings: Optional[Tensor] = None,
    top_ks_for_ndcg: Optional[List[int]] = None,
) -> Dict[str, Tensor]:
    """Main function to calculate all different retrieval metrics given embeddings for text dual
    encoder model.

    The retrieval is asymmetric meaning that relevant texts are retrieved given each valid query.

    Args:
        top_ks_for_accuracy: The values for k for which to compute accuracy.
        query_embeddings: Query embeddings with shape [num_queries, 1, dim].
        query_paddings: Query paddings with shape [num_queries, 1]. 1 means paddings and 0
            means valid queries.
        text_positive_embeddings: Positive text embeddings with shape
            [num_queries, max_positive_texts, dim].
        text_positive_paddings: Positive text paddings with shape
            [num_queries, max_positive_texts].
        text_negative_embeddings: Negative text embeddings with shape
            [num_queries, max_negative_texts, dim].
        text_negative_paddings: Negative text paddings with shape
            [num_queries, max_negative_texts].
        top_ks_for_ndcg: Optional. The values for k for which to compute nDCG metrics.
            If not None, return nDCG for specified k values.

    Returns:
        A dict containing all different metrics for text dual encoder model.
    """
    flattened_embeddings_and_paddings = flatten_and_concat_embeddings(
        left_positive_embeddings=query_embeddings,
        right_positive_embeddings=text_positive_embeddings,
        right_positive_paddings=text_positive_paddings,
        right_negative_embeddings=text_negative_embeddings,
        right_negative_paddings=text_negative_paddings,
    )
    flattened_query_embeddings = flattened_embeddings_and_paddings[FLATTENED_LEFT_EMBEDDINGS]
    flattened_text_embeddings = flattened_embeddings_and_paddings[FLATTENED_RIGHT_EMBEDDINGS]
    text_paddings = flattened_embeddings_and_paddings[RIGHT_PADDINGS]

    # Shape: [num_queries, num_queries * (max_positive_texts + max_negative_texts)].
    sim = contrastive_logits(flattened_query_embeddings, flattened_text_embeddings)
    return calculate_retrieval_metrics_from_similarity_matrix(
        sim=sim,
        text_positive_paddings=text_positive_paddings,
        query_paddings=query_paddings,
        text_paddings=text_paddings,
        top_ks_for_accuracy=top_ks_for_accuracy,
        top_ks_for_ndcg=top_ks_for_ndcg,
    )


def calculate_retrieval_metrics_from_similarity_matrix(
    *,
    sim: Tensor,
    text_positive_paddings: Tensor,
    query_paddings: Tensor,
    text_paddings: Tensor,
    top_ks_for_accuracy: List[int],
    top_ks_for_ndcg: Optional[List[int]] = None,
) -> Dict[str, Tensor]:
    """Function to calculate all different retrieval metrics given similarity matrix.

    The retrieval is asymmetric meaning that relevant texts are retrieved given each valid query.

    Args:
        sim: Similarity matrix with shape
            [num_queries, num_queries * (max_positive_texts + max_negative_texts)].
        text_positive_embeddings: Positive text embeddings with shape
            [num_queries, max_positive_texts, dim].
        query_paddings: Query paddings with shape [num_queries, 1]. 1 means paddings and 0
            means valid queries.
        text_paddings: Text paddings with shape
            [num_queries * (max_positive_texts + max_negative_texts)]. 1 means paddings and 0 means
            valid queries.
        top_ks_for_accuracy: The values for k for which to compute accuracy.
        top_ks_for_ndcg: Optional. The values for k for which to compute nDCG.
            If not None, return nDCG for specified k values.

    Returns:
        A dict containing all different metrics for text dual encoder model.
    """
    num_queries, max_positive_texts = text_positive_paddings.shape
    total_num_positive_texts = num_queries * max_positive_texts
    # Shape: [num_queries, max_positive_texts].
    # gt_targets[i][j] = i * max_positive_texts + j.
    gt_targets = jnp.arange(0, total_num_positive_texts, dtype=jnp.int32).reshape(
        num_queries, max_positive_texts
    )

    # Computes accuracy@K.
    gt_targets_for_top_k_accuracy = (
        gt_targets * (1 - text_positive_paddings) - text_positive_paddings
    )

    retrieval_accuracy_at_k = top_k_accuracy(
        sim,
        gt_targets_for_top_k_accuracy,
        top_ks_for_accuracy,
        similarity_bias=text_paddings * NEG_INF,
    )

    # Computes average_rank.
    relevance_labels = jnp.zeros_like(sim)
    # Shape: [num_queries, num_queries * (max_positive_texts + max_negative_texts)].
    # relevance_labels[i][j] = 1 iff the j-th text is relevant for query i.
    relevance_labels = relevance_labels.at[
        jnp.expand_dims(jnp.arange(num_queries), 1), gt_targets
    ].set(1)
    relevance_labels = relevance_labels * (1 - text_paddings)
    sim_with_masks = sim + text_paddings * NEG_INF
    avg_rank = average_rank(
        scores=sim_with_masks, relevance_labels=relevance_labels, query_padding=query_paddings
    )["avg_rank"]

    metrics = {}

    if top_ks_for_ndcg:
        # Used to mask out queries which don't have any relevant items.
        no_relevant_item_query_mask = relevance_labels.max(axis=-1) == 0
        ndcg_metrics = ndcg_at_k(
            scores=sim_with_masks, relevance_labels=relevance_labels, top_ks=top_ks_for_ndcg
        )
        for k, ndcg_metric in ndcg_metrics.items():
            # Calculate average nDCG at each postion k.
            metrics.update(
                calculate_mean_metrics(
                    metric_name=f"ndcg@{k}",
                    query_metrics=ndcg_metric,
                    query_padding=jnp.logical_or(
                        query_paddings.reshape(-1), no_relevant_item_query_mask
                    ),
                )
            )

    # Shape: [num_queries].
    flattened_valid_queries = 1 - jnp.reshape(query_paddings, -1)
    num_valid_queries = jnp.sum(flattened_valid_queries)

    for idx, k in enumerate(top_ks_for_accuracy):
        metrics[f"retrieval_accuracy@{k}"] = (
            jnp.sum(retrieval_accuracy_at_k[idx] * flattened_valid_queries) / num_valid_queries
        )
    metrics["avg_rank"] = avg_rank

    return metrics


class TextDualEncoderMetricCalculator(GlobalMetricCalculator):
    """A metric calculator for text dual encoder model.

    Currently it supports following metrics:
        retrieval_accuracy@K: Accuracy@K of highest-ranked positive text predicted by model.
        avg_rank: Computes average rank of each query's first relevant text.
        ndcg@k: Average of nDCG@K for each query if top_ks_for_ndcg is set.
    """

    @config_class
    class Config(GlobalMetricCalculator.Config):
        # Name of text dual encoder's left encoder.
        left_encoder_name: Required[str] = REQUIRED
        # Name of text dual encoder's right encoder.
        right_encoder_name: Required[str] = REQUIRED
        # The values for k for which to compute accuracy.
        top_ks_for_accuracy: Required[List[int]] = REQUIRED
        # The values for k for which to compute nDCG.
        top_ks_for_ndcg: Optional[List[int]] = None

    def _calculate_metrics(self, outputs: PredictionOutputs) -> Dict[str, Tensor]:
        cfg = self.config
        predict_outputs, input_batch = outputs.predict_outputs, outputs.input_batch
        text_negative_embeddings = predict_outputs[cfg.right_encoder_name].get(
            NEGATIVE_EMBEDDINGS, None
        )
        metrics = _calculate_retrieval_metrics_from_embeddings(
            top_ks_for_accuracy=cfg.top_ks_for_accuracy,
            top_ks_for_ndcg=cfg.top_ks_for_ndcg,
            query_embeddings=predict_outputs[cfg.left_encoder_name][POSITIVE_EMBEDDINGS],
            query_paddings=input_batch[cfg.left_encoder_name][POSITIVE_PADDINGS],
            text_positive_embeddings=predict_outputs[cfg.right_encoder_name][POSITIVE_EMBEDDINGS],
            text_positive_paddings=input_batch[cfg.right_encoder_name][POSITIVE_PADDINGS],
            text_negative_embeddings=text_negative_embeddings,
            text_negative_paddings=input_batch[cfg.right_encoder_name][NEGATIVE_PADDINGS]
            if text_negative_embeddings is not None
            else None,
        )

        formatted_metrics = {}
        for name, val in metrics.items():
            formatted_metrics[self.formatted_metric_name(name)] = val

        return formatted_metrics
