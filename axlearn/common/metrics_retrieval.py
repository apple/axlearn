# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# scikit-learn/scikit-learn:
# Copyright (c) 2007-2023 The scikit-learn developers. All rights reserved.
# Licensed under BSD 3 clause.

"""Retrieval metrics."""
from typing import Optional

import jax
import jax.numpy as jnp

from axlearn.common.utils import Tensor, cast_floats


def top_k_accuracy(
    sim: Tensor,
    gt_targets: Optional[Tensor],
    top_ks: list[int],
    similarity_bias: Tensor = None,
    relevance_labels: Optional[Tensor] = None,
    return_counts: bool = False,
) -> Tensor:
    """Compute Top@K accuracy.

    Args:
        sim: The similarity logits between each source and target. Shape: [M, N]. dtype: float.
        gt_targets: The gt target for the source embeddings.
            Shape: [M, max_num_gt_target_per_instance]. dtype: int32.
            The max_num_gt_target_per_instance is the maximum number of groundtruth
                targets per instance.
            For example, in COCO image to text retrieval. One image is aligned
                with 5 texts. The max_num_gt_target_per_instance = 5
                    correct_targets can be [[0, 1, 2, 3, 4],
                                            [5, 6, 7, 8, -1]]
                    This means the first image is aligned to text_id=0 to text_id=4.
                    The second image is aligned to text_id=5 to text_id=8.
                    As the second image is only aligned to 4 texts (not 5).
                    -1 is added for padding purpose.
                In COCO text to image retrieval. One text is aligned with 1 image.
                    The max_num_gt_target_per_instance = 1
                    correct_targets can be [[0], [0], [0],
                                            [1], [1], [1]]
                    This means the first three text is aligned to the first image.
                    The last three text is aligned to the second image.

        top_ks: Compute accuracy @ k for each of the values of k provided in this list.
        similarity_bias: A Tensor with representing the bias to the similarity.
            If shape == [M, N],  similarity_bias[i,j] = NEG.INF means the j-th target is masked for
                i-th source during the top_k selection.
            If Shape == [N], similarity_bias[j] = NEG.INF means the j-th target is masked for
                any source during the top_k selection.
        relevance_labels: Optional 0/1 tensor with the same shape as sim.
            relevance_labels[i, j] = 1 iff the j-th item is relevant for query i. Used
            instead of `gt_targets` if provided.
        return_counts: A boolean,
            if True return how many true target was in the top_k predicted targets,
            else return whether the true target was in the top_k predicted targets.
            By default, is False.

    Returns:
        top_ks_correct: A float32 Tensor indicating whether (or how many) the true target
            was in the top_k predicted targets. Shape [len(top_k), M,].


    Raises:
        ValueError: if neither `gt_targets` nor `relevance_labels` were provided.
    """
    sim = cast_floats(sim, jnp.float32)
    # The similarity between each source and target. Shape: [M, N].
    if similarity_bias is None:
        similarity_bias = jnp.zeros(sim.shape[1], dtype=sim.dtype)
    if similarity_bias.shape != sim.shape:
        similarity_bias = similarity_bias.reshape(1, -1)
    assert (
        similarity_bias.shape[1] == sim.shape[1]
    ), f"similarity_bias.shape={similarity_bias.shape}, but sim.shape={sim.shape}"
    sim = sim + similarity_bias

    if gt_targets is None and relevance_labels is None:
        raise ValueError("Either `gt_targets` or `relevance_labels` is required.")

    # express_mode is used for the scenario where only one positive gt_target exists:
    # gt_target.shape = [M, 1].
    # In this case, if M and N are very large (e.g. M=50000, N=1000).
    # The jax.nn.one_hot is very slow.
    # express_mode is used for solving this issue.
    express_mode = False
    if gt_targets is not None and gt_targets.shape[1] == 1:
        express_mode = True
    elif relevance_labels is None:
        # Shape [M, N]. gt_targets[i, j] > 0 if target j is one of the groundtruth
        # targets for source i.
        relevance_labels = jnp.sum(jax.nn.one_hot(gt_targets, sim.shape[1]), 1)

    top_ks_correct = []
    for k in top_ks:
        # Indices corresponding to the top k targets. Shape: [M, top_k].
        indices = jax.lax.top_k(sim, k)[1]
        # in_retrieval_top_k has shape [M, N].
        if express_mode:
            correct = jnp.sum(indices == gt_targets, 1)
        else:
            correct = jnp.sum(jnp.take_along_axis(relevance_labels, indices, axis=1), -1)
        top_ks_correct.append(correct)
    top_ks_correct = jnp.stack(top_ks_correct)

    if not return_counts:
        top_ks_correct = (top_ks_correct > 0).astype(jnp.float32)
    return top_ks_correct


def top_k_recall(
    sim: Tensor,
    gt_targets: Optional[Tensor],
    top_ks: list[int],
    similarity_bias: Optional[Tensor] = None,
    relevance_labels: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute Top@K recall.

    Recall@K = {# of relevant docs in top k retrieved docs} / {min(K, # of total relevant docs)}

    Args:
        sim: The similarity logits between each source and target. Shape: [M, N]. dtype: float.
            See `sim` in `top_k_accuracy` for more details.
        gt_targets: The gt target for the source embeddings.
            Shape: [M, max_num_gt_target_per_instance]. dtype: int32.
            See `gt_targets` in `top_k_accuracy` for more details.
        top_ks: Compute recall @ k for each of the values of k provided in this list.
        similarity_bias: A Tensor representing the bias to the similarity.
            See `similarity_bias` in `top_k_accuracy` for more details.
        relevance_labels: Optional 0/1 tensor with the same shape as sim.
            If None, infer from gt_targets.
            See `relevance_labels` in `top_k_accuracy` for more details.

    Returns:
        top_ks_recall: A float32 Tensor indicating recall at k. Shape [len(top_k), M,].

    Raises:
        ValueError: if neither `gt_targets` nor `relevance_labels` were provided.
    """
    if gt_targets is None and relevance_labels is None:
        raise ValueError("Either `gt_targets` or `relevance_labels` is required.")

    total_num_relevant_items = None
    # Try to get relevance labels first to avoid double conversion in top_k_accuracy.
    if relevance_labels is None:
        assert gt_targets is not None
        if gt_targets.shape[1] == 1:
            # Leave relevance_labels as None to enable express mode in `top_k_accuracy`.
            total_num_relevant_items = jnp.where(gt_targets > 0, 1, 0)
            total_num_relevant_items = jnp.squeeze(total_num_relevant_items, 1)
        else:
            # Shape [M, N]. gt_targets[i, j] > 0 if target j is one of the groundtruth
            # targets for source i.
            relevance_labels = jnp.sum(jax.nn.one_hot(gt_targets, sim.shape[1]), 1)
    if total_num_relevant_items is None:
        # Get total_num_relevant_items for non-express mode.
        # Shape: [num_queries].
        assert relevance_labels is not None
        total_num_relevant_items = jnp.sum(relevance_labels, axis=-1)
    # Compute the number of correct items at k for each query.
    correct_at_k = top_k_accuracy(
        sim=sim,
        relevance_labels=relevance_labels,
        top_ks=top_ks,
        similarity_bias=similarity_bias,
        gt_targets=gt_targets,
        return_counts=True,
    )
    top_ks_recall = []
    for k, correct_metrics in zip(top_ks, correct_at_k):
        denorm = jnp.maximum(jnp.minimum(total_num_relevant_items, k), 1.0)
        top_ks_recall.append(correct_metrics / denorm)
    top_ks_recall = jnp.stack(top_ks_recall)
    return top_ks_recall


def _reciprocal_rank(
    *,
    scores: Tensor,
    relevance_labels: Tensor,
    query_padding: Tensor,
) -> tuple[Tensor, Tensor]:
    """Computes reciprocal rank of each query's first relevant item and updated query_padding that
    excludes query without any relevant item.

    Args:
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        query_padding: 0/1 tensor with shape [num_queries, 1]. 1 means paddings and 0 means valid
            queries.

    Returns:
        reciprocal_rank: Reciprocal rank of each query's first relevant item shape [num_queries,].
        query_padding: Updated query_padding that excludes query without any relevant item.
    """
    num_queries, num_items = scores.shape

    # Shape: [num_queries, num_items].
    # Sort in decreasing order.
    indices = jnp.argsort(-scores)
    # Shape: [num_queries, num_items].
    # relevance_labels_sorted_by_scores[i, j] = 1 iff the j-th ranked item is relevant for query i.
    relevance_labels_sorted_by_scores = relevance_labels[
        jnp.expand_dims(jnp.arange(num_queries), 1), indices
    ]
    # Shape: [num_queries,].
    # no_relevant_item_query_mask[i] = True iff the i-th query doesn't have relevant item.
    no_relevant_item_query_mask = relevance_labels.max(axis=-1) == 0
    item_rank = jnp.arange(1, num_items + 1)
    # Shape: [num_queries,].
    reciprocal_rank = jnp.max(1.0 / item_rank * relevance_labels_sorted_by_scores, axis=-1)
    query_padding = jnp.logical_or(query_padding.reshape(-1), no_relevant_item_query_mask)
    return reciprocal_rank, query_padding


def mean_reciprocal_rank(
    *,
    scores: Tensor,
    relevance_labels: Tensor,
    query_padding: Tensor,
) -> dict[str, Tensor]:
    """Computes mean reciprocal rank (MRR) of each query's first relevant item.

    Padded query or query without any relevant item will be excluded when averaging.

    Args:
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        query_padding: 0/1 tensor with shape [num_queries, 1]. 1 means paddings and 0 means valid
            queries.

    Returns:
        A dict {"mrr": mrr}.
    """
    reciprocal_rank, query_padding = _reciprocal_rank(
        scores=scores, relevance_labels=relevance_labels, query_padding=query_padding
    )
    return calculate_mean_metrics(
        metric_name="mrr",
        query_metrics=reciprocal_rank,
        query_padding=query_padding,
    )


def average_rank(
    *,
    scores: Tensor,
    relevance_labels: Tensor,
    query_padding: Tensor,
) -> dict[str, Tensor]:
    """Computes average rank of each query's first relevant item.

    Padded query or query without any relevant item will be excluded when averaging.

    Args:
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        query_padding: 0/1 tensor with shape [num_queries, 1]. 1 means paddings and 0 means valid
            queries.

    Returns:
        A dict {"avg_rank": avg_rank}.
    """
    reciprocal_rank, query_padding = _reciprocal_rank(
        scores=scores, relevance_labels=relevance_labels, query_padding=query_padding
    )

    rank = jnp.where(
        reciprocal_rank == 0,
        0.0,
        1.0 / reciprocal_rank,
    )
    return calculate_mean_metrics(
        metric_name="avg_rank",
        query_metrics=rank,
        query_padding=query_padding,
    )


def average_precision_at_k(
    scores: Tensor,
    relevance_labels: Tensor,
    top_ks: list[int],
) -> dict[int, Tensor]:
    """Computes Average Precision@K (AP@K) metrics.

    AP@K = sum_{k=1}^{K}(Precision@k * rel(k)) / min(K, total_num_relevant_items)
    where rel(k) = 1 if the k-th item sorted by scores in decreasing order is relevant otherwise 0.

    Args:
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        top_ks: List of Ks to compute AP@K with. -1 means AP@num_items, i.e., looking at all
            candidate items sorted by scores.

    Returns:
        ap_at_k: A dict having each of K in top_ks as keys and AP@K list with a length of
            num_queries containing AP@K for each query as values.
    """
    num_queries, num_items = scores.shape
    # The computation of AP@K is accumulative so we obtain all other AP@Ks for free when we get that
    # for the biggest K value.
    max_k = max(top_ks) if -1 not in top_ks else num_items
    assert 0 < max_k <= num_items

    # Shape: [num_queries, max_k].
    _, indices = jax.lax.top_k(scores, max_k)
    # Shape: [num_queries, max_k].
    # relevance_labels_in_top_max_k[i, j] = 1 iff the j-th ranked item is relevant for query i.
    relevance_labels_in_top_max_k = relevance_labels[
        jnp.expand_dims(jnp.arange(num_queries), 1), indices
    ]
    # Denominator when calculating Precision@k, where k=1,2,...,max_k.
    item_rank = jnp.arange(1, max_k + 1)
    # Shape: [num_queries, max_k].
    precision_at_k_list = jnp.cumsum(relevance_labels_in_top_max_k, axis=-1) / item_rank
    # Shape: [num_queries, 1].
    total_num_relevant_items = jnp.sum(relevance_labels, axis=-1, keepdims=True)
    # Shape: [num_queries, max_k].
    min_of_k_and_total_num_relevant_items = jnp.minimum(
        jnp.tile(item_rank, (num_queries, 1)), total_num_relevant_items
    )
    # sum_{k=1}^{K}(Precision@k * rel(k)) / min(K, total_num_relevant_items)
    # Average precision will be defined as 0.0 when there is no relevant item for that query.
    # Shape: [num_queries, max_k].
    ap_list = jnp.where(
        min_of_k_and_total_num_relevant_items == 0,
        0.0,
        jnp.cumsum(precision_at_k_list * relevance_labels_in_top_max_k, axis=-1)
        / min_of_k_and_total_num_relevant_items,
    )

    ap_at_k = {}
    for k in top_ks:
        if k == -1:
            ap_at_k[k] = ap_list[:, k]
        else:
            ap_at_k[k] = ap_list[:, k - 1]
    return ap_at_k


def _tie_averaged_dcg(*, y_true: Tensor, y_score: Tensor, discount_factor: Tensor) -> Tensor:
    """Computes tie-aware DCG by averaging over possible permutations of ties.

    DCG@K(gains) = sum_{i=1}^{max_k} gain(i) * discount_factor(i)

    where:
        * y_score is divided into different equivalence classes,
            with each equivalence class having a unique score.
        * gain(i) is the average gain for all items within the same equivalence class as item i,
            so gain(i) = sum_{j ∈ I_c} gain(j) / n_c where item i belongs to group c and
            I_c is the indices of all items that belongs to group c, with n_c items in the class.
            gain(i) and gain(j) mean y_true_i or y_true_j.
            Note in this context indices are 1-indexed.

    Ref (Sect. 2.6):
    https://www.microsoft.com/en-us/research/publication/computing-information-retrieval-performance-measures-efficiently-in-the-presence-of-tied-scores
    Implementation Ref:
    https://github.com/scikit-learn/scikit-learn/blob/cb15a82e6439feda50b0605d70ce6d06c2eac7fd/sklearn/metrics/_ranking.py#L1462-L1507

    Args:
        y_true: Float tensor of shape [num_items] where i is the relevance score (gain) of the
            i-th item, where all values must be >=0, and a 0 value can represent a padding item.
        y_score: Predicted relevance scores with shape [num_items]. Users could mask y_score[i]
            with NEG_INF such that the i-th item is masked.
            The corresponding y_true[i] for a masked item must be 0.
        discount_factor: Float tensor of shape [max_k] where element i
            describes the discount factor the gain of item i is multiplied by.

    Returns:
        A tensor of shape [max_k] where the k-th value (1-indexed) represents the tie-aware DCG@k.
    """
    max_k = discount_factor.shape[0]
    # Get counts of unique scores and indices to the unique scores to restore the
    # original sorted scores.
    # inv shape: [num_items].
    # counts shape: [max_k].
    # We care about the counts of up to and including the first max_k equivalence classes.
    # If there are fewer than max_k classes, the counts of the extra classes are 0s.
    _, inv, counts = jnp.unique(-y_score, return_inverse=True, return_counts=True, size=max_k)
    # Get average gain for each equivalence class.
    # Average gain is calculated from all items that belongs to the same class
    # even if when max_k < num_items.
    # Shape: [max_k].
    group_gains = jnp.zeros(max_k)
    group_gains = group_gains.at[inv].add(y_true)
    group_gains = group_gains / jnp.maximum(counts, 1)
    # Repeat each avg. gain by number of occurrences of the score in each equivalence class.
    repeated_group_gains = jnp.repeat(group_gains, counts, total_repeat_length=max_k)
    # Use cumsum to get the DCG@k for each cutoff k.
    return jnp.cumsum(repeated_group_gains * discount_factor)


def ndcg_at_k(
    scores: Tensor,
    relevance_labels: Tensor,
    top_ks: list[int],
    ignore_ties: bool = True,
) -> dict[int, Tensor]:
    """Computes Normalized Discounted Cumulative Gain@K (NDCG@K) metrics.

    When ignoring ties:
        NDCG@K = DCG@K(relevance_labels_sorted_by_scores) / IDCG@K where
        DCG@K(gains) = sum_{i=1}^{K} gains_i / log2(i+1) and
        IDCG@K = DCG@K(sorted_relevance_labels).

    For tie-aware NDCG, DCG is computed using _tie_averaged_dcg.

    Args:
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
            The corresponding relevance_labels[i, j] for a masked item must be 0.
        relevance_labels: Float tensor with the same shape as scores. relevance_labels[i, j]
            describes the relevance score (gain) between the j-th item and i-th query, where all
            values must be >=0, and a 0 value can represent a padding item.
        top_ks: List of Ks to compute NDCG@K with. -1 means NDCG@num_items, i.e., looking at all
            candidate items sorted by scores.
        ignore_ties: If true, assume that there are no ties in scores for efficiency. If false,
            compute tie-aware NDCG by averaging over possible permutations of ties.

    Returns:
        A dict having each of K in top_ks as keys and NDCG@K list with a length of num_queries
        containing NDCG@K for each query as values.
    """
    num_queries, num_items = scores.shape
    # The computation of NDCG@K is accumulative so we obtain all other NDCG@Ks for free when we get
    # that for the biggest K value.
    max_k = max(top_ks) if -1 not in top_ks else num_items
    assert 0 < max_k <= num_items
    discount_factors = 1 / jnp.log2(jnp.arange(2, max_k + 2))

    if ignore_ties:
        # Shape: [num_queries, max_k].
        _, indices_of_sorted_scores = jax.lax.top_k(scores, max_k)
        # Shape: [num_queries, max_k].
        relevance_labels_sorted_by_scores = relevance_labels[
            jnp.expand_dims(jnp.arange(num_queries), 1), indices_of_sorted_scores
        ]

        # Shape: [num_queries, max_k].
        _, indices_of_sorted_relevance_labels = jax.lax.top_k(relevance_labels, max_k)
        # Shape: [num_queries, max_k].
        sorted_relevance_labels = relevance_labels[
            jnp.expand_dims(jnp.arange(num_queries), 1),
            indices_of_sorted_relevance_labels,
        ]

        # Shape: [num_queries, max_k].
        dcg = jnp.cumsum(relevance_labels_sorted_by_scores * discount_factors, axis=-1)
        # Shape: [num_queries, max_k].
        idcg = jnp.cumsum(sorted_relevance_labels * discount_factors, axis=-1)
    else:
        auto_batch_tie_averaged_dcg = jax.vmap(_tie_averaged_dcg)
        discount_factors = jnp.tile(discount_factors, reps=(num_queries, 1))
        dcg = auto_batch_tie_averaged_dcg(
            y_true=relevance_labels, y_score=scores, discount_factor=discount_factors
        )
        idcg = auto_batch_tie_averaged_dcg(
            y_true=relevance_labels,
            y_score=relevance_labels,
            discount_factor=discount_factors,
        )

    ndcg = jnp.where(idcg == 0, 0.0, dcg / idcg)
    metrics = {}
    for k in top_ks:
        if k == -1:
            metrics[k] = ndcg[:, k]
        else:
            metrics[k] = ndcg[:, k - 1]

    return metrics


def calculate_mean_average_precision_metrics(
    *,
    top_ks: list[int],
    scores: Tensor,
    relevance_labels: Tensor,
    query_padding: Tensor,
    categories: Optional[Tensor] = None,
    categories_names: Optional[tuple[str, ...]] = None,
) -> dict[str, Tensor]:
    """Calculates mean average precision at k (MAP@k) metrics.

    MAP@K = 1/n * sum(AP@K)
    where AP@K = sum_{k=1}^{K}(Precision@k * rel(k)) / min(K, total_num_relevant_items)
    where rel(k) = 1 if the k-th item sorted by scores in decreasing order is relevant otherwise 0.

    Args:
        top_ks: A list of K's for top-k stats.
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        query_padding: Bool tensor [num_queries] with True values representing padding.
        categories: Optional [num_queries] int tensor with examples category ids. Used for
            breaking down metrics by queries category.
        categories_names: Optional tuple of cateogory names - category `i` gets name
            `categories_names[i]`.

    Returns:
        A dict containing "MAP@{k}" - mean average precision at top-k and
        "MAP@{k}_{category_name}" for per-category values if categories were provided.

    Raises:
        ValueError: if categories_names weren't provided, but categories were.
    """
    # Compute per-query metrics.
    ap_at_k = average_precision_at_k(
        scores=scores, relevance_labels=relevance_labels, top_ks=top_ks
    )
    metrics = {}
    for k, query_metrics in ap_at_k.items():
        metrics.update(
            calculate_mean_metrics(
                metric_name=metric_at_k_name("MAP", k),
                query_metrics=query_metrics,
                query_padding=query_padding,
                query_categories=categories,
                categories_names=categories_names,
            )
        )
    return metrics


def calculate_accuracy_metrics(
    *,
    top_ks: list[int],
    scores: Tensor,
    relevance_labels: Tensor,
    query_padding: Tensor,
    categories: Optional[Tensor] = None,
    categories_names: Optional[tuple[str, ...]] = None,
) -> dict[str, Tensor]:
    """Calculates accuracy at k (accuracy@k) metrics.

    Accuracy@k equals to a ratio of queries with at least one relevant result in the top k results.

    Args:
        top_ks: A list of K's for top-k stats.
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        query_padding: Bool tensor [num_queries] with True values representing padding.
        categories: Optional [num_queries] int tensor with examples category ids. Used for
            breaking down metrics by queries category.
        categories_names: Optional tuple of cateogory names - category `i` gets name
            `categories_names[i]`.

    Returns:
        A dict containing "accuracy@{k}" - retrieval accuracy at top-k and
        "accuracy@{k}_{category_name}" for per-category values if categories were provided.

    Raises:
        ValueError: if categories_names weren't provided, but categories were.
    """
    # Compute per-query metrics.
    accuracy_at_k = top_k_accuracy(
        sim=scores, relevance_labels=relevance_labels, top_ks=top_ks, gt_targets=None
    )
    metrics = {}
    for k, query_metrics in zip(top_ks, accuracy_at_k):
        metrics.update(
            calculate_mean_metrics(
                metric_name=metric_at_k_name("accuracy", k),
                query_metrics=query_metrics,
                query_padding=query_padding,
                query_categories=categories,
                categories_names=categories_names,
            )
        )
    return metrics


def calculate_recall_metrics(
    *,
    top_ks: list[int],
    scores: Tensor,
    relevance_labels: Tensor,
    query_padding: Tensor,
    categories: Optional[Tensor] = None,
    categories_names: Optional[tuple[str, ...]] = None,
) -> dict[str, Tensor]:
    """Calculates recall at k (accuracy@k) metrics.

    Recall@k equals to average of recall over queries.

    For each query, recall@K =
        {# of relevant docs in top k retrieved docs} / {min(K, # of total relevant docs)}

    Args:
        top_ks: A list of K's for top-k stats.
        scores: Predicted relevance scores between queries and items with shape
            [num_queries, num_items]. Users could mask scores[i, j] with NEG_INF such that the j-th
            item is masked for i-th query.
        relevance_labels: 0/1 Tensor with the same shape as scores. relevance_labels[i, j] = 1
            iff the j-th item is relevant for query i.
        query_padding: Bool tensor [num_queries] with True values representing padding.
        categories: Optional [num_queries] int tensor with examples category ids. Used for
            breaking down metrics by queries category.
        categories_names: Optional tuple of cateogory names - category `i` gets name
            `categories_names[i]`.

    Returns:
        A dict containing "recall@{k}" - retrieval recall at top-k and
        "recall@{k}_{category_name}" for per-category values if categories were provided.

    Raises:
        ValueError: if categories_names weren't provided, but categories were.
    """
    # Compute per-query metrics.
    recall_at_k = top_k_recall(
        sim=scores, relevance_labels=relevance_labels, top_ks=top_ks, gt_targets=None
    )
    metrics = {}
    for k, query_metrics in zip(top_ks, recall_at_k):
        metrics.update(
            calculate_mean_metrics(
                metric_name=metric_at_k_name("recall", k),
                query_metrics=query_metrics,
                query_padding=query_padding,
                query_categories=categories,
                categories_names=categories_names,
            )
        )
    return metrics


def metric_at_k_name(name: str, k: int):
    return name if k < 0 else f"{name}@{k}"


def calculate_mean_metrics(
    *,
    metric_name: str,
    query_metrics: Tensor,
    query_padding: Tensor,
    query_categories: Optional[Tensor] = None,
    categories_names: Optional[tuple[str, ...]] = None,
) -> dict[str, Tensor]:
    num_valid_or_one = jnp.maximum((1 - query_padding).sum(), 1.0)
    query_metrics = jnp.where(query_padding, 0.0, query_metrics)
    metrics = {metric_name: query_metrics.sum() / num_valid_or_one}
    if query_categories is not None:
        if categories_names is None:
            raise ValueError("`categories_names` is required when categoires were provided.")
        ap_sum_per_category = jax.ops.segment_sum(
            query_metrics, query_categories, num_segments=len(categories_names)
        )
        num_per_category = jax.ops.segment_sum(
            1 - query_padding.astype(jnp.int32),
            query_categories,
            num_segments=len(categories_names),
        )
        per_category_metrics = []
        for i, category_name in enumerate(categories_names):
            # Do not consider the category which doesn't have any query.
            if num_per_category[i] == 0:
                continue
            per_category_metric = ap_sum_per_category[i] / jnp.maximum(num_per_category[i], 1)
            metrics[f"{metric_name}_{category_name}"] = per_category_metric
            per_category_metrics.append(per_category_metric)
        metrics[f"{metric_name}_avg_category"] = jnp.mean(jnp.array(per_category_metrics))
    return metrics
