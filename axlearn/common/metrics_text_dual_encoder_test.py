# Copyright © 2023 Apple Inc.

"""Dual encoder text metrics."""
# pylint: disable=pointless-string-statement, duplicate-code
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax.experimental.pjit import pjit

from axlearn.common.eval_retrieval_test import DummyRetrievalModel
from axlearn.common.metrics_text_dual_encoder import (
    TextDualEncoderMetricCalculator,
    calculate_retrieval_metrics_from_similarity_matrix,
)
from axlearn.common.test_utils import TestCase
from axlearn.common.text_dual_encoder import (
    NEGATIVE_EMBEDDINGS,
    NEGATIVE_PADDINGS,
    POSITIVE_EMBEDDINGS,
    POSITIVE_PADDINGS,
)
from axlearn.common.utils import Tensor

BATCH_SIZE = 2

LEFT_ENCODER_NAME = "query_encoder"
RIGHT_ENCODER_NAME = "passage_encoder"


def dummy_data_generator(
    query_embeddings: Tensor,
    query_paddings: Tensor,
    text_positive_embeddings: Tensor,
    text_positive_paddings: Tensor,
    text_negative_embeddings: Optional[Tensor] = None,
    text_negative_paddings: Optional[Tensor] = None,
):
    for batch_idx in range(int(np.ceil(len(query_embeddings) / BATCH_SIZE))):
        start, end = batch_idx * BATCH_SIZE, (batch_idx + 1) * BATCH_SIZE
        yield {
            LEFT_ENCODER_NAME: {
                POSITIVE_PADDINGS: query_paddings[start:end],
            },
            RIGHT_ENCODER_NAME: {
                POSITIVE_PADDINGS: text_positive_paddings[start:end],
                NEGATIVE_PADDINGS: text_negative_paddings[start:end]
                if text_negative_paddings is not None
                else None,
            },
            "predicted": {
                LEFT_ENCODER_NAME: {
                    POSITIVE_EMBEDDINGS: query_embeddings[start:end],
                },
                RIGHT_ENCODER_NAME: {
                    POSITIVE_EMBEDDINGS: text_positive_embeddings[start:end],
                    NEGATIVE_EMBEDDINGS: text_negative_embeddings[start:end]
                    if text_negative_embeddings is not None
                    else None,
                },
            },
        }


def _compute_metrics(
    *,
    data_generator,
    top_ks_for_accuracy: List[int],
    top_ks_for_ndcg: Optional[List[int]] = None,
) -> Dict:
    """Mock function that calls forward() function of TextDualEncoderMetricCalculator and
        calculates evaluation metrics.

    Args:
        data_generator: Mock data generator that generates batches of input data and corresponding
            predicted embeddings.
        top_ks_for_accuracy: The values for k for which to compute accuracy.
        top_ks_for_ndcg: Optional. The values for k for which to compute nDCG.

    Returns:
        A dict containing all the calculated metrics.
    """
    # pylint: disable=duplicate-code
    with jax.sharding.Mesh(
        jax.experimental.mesh_utils.create_device_mesh((1, 1)), ("data", "model")
    ):
        model = DummyRetrievalModel.default_config().set(name="model").instantiate(parent=None)
        model_param_partition_specs = jax.tree_util.tree_map(
            lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
        )
        calculator: TextDualEncoderMetricCalculator = (
            TextDualEncoderMetricCalculator.default_config()
            .set(
                name="calculator",
                left_encoder_name=LEFT_ENCODER_NAME,
                right_encoder_name=RIGHT_ENCODER_NAME,
                top_ks_for_accuracy=top_ks_for_accuracy,
                top_ks_for_ndcg=top_ks_for_ndcg,
            )
            .instantiate(
                parent=None, model=model, model_param_partition_specs=model_param_partition_specs
            )
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
    # pylint: enable=duplicate-code


class TextDualEncoderMetricCalculatorTest(TestCase):
    @parameterized.parameters(dict(top_ks_for_ndcg=[1, 4]), dict(top_ks_for_ndcg=None))
    def test_calculate_metrics(self, top_ks_for_ndcg):
        query_embeddings = jnp.asarray(
            [
                [[4, 1]],
                [[6, -1]],
                [[3, 2]],
                [[1, 3]],
            ],
            dtype=jnp.float32,
        )
        query_paddings = jnp.asarray([[0], [0], [0], [1]])
        text_positive_embeddings = jnp.asarray(
            [
                [[4, 1], [1, 3]],
                [[1, 5], [1, 1]],
                [[1, 0], [4, 3]],
                [[1, -1], [2, 6]],
            ],
            dtype=jnp.float32,
        )
        text_positive_paddings = jnp.asarray([[0, 0], [0, 0], [0, 0], [0, 1]])
        """
        similarity matrix:
        [[17  7  9  5  4 19  3 14]
        [23  3  1  5  6 21  7  6]
        [14  9 13  5  3 18  1 18]
        [ 7 10 16  4  1 13 -2 20]]
        """
        # Tests when there is no negative text. text_negative_embeddings and
        # text_negative_paddings are set as None.
        text_negative_embeddings = None
        text_negative_paddings = None
        top_ks_for_accuracy = [1, 4]
        summaries = _compute_metrics(
            top_ks_for_accuracy=top_ks_for_accuracy,
            top_ks_for_ndcg=top_ks_for_ndcg,
            data_generator=dummy_data_generator(
                query_embeddings,
                query_paddings,
                text_positive_embeddings,
                text_positive_paddings,
                text_negative_embeddings,
                text_negative_paddings,
            ),
        )

        summaries_from_similarity_matrix = calculate_retrieval_metrics_from_similarity_matrix(
            sim=jnp.einsum(
                "i d, j d -> i j",
                jnp.reshape(query_embeddings, (-1, 2)),
                jnp.reshape(text_positive_embeddings, (-1, 2)),
            ),
            text_positive_paddings=text_positive_paddings,
            query_paddings=query_paddings,
            text_paddings=jnp.reshape(text_positive_paddings, -1),
            top_ks_for_accuracy=top_ks_for_accuracy,
            top_ks_for_ndcg=top_ks_for_ndcg,
        )
        expected_metrics = {
            "avg_rank": (2 + 5 + 1) / 3,
            "retrieval_accuracy@1": 1 / 3,
            "retrieval_accuracy@4": 2 / 3,
        }
        if top_ks_for_ndcg:
            # Average of per-query nDCG.
            expected_metrics.update(
                {
                    "ndcg@1": (0 + 0 + 1) / 3,
                    "ndcg@4": (0.650921 + 0 + 0.6131472) / 3,
                }
            )

        self.assertNestedAllClose(summaries, expected_metrics)
        self.assertNestedAllClose(summaries_from_similarity_matrix, expected_metrics)

        # Tests when there is no negative text. text_negative_embeddings and
        # text_negative_paddings have a shape of (num_examples, 0).
        text_negative_embeddings = jnp.zeros((4, 0))
        text_negative_paddings = jnp.zeros((4, 0))
        summaries = _compute_metrics(
            top_ks_for_accuracy=[1, 4],
            top_ks_for_ndcg=top_ks_for_ndcg,
            data_generator=dummy_data_generator(
                query_embeddings,
                query_paddings,
                text_positive_embeddings,
                text_positive_paddings,
                text_negative_embeddings,
                text_negative_paddings,
            ),
        )
        self.assertNestedAllClose(summaries, expected_metrics)

        # Tests when there are negative texts.
        text_negative_embeddings = jnp.asarray(
            [
                [[2, 3]],
                [[4, 2]],
                [[1, -1]],
                [[-2, 2]],
            ]
        )
        text_negative_paddings = jnp.asarray([[0], [0], [0], [1]])
        top_ks_for_accuracy = [1, 7]
        """
        similarity_matrix:
        [[ 17   7   9   5   4  19   3  10  11  18   3  -6]
         [ 23   3   1   5   6  21   7   0   9  22   7 -14]
         [ 14   9  13   5   3  18   1  15  12  16   1  -2]
         [  7  10  16   4   1  13  -2  19  11  10  -2   4]]
        """
        summaries = _compute_metrics(
            top_ks_for_accuracy=top_ks_for_accuracy,
            top_ks_for_ndcg=top_ks_for_ndcg,
            data_generator=dummy_data_generator(
                query_embeddings,
                query_paddings,
                text_positive_embeddings,
                text_positive_paddings,
                text_negative_embeddings,
                text_negative_paddings,
            ),
        )
        text_embeddings = jnp.concatenate(
            [
                jnp.reshape(text_positive_embeddings, (-1, 2)),
                jnp.reshape(text_negative_embeddings, (-1, 2)),
            ],
            axis=0,
        )
        text_paddings = jnp.concatenate(
            [jnp.reshape(text_positive_paddings, -1), jnp.reshape(text_negative_paddings, -1)],
            axis=0,
        )
        summaries_from_similarity_matrix = calculate_retrieval_metrics_from_similarity_matrix(
            sim=jnp.einsum(
                "i d, j d -> i j",
                jnp.reshape(query_embeddings, (-1, 2)),
                text_embeddings,
            ),
            text_positive_paddings=text_positive_paddings,
            query_paddings=query_paddings,
            text_paddings=text_paddings,
            top_ks_for_accuracy=top_ks_for_accuracy,
            top_ks_for_ndcg=top_ks_for_ndcg,
        )
        expected_metrics = {
            "avg_rank": (3 + 8 + 1) / 3,
            "retrieval_accuracy@1": 1 / 3,
            "retrieval_accuracy@7": 2 / 3,
        }
        if top_ks_for_ndcg:
            # Average of per-query nDCG.
            expected_metrics.update(
                {
                    "ndcg@1": (0 + 0 + 1) / 3,
                    "ndcg@4": (0.3065736 + 0 + 0.6131472) / 3,
                }
            )
        self.assertNestedAllClose(summaries, expected_metrics)
        self.assertNestedAllClose(summaries_from_similarity_matrix, expected_metrics)
