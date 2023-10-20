# Copyright © 2023 Apple Inc.

"""Input processing utilities on tf.data for ranking-related tasks."""
from typing import Dict

import seqio
import tensorflow as tf

from axlearn.common import input_tf_data


def rank_by_value(
    *, input_field: str, output_field: str, ascending: bool = True, allow_ties: bool = False
) -> input_tf_data.DatasetToDatasetFn:
    """Returns a DatasetToDatasetFn that stores the ranks of input_field in output_field.

    Note the rank starts at 1.

    Args:
        input_field: The field whose value will be ranked.
        output_field: The field to store the ranks into.
        ascending: True to rank in ascending order or False to rank in descending order.
        allow_ties: If true, multiple elements could have the same rank. Ranks could have gaps
            in between to account for duplicate ranks. For example, the ranks of [2, 2, 4]
            would be [1, 1, 3] when allow_ties is True, and [1, 2, 3] otherwise.

    Returns:
        A DatasetToDatasetFn where each input example is ranked according to the value
        of input_field, and the ranks are stored in output_field.
    """

    def example_fn(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        if len(example[input_field].shape) != 1:
            raise NotImplementedError(
                f"Only implemented for rank-1 tensors. Got rank-{len(example[input_field].shape)}."
            )

        if allow_ties:
            values = example[input_field] * -1 if not ascending else example[input_field]
            sorted_idx = tf.argsort(values, stable=True)
            sorted_values = tf.gather(params=values, indices=sorted_idx)

            _, idx, counts = tf.unique_with_counts(sorted_values)
            cumsum = tf.zeros_like(counts)
            # Index i stores total number of values that come before position i, exclusive.
            # pylint: disable-next=unexpected-keyword-arg,no-value-for-parameter
            cumsum = tf.concat([cumsum[:1], tf.cumsum(counts[:-1])], axis=0)
            ranks = tf.gather(params=cumsum, indices=idx) + 1
            ranks = tf.scatter_nd(
                indices=tf.expand_dims(sorted_idx, 1),
                updates=ranks,
                shape=tf.convert_to_tensor([len(values)]),
            )
        else:
            ranks = (
                tf.argsort(
                    tf.argsort(
                        example[input_field],
                        direction="ASCENDING" if ascending else "DESCENDING",
                        stable=True,
                    )
                )
                + 1
            )
        example[output_field] = ranks
        return example

    return seqio.map_over_dataset(example_fn)
