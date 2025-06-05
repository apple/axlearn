# Copyright © 2023 Apple Inc.

"""Input processing utilities on tf.data for ranking-related tasks."""

import seqio
import tensorflow as tf

from axlearn.common import input_tf_data


def rank_by_value(
    *, input_key: str, output_key: str, ascending: bool, allow_ties: bool
) -> input_tf_data.DatasetToDatasetFn:
    """Returns a DatasetToDatasetFn that stores the ranks of input_field in output_field.

    Note the rank starts at 1.

    Args:
        input_key: The field whose value will be ranked.
        output_key: The field to store the ranks into.
        ascending: True to rank in ascending order or false to rank in descending order.
        allow_ties: If true, multiple elements could have the same rank. Ranks could have gaps
            in between to account for duplicate ranks. If false, ranks will never have ties,
            even when values are equivalent - stable sorting by values is used.
            For example, the ranks of [2, 2, 4] would be [1, 1, 3] when allow_ties is true,
            and [1, 2, 3] otherwise.

    Returns:
        A DatasetToDatasetFn where each input example is ranked according to the value
        of input_field, and the ranks are stored in output_field.
    """

    def example_fn(example: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        if len(example[input_key].shape) != 1:
            raise NotImplementedError(
                f"Only implemented for rank-1 tensors. Got rank-{len(example[input_key].shape)}."
            )

        inputs = example[input_key]
        if not ascending:
            inputs *= -1
        idx = tf.argsort(inputs, stable=True)
        ranks = tf.math.invert_permutation(idx)
        if allow_ties:
            # Construct an arange where index repeats for repeated inputs.
            # The index is for sorted inputs.
            _, _, counts = tf.unique_with_counts(tf.gather(inputs, idx))
            # pylint: disable-next=no-value-for-parameter
            idx = tf.repeat(tf.cumsum(counts, exclusive=True), counts)
            # Permute the arange to the rank order, to fetch the sorted, repeated index.
            ranks = tf.gather(idx, ranks)
        example[output_key] = ranks + 1
        return example

    return seqio.map_over_dataset(example_fn)
