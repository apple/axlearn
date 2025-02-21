# Copyright © 2024 Apple Inc.

"""Tests for attention_bias.py."""
from typing import Optional

import chex
import jax.numpy as jnp
import jax.util
from absl.testing import absltest, parameterized
from jax.sharding import PartitionSpec

from axlearn.common import attention_bias, test_utils
from axlearn.common.attention_bias import (
    CausalAttentionBias,
    CompositeAttentionBias,
    MaskFnAttentionBias,
    SegmentIdAttentionBias,
    TensorAttentionBias,
    sliding_window_causal_mask,
)
from axlearn.common.utils import Tensor


class MaskTest(test_utils.TestCase):
    @parameterized.parameters(
        [0, [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]],
        [2, [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1]]],
        [4, [[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]],
    )
    def test_sliding_window_mask(self, left_context, expected):
        mask_fn = sliding_window_causal_mask(sliding_window_size=left_context)
        step_len = 5
        target_positions = jnp.arange(step_len)[:, None]
        source_positions = jnp.arange(step_len)[None, :]
        bool_mask = mask_fn(target_positions, source_positions)
        out_mask = bool_mask.astype(jnp.int32)
        self.assertEqual(out_mask.tolist(), expected)


class AttentionBiasTest(test_utils.TestCase):
    @parameterized.parameters(
        [attention_bias.ZeroAttentionBias(), False],
        [attention_bias.CausalAttentionBias(shape=(5, 5)), True],
        [attention_bias.MaskFnAttentionBias(attention_bias.causal_mask, shape=(5, 5)), True],
        [attention_bias.TensorAttentionBias.from_tensor(jnp.ones((5, 5))), True],
    )
    def test_has_bias(self, bias, expected):
        self.assertEqual(bias.has_value(), expected)

    def test_causal_attention_bias(self):
        bias = attention_bias.CausalAttentionBias(shape=(5, 5))
        chex.assert_trees_all_close(bias.value(), attention_bias.make_causal_biases(5)[None, None])
        self.assertIsInstance(bias, attention_bias.CausalAttentionBias)

        bias = attention_bias.MaskFnAttentionBias(attention_bias.causal_mask, shape=(5, 5))
        self.assertNotIsInstance(bias, attention_bias.CausalAttentionBias)

    def test_zero_attention_bias(self):
        bias = attention_bias.ZeroAttentionBias()
        self.assertEqual(bias.value(), None)

        bias = attention_bias.MaskFnAttentionBias(None, shape=(5, 5))
        self.assertNotIsInstance(bias, attention_bias.ZeroAttentionBias)

        self.assertNotIsInstance(
            attention_bias.CausalAttentionBias(shape=(5, 5)), attention_bias.ZeroAttentionBias
        )

    def test_base_attention_bias_value(self):
        """Tests `BaseAttentionBias.value()`."""
        # pylint: disable=function-redefined

        class TestAttentionBias(attention_bias.BaseAttentionBias):
            def _value(self) -> Tensor:
                return jnp.ones((5, 7))

        self.assertEqual(TestAttentionBias().value().shape, (1, 1, 5, 7))

        class TestAttentionBias(attention_bias.BaseAttentionBias):
            def _value(self) -> Tensor:
                return jnp.ones((3, 5, 7))

        self.assertEqual(TestAttentionBias().value().shape, (3, 1, 5, 7))

        class TestAttentionBias(attention_bias.BaseAttentionBias):
            def _value(self) -> Tensor:
                return jnp.ones((2, 3, 5, 7))

        self.assertEqual(TestAttentionBias().value().shape, (2, 3, 5, 7))

    def test_base_attention_bias_and_residual(self):
        """Tests `BaseAttentionBias.bias_and_residual()`."""
        bias = attention_bias.ZeroAttentionBias()
        self.assertEqual(
            bias.bias_and_residual(attention_bias.ZeroAttentionBias),
            attention_bias.BiasAndResidual(bias=bias, residual=CompositeAttentionBias([])),
        )
        self.assertEqual(
            bias.bias_and_residual(attention_bias.BaseAttentionBias),
            attention_bias.BiasAndResidual(bias=bias, residual=CompositeAttentionBias([])),
        )
        self.assertEqual(
            bias.bias_and_residual(int), attention_bias.BiasAndResidual(bias=None, residual=bias)
        )

    @parameterized.parameters(
        [
            attention_bias.CompositeAttentionBias(
                [attention_bias.ZeroAttentionBias(), attention_bias.ZeroAttentionBias()]
            ),
            False,
        ],
        [
            attention_bias.CompositeAttentionBias(
                [
                    attention_bias.CausalAttentionBias(shape=(5, 5)),
                    attention_bias.CausalAttentionBias(shape=(5, 5)),
                ]
            ),
            True,
        ],
        [
            attention_bias.CompositeAttentionBias(
                [
                    attention_bias.CausalAttentionBias(shape=(5, 5)),
                    attention_bias.ZeroAttentionBias(),
                ]
            ),
            True,
        ],
        [
            attention_bias.CompositeAttentionBias(
                [
                    attention_bias.ZeroAttentionBias(),
                    attention_bias.CausalAttentionBias(shape=(5, 5)),
                ]
            ),
            True,
        ],
    )
    def test_composite_attention_has_bias(self, bias, expected):
        self.assertEqual(bias.has_value(), expected)

    def test_bias_and_residual_has_bias(self):
        bias = attention_bias.CompositeAttentionBias(
            [
                attention_bias.CausalAttentionBias(shape=(5, 5)),
                attention_bias.MaskFnAttentionBias(attention_bias.causal_mask, shape=(5, 5)),
            ]
        )
        bias_and_residual = bias.bias_and_residual(attention_bias.CausalAttentionBias)
        self.assertTrue(bias_and_residual.has_value())
        bias_and_residual = bias.bias_and_residual(attention_bias.MaskFnAttentionBias)
        self.assertTrue(bias_and_residual.has_value())

    def test_composite_attention_bias_zero(self):
        # Test handling of zero biases.
        bias = attention_bias.CompositeAttentionBias(
            [attention_bias.ZeroAttentionBias(), attention_bias.ZeroAttentionBias()]
        )
        self.assertEqual(bias.value(), None)
        self.assertEqual(bias._nonzero(), [])  # pylint: disable=protected-access
        # The partition spec needs to have the same structure as the biases list.
        self.assertEqual(bias.partition_spec({}).biases, [PartitionSpec(), PartitionSpec()])

    def test_composite_attention_bias(self):
        # Test value().
        b1 = attention_bias.CausalAttentionBias(shape=(5, 5))
        # Opposite of causal mask.
        b2 = attention_bias.MaskFnAttentionBias(shape=(5, 5), mask=lambda q, k: q < k)
        expected = attention_bias.MaskFnAttentionBias(
            shape=(5, 5),
            mask=lambda q, k: jnp.zeros(jnp.broadcast_shapes(q.shape, k.shape), dtype=bool),
        )
        bias = attention_bias.CompositeAttentionBias([b1, b2])
        self.assertNestedEqual(bias.value(), expected.value())

        # Test adding biases.
        bias = b1 + b2
        self.assertNestedEqual(bias.value(), expected.value())

        # Test bias_and_residual().
        bias = attention_bias.CompositeAttentionBias([b2, b1])
        bias_and_residual = bias.bias_and_residual(attention_bias.CausalAttentionBias)
        self.assertNestedEqual(bias_and_residual.bias.value(), b1.value())
        self.assertNestedEqual(bias_and_residual.residual.value(), b2.value())

        bias_and_residual = bias.bias_and_residual(attention_bias.MaskFnAttentionBias)
        self.assertNestedEqual(bias_and_residual.bias.value(), bias.value())
        self.assertIs(bias_and_residual.residual.value(), None)

        bias_and_residual = bias.bias_and_residual(attention_bias.ZeroAttentionBias)
        self.assertNestedEqual(bias_and_residual.bias, None)
        self.assertNestedEqual(bias_and_residual.residual.value(), bias.value())

        bias_and_residual = bias.bias_and_residual(attention_bias.CompositeAttentionBias)
        self.assertNestedEqual(bias_and_residual.bias.value(), bias.value())
        self.assertNestedEqual(bias_and_residual.residual.value(), None)

        bias_and_residual = (b1 + b1).bias_and_residual(attention_bias.CausalAttentionBias)
        self.assertNestedEqual(bias_and_residual.bias.value(), b1.value())
        self.assertNestedEqual(bias_and_residual.residual.value(), None)

    def test_bias_and_residual_repeated_call(self):
        """Test repeated calls to `bias_and_residual()`."""
        b1 = attention_bias.CausalAttentionBias(shape=(5, 5))
        # Opposite of causal mask.
        b2 = attention_bias.MaskFnAttentionBias(shape=(5, 5), mask=lambda q, k: q < k)
        bias = attention_bias.CompositeAttentionBias([b2, b1])
        causal_bias, residual = bias.bias_and_residual(attention_bias.CausalAttentionBias)
        mask_fn_bias, residual = residual.bias_and_residual(attention_bias.MaskFnAttentionBias)
        self.assertIs(causal_bias, b1)
        self.assertIs(mask_fn_bias, b2)
        self.assertIs(residual.value(), None)

        # Test nested CompositeAttentionBias.
        bias = CompositeAttentionBias([CompositeAttentionBias([b1]), b2])
        bias_and_residual = bias.bias_and_residual(attention_bias.MaskFnAttentionBias)
        self.assertNestedEqual(bias_and_residual.bias.value(), bias.value())
        self.assertIsInstance(bias_and_residual.bias, attention_bias.MaskFnAttentionBias)
        self.assertIs(bias_and_residual.residual.value(), None)

    def test_split(self):
        b1 = attention_bias.CausalAttentionBias(shape=(5, 5))
        # Opposite of causal mask.
        b2 = attention_bias.MaskFnAttentionBias(shape=(5, 5), mask=lambda q, k: q < k)
        bias = attention_bias.CompositeAttentionBias([b2, b1])
        causal_bias, mask_fn_bias, residual = attention_bias.split(
            bias, attention_bias.CausalAttentionBias, attention_bias.MaskFnAttentionBias
        )
        self.assertIs(causal_bias, b1)
        self.assertIs(mask_fn_bias, b2)
        self.assertIs(residual.value(), None)

        zero_bias, residual = attention_bias.split(bias, attention_bias.TensorAttentionBias)
        self.assertIs(zero_bias.value(), None)
        self.assertNestedEqual(residual.value(), bias.value())

        b3 = attention_bias.SegmentIdAttentionBias(jnp.asarray([1, 1, 2, 2, 2]))
        segment, mask, residual = attention_bias.split(
            b1 + b3, attention_bias.SegmentIdAttentionBias, attention_bias.MaskFnAttentionBias
        )
        self.assertIs(segment, b3)
        self.assertIs(mask, b1)
        self.assertIs(residual.value(), None)

    @parameterized.product(
        causal=[None, attention_bias.CausalAttentionBias(shape=(3, 3))],
        segment_ids=[None, attention_bias.SegmentIdAttentionBias(jnp.asarray([1, 2, 3]))],
        mask=[None, attention_bias.MaskFnAttentionBias(mask=lambda q, k: q < k, shape=(3, 3))],
    )
    def test_split_subsets(
        self,
        causal: Optional[CausalAttentionBias],
        segment_ids: Optional[SegmentIdAttentionBias],
        mask: Optional[MaskFnAttentionBias],
    ):
        """Tests split() where the input CompositeBias contains any possible subsets of a
        causal, segment id, and mask fn attention bias.
        """
        bias_list = [mask, causal, segment_ids]
        bias_list = [b for b in bias_list if b is not None]
        bias = attention_bias.CompositeAttentionBias(bias_list)
        new_bias_list = attention_bias.split(
            bias,
            attention_bias.CausalAttentionBias,
            attention_bias.SegmentIdAttentionBias,
            attention_bias.MaskFnAttentionBias,
        )
        new_bias_list = [b if b.has_value() else None for b in new_bias_list]
        expected = [causal, segment_ids, mask, None]
        for b1, b2 in jax.util.safe_zip(new_bias_list, expected):
            self.assertIs(b1, b2)

    def test_tensor_attention_bias(self):
        bias = attention_bias.TensorAttentionBias.from_tensor(jnp.ones((5, 7)))
        self.assertNestedEqual(bias.value(), jnp.ones((1, 1, 5, 7)))

    def test_segment_id_attention_bias(self):
        bias = attention_bias.SegmentIdAttentionBias(
            jnp.asarray([[1, 1, 2, 2, 2, 0], [1, 2, 3, 4, 5, 6]])
        )
        expected = attention_bias.bool_to_bias(
            jnp.asarray(
                [
                    [
                        [True, True, False, False, False, False],
                        [True, True, False, False, False, False],
                        [False, False, True, True, True, False],
                        [False, False, True, True, True, False],
                        [False, False, True, True, True, False],
                        [False, False, False, False, False, True],
                    ],
                    jnp.eye(6, 6, dtype=bool),
                ],
                dtype=bool,
            )
        )
        expected = expected[:, None, :, :]
        self.assertNestedEqual(bias.value(), expected)

    def test_mask_fn_attention_bias_from_sequence(self):
        """Tests `MaskFnAttentionBias.from_sequence()`."""
        b1 = attention_bias.CausalAttentionBias(shape=(5, 5))
        # Opposite of causal mask.
        b2 = attention_bias.MaskFnAttentionBias(shape=(5, 5), mask=lambda q, k: q < k)

        self.assertNestedEqual(
            attention_bias.MaskFnAttentionBias.from_sequence([b1, b2]).value(), (b1 + b2).value()
        )
        self.assertIsInstance(
            attention_bias.MaskFnAttentionBias.from_sequence([b1]),
            attention_bias.CausalAttentionBias,
        )
        self.assertIs(attention_bias.MaskFnAttentionBias.from_sequence([]), None)

    def test_mask_fn_attention_bias(self):
        bias = attention_bias.MaskFnAttentionBias(mask=lambda q, k: q >= k, shape=(5, 5))
        self.assertNestedEqual(
            bias.value(), jnp.asarray(attention_bias.make_causal_biases(5))[None, None]
        )

        bias = attention_bias.MaskFnAttentionBias(
            mask=lambda q, k: q >= k, shape=(4, 7), target_positions=jnp.asarray([3, 1])
        )
        expected = jnp.asarray(
            [
                [
                    [True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False],
                    [True, True, True, True, True, True, False],
                    [True, True, True, True, True, True, True],
                ],
                [
                    [True, True, False, False, False, False, False],
                    [True, True, True, False, False, False, False],
                    [True, True, True, True, False, False, False],
                    [True, True, True, True, True, False, False],
                ],
            ],
            dtype=bool,
        )
        expected = attention_bias.bool_to_bias(expected)[:, None, :]
        self.assertNestedEqual(bias.value(), expected)

    def test_mask_fn_attention_bias_target_positions_ndim(self):
        """Tests mask_fn_attention_bias` when `target_positions.ndim == 2."""
        bias = attention_bias.MaskFnAttentionBias(
            mask=attention_bias.causal_mask,
            shape=(5, 5),
            target_positions=jnp.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]),
        )
        expected = jnp.asarray(
            [
                [
                    attention_bias.causal_mask(*jnp.indices([5, 5])),
                ],
                [
                    attention_bias.causal_mask(*jnp.indices([5, 5]))[::-1, :],
                ],
            ],
            dtype=bool,
        )
        self.assertNestedEqual(bias.bool_value(), expected)

    def test_mask_fn_attention_bias_with_target_positions(self):
        # Ensure that MaskFnAttentionBias provides the mask_fn callback with target_positions and
        # source_positions tensors of the same rank.
        batch, target_len, source_len = 2, 5, 4
        time_step = jnp.arange(batch)

        def mask_fn(target_positions, source_positions):
            self.assertEqual(target_positions.shape, (batch, target_len, 1))
            self.assertEqual(source_positions.shape, (1, 1, source_len))
            return attention_bias.causal_mask(target_positions, source_positions)

        bias = attention_bias.MaskFnAttentionBias(
            mask=mask_fn, shape=(target_len, source_len), target_positions=time_step
        )
        ref_bias = attention_bias.MaskFnAttentionBias(
            attention_bias.causal_mask, shape=(target_len, source_len), target_positions=time_step
        )
        chex.assert_trees_all_close(bias.value(), ref_bias.value())

    def test_bool_tensor_attention_bias(self):
        bias = attention_bias.BoolTensorAttentionBias.from_tensor(jnp.ones((5, 7), dtype=bool))
        self.assertNestedEqual(
            bias.value(), attention_bias.bool_to_bias(jnp.ones((1, 1, 5, 7), dtype=bool))
        )

    def test_astype(self):
        bias = TensorAttentionBias.from_tensor(jnp.ones((5, 7), dtype=jnp.float32))
        self.assertEqual(bias.value().dtype, jnp.float32)
        bias = bias.astype(jnp.bfloat16)
        self.assertEqual(bias.value().dtype, jnp.bfloat16)


if __name__ == "__main__":
    absltest.main()
