# Copyright © 2026 Apple Inc.

"""Tests for streaming_base helper functions."""

import jax.numpy as jnp
from absl.testing import absltest, parameterized

from axlearn.audio.streaming.streaming_base import (
    StreamingBase,
    compute_decoder_segment_pad,
    compute_encoder_segment_pad,
    next_segment_pos,
)
from axlearn.common.config import config_class
from axlearn.common.module import nowrap
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import Nested, Tensor


class MockLayer(StreamingBase):
    """Mock layer for testing segment_pad helpers."""

    @config_class
    class Config(StreamingBase.Config):
        in_stride: int = 1
        out_stride: int = 1
        segment_pad: int = 0

    @classmethod
    def in_stride(cls, cfg: Config) -> int:
        return cfg.in_stride

    @classmethod
    def out_stride(cls, cfg: Config) -> int:
        return cfg.out_stride

    @classmethod
    def segment_pad(cls, cfg: Config) -> int:
        return cfg.segment_pad

    @nowrap
    def init_states(self, *, batch_size: int, dtype: jnp.dtype) -> Nested[Tensor]:
        return dict()

    def extend_step(
        self, *, cached_states: Nested[Tensor], input_data: Nested[Tensor], is_prefill: bool = False
    ) -> tuple[Nested[Tensor], Nested[Tensor]]:
        del is_prefill
        return dict(), dict()


def _mock_cfg(in_stride=1, out_stride=1, segment_pad=0):
    return MockLayer.default_config().set(
        in_stride=in_stride, out_stride=out_stride, segment_pad=segment_pad
    )


class ComputeSegmentPadTest(TestCase):
    @parameterized.parameters(
        dict(layers=[], expected=0),
        dict(layers=[_mock_cfg(segment_pad=5, in_stride=2)], expected=5),
        dict(
            layers=[_mock_cfg(segment_pad=2, in_stride=4), _mock_cfg(segment_pad=3, in_stride=1)],
            expected=12,
        ),
        # Similar to STFT + subsampler.
        dict(
            layers=[
                _mock_cfg(segment_pad=1, in_stride=2),
                _mock_cfg(segment_pad=1, in_stride=3),
                _mock_cfg(segment_pad=160, in_stride=160),
            ],
            expected=960,
        ),
    )
    def test_compute_encoder_segment_pad(self, layers, expected):
        self.assertEqual(compute_encoder_segment_pad(layers), expected)

    @parameterized.parameters(
        dict(layers=[], expected=0),
        dict(layers=[_mock_cfg(segment_pad=5, out_stride=2)], expected=5),
        dict(
            layers=[
                _mock_cfg(segment_pad=2, out_stride=4),
                _mock_cfg(segment_pad=10, out_stride=1),
            ],
            expected=3,
        ),
        # Similar to upsampler + iSTFT.
        dict(
            layers=[
                _mock_cfg(segment_pad=2, out_stride=160),
                _mock_cfg(segment_pad=1, out_stride=3),
                _mock_cfg(segment_pad=2, out_stride=2),
            ],
            expected=2,
        ),
    )
    def test_compute_decoder_segment_pad(self, layers, expected):
        self.assertEqual(compute_decoder_segment_pad(layers), expected)


class NextSegmentPosTest(TestCase):
    @parameterized.parameters(
        dict(current_len=0, segment_pad=0, stride=1, expected=(0, 0)),
        dict(current_len=5, segment_pad=3, stride=1, expected=(8, 3)),
        dict(current_len=5, segment_pad=0, stride=4, expected=(8, 3)),
        dict(current_len=5, segment_pad=3, stride=4, expected=(8, 3)),
        dict(current_len=10, segment_pad=5, stride=8, expected=(16, 6)),
        dict(current_len=960, segment_pad=960, stride=960, expected=(1920, 960)),
    )
    def test_next_segment_pos(self, current_len, segment_pad, stride, expected):
        self.assertEqual(
            next_segment_pos(current_len, segment_pad=segment_pad, stride=stride), expected
        )


if __name__ == "__main__":
    absltest.main()
