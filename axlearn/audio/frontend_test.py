# Copyright © 2023 Apple Inc.

"""Tests audio frontends."""

import functools
import math

import jax
import jax.numpy as jnp
import pytest
import tensorflow as tf
from absl.testing import parameterized

from axlearn.audio.frontend import LogMelFrontend, _ms_to_samples, scale_by_mean_std
from axlearn.audio.frontend_utils_test import (
    _fake_audio,
    _ref_framer,
    _ref_log_mel_spectrogram,
    _ref_pre_emphasis,
)
from axlearn.common.config import config_for_function
from axlearn.common.module import functional as F


class LogMelFrontendTest(parameterized.TestCase, tf.test.TestCase):
    """Tests LogMelFrontend."""

    @parameterized.parameters(
        dict(frame_size_ms=3.90625, hop_size_ms=1),
        dict(frame_size_ms=1, hop_size_ms=0.15625),
    )
    def test_instantiate(self, frame_size_ms, hop_size_ms):
        num_filters, sample_rate = 80, 16_000
        cfg = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
        )
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            cfg.set(name="test").instantiate(parent=None)

    def _jit_forward(self, layer, inputs, paddings):
        @jax.jit
        def jit_forward(inputs, paddings):
            return F(
                layer,
                inputs=dict(inputs=inputs, paddings=paddings),
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state={},
            )

        test_outputs, _ = jit_forward(inputs, paddings)
        return test_outputs

    @parameterized.product(
        [
            dict(frame_size_ms=25, hop_size_ms=10),
            dict(frame_size_ms=25, hop_size_ms=25),
        ],
    )
    @pytest.mark.fp64
    def test_against_ref(self, frame_size_ms, hop_size_ms):
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters = 80

        # Construct fake inputs.
        inputs, paddings = _fake_audio(
            batch_size=batch_size, seq_len=max_seconds * sample_rate, dtype=jnp.float64
        )

        # Compute ref outputs.
        ref_outputs, ref_paddings = _ref_frontend(
            inputs=inputs,
            paddings=paddings,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
            coeff=0.97,
            num_filters=num_filters,
            lower_edge_hertz=125.0,
            upper_edge_hertz=7600.0,
            mel_floor=1.0,
        )

        # Compute test outputs.
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
        )
        layer = cfg.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)

        # Only compare the non-padding outputs.
        ref_outputs = ref_outputs * (1 - tf.cast(ref_paddings, ref_outputs.dtype))[..., None]
        self.assertAllClose(ref_outputs[..., None], test_outputs["outputs"])
        self.assertAllClose(ref_paddings, test_outputs["paddings"])

    @pytest.mark.fp64
    def test_normalization(self):
        sample_rate, batch_size, max_seconds = 16_000, 4, 13
        num_filters, frame_size_ms, hop_size_ms = 80, 25, 10
        cfg: LogMelFrontend.Config = LogMelFrontend.default_config().set(
            num_filters=num_filters,
            sample_rate=sample_rate,
            frame_size_ms=frame_size_ms,
            hop_size_ms=hop_size_ms,
        )
        layer = cfg.set(name="test").instantiate(parent=None)

        def compute_mean_std(outputs):
            # Compute mean/std of features. [batch, num_frames, num_filters, 1] -> [num_filters].
            non_padding = jnp.sum(1 - outputs["paddings"])
            mean = jnp.sum(outputs["outputs"], axis=(0, 1, -1)) / non_padding
            mean_square = jnp.sum(outputs["outputs"] ** 2, axis=(0, 1, -1)) / non_padding
            return mean, jnp.sqrt(mean_square - mean**2)

        inputs, paddings = _fake_audio(
            batch_size=batch_size, seq_len=max_seconds * sample_rate, dtype=jnp.float64
        )
        test_outputs = self._jit_forward(layer, inputs, paddings)

        # Compute again with scaling by pre-computed mean and std.
        mean, std = compute_mean_std(test_outputs)
        cfg.scaling = config_for_function(
            lambda: functools.partial(scale_by_mean_std, mean=mean.tolist(), std=std.tolist())
        )
        layer = cfg.set(name="test").instantiate(parent=None)
        test_outputs = self._jit_forward(layer, inputs, paddings)

        # Check outputs.
        mean, std = compute_mean_std(test_outputs)
        self.assertTrue(jnp.allclose(mean, 0, atol=1e-5))
        self.assertTrue(jnp.allclose(std, 1.0, atol=1e-5))


def _ref_frontend(
    *,
    inputs: tf.Tensor,
    paddings: tf.Tensor,
    sample_rate: int,
    frame_size_ms: int,
    hop_size_ms: int,
    coeff: float,
    num_filters: int,
    lower_edge_hertz: float,
    upper_edge_hertz: float,
    mel_floor: float,
):
    """Lingvo ASR frontend.

    Reference:
    https://github.com/tensorflow/lingvo/blob/4a9097a212622d99d7f8e2379804dbffdc44a97f/lingvo/tasks/asr/frontend.py#L330
    """
    frame_size = int(round(_ms_to_samples(frame_size_ms, sample_rate=sample_rate)))
    frame_step = int(round(_ms_to_samples(hop_size_ms, sample_rate=sample_rate)))

    inputs, output_paddings = _ref_framer(
        inputs=inputs, paddings=paddings, frame_size=frame_size + 1, frame_step=frame_step
    )
    inputs = _ref_pre_emphasis(inputs=inputs, coeff=coeff)
    inputs = tf.signal.hann_window(frame_size, dtype=inputs.dtype) * inputs
    fft_size = int(max(512.0, math.pow(2, math.ceil(math.log(frame_size + 1, 2)))))
    outputs = _ref_log_mel_spectrogram(
        inputs=inputs,
        fft_size=fft_size,
        num_filters=num_filters,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
        compute_energy=False,
        mel_floor=mel_floor,
    )
    return outputs, output_paddings
