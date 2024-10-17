# Copyright © 2023 Apple Inc.

"""Tests optimizer schedules."""
import math

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging
from absl.testing import absltest, parameterized

from axlearn.common import schedule
from axlearn.common.schedule import adafactor_decay_rate, as_schedule_fn


# pylint: disable=no-self-use
class ScheduleTest(parameterized.TestCase):
    """Tests schedules."""

    def test_constant(self):
        value = 3.14
        s = as_schedule_fn(value)
        for step in range(10):
            self.assertEqual(value, s(step))

    def test_linear(self):
        s = schedule.polynomial(begin_step=10, begin_value=1, end_step=20, end_value=2)
        for step in range(30):
            value = s(step)
            if step < 10:
                self.assertEqual(1, value)
            elif step > 20:
                self.assertEqual(2, value)
            else:
                self.assertEqual(step / 10, value)

    def test_sqrt(self):
        s = schedule.polynomial(power=0.5, begin_step=0, begin_value=0, end_step=100, end_value=10)
        for step in range(10):
            value = s(step)
            np.testing.assert_allclose(math.sqrt(step / 100) * 10, value, atol=1e-6)

    def test_exponential(self):
        s = schedule.exponential(begin_step=0, begin_value=1, end_step=100, end_value=0.01)
        self.assertAlmostEqual(1, s(0))
        self.assertAlmostEqual(0.1, s(50))
        self.assertAlmostEqual(0.01, s(100))
        self.assertAlmostEqual(0.01, s(101))

    def test_inverse_sqrt(self):
        s = schedule.inverse_sqrt
        for step in range(1, 11):
            value = s(step)
            self.assertAlmostEqual(1 / math.sqrt(step), value)

        # Test warmup.
        warmup_steps = 10
        for step in range(1, warmup_steps + 5):
            value = s(step, warmup_steps=warmup_steps)
            # Constant lr during warmup.
            if step <= warmup_steps:
                self.assertAlmostEqual(1 / math.sqrt(warmup_steps), value)
            else:
                self.assertAlmostEqual(1 / math.sqrt(step), value)

        # Test warmup_steps > 0.
        with self.assertRaisesRegex(ValueError, "> 0"):
            s(1, warmup_steps=0)

    def test_stepwise(self):
        s = jax.jit(schedule.stepwise(start_step=[100, 200], sub=[0.1, 0.01, 0.001]))
        for step in range(0, 300, 50):
            value = s(step)
            if step < 100:
                self.assertEqual(0.1, value)
            elif step < 200:
                self.assertEqual(0.01, value)
            else:
                self.assertEqual(0.001, value)

    def test_decay_bias_correction(self):
        decay = 0.999
        s = schedule.decay_bias_correction(decay)
        samples = [3, 4, 1, 2, 4, 6, 8, 9]
        moment = 0
        for i, sample in enumerate(samples):
            decay_i = s(i)
            moment = (1 - decay_i) * sample + decay_i * moment
            # Initially moment is an approximation of the mean value.
            self.assertAlmostEqual(moment, sum(samples[: i + 1]) / (i + 1), places=2)
        # As step --> inf, s(step) --> decay.
        self.assertAlmostEqual(decay, s(10**6))

    def test_t5(self):
        s = jax.jit(
            schedule.stepwise(
                start_step=[100],
                sub=[
                    schedule.polynomial(end_step=100, end_value=0.1),
                    lambda step: schedule.inverse_sqrt(step + 100),
                ],
            )
        )
        for step in range(200):
            value = s(step)
            logging.info("step=%s value=%s", step, value)
            if step <= 100:
                self.assertAlmostEqual((step / 100) * 0.1, value)
            else:
                self.assertAlmostEqual(1 / math.sqrt(step), value)

    @parameterized.product(
        warmup_steps=(0, 100),
        decay_begin_step=(None, 50, 200),
    )
    def test_cosine_with_linear_warmup(self, warmup_steps, decay_begin_step):
        peak_lr = 0.1
        max_step = 300
        s = jax.jit(
            schedule.cosine_with_linear_warmup(
                peak_lr=peak_lr,
                max_step=max_step,
                warmup_steps=warmup_steps,
                decay_begin_step=decay_begin_step,
            )
        )
        decay_begin_step = max(warmup_steps, decay_begin_step or 0)
        for step in range(0, 301, 50):
            value = s(step)
            if step < warmup_steps:
                # Test linear warmup.
                self.assertEqual(peak_lr * step / warmup_steps, value)
            elif warmup_steps <= step <= decay_begin_step:
                # Test constant
                self.assertEqual(peak_lr, value)
            else:
                # Test cosine decay schedule.
                cosine_rate = (
                    peak_lr
                    * 0.5
                    * (
                        1
                        + jnp.cos(
                            jnp.pi * (step - decay_begin_step) / (max_step - decay_begin_step)
                        )
                    )
                )
                self.assertAlmostEqual(cosine_rate, value)

    def test_constant_with_linear_warmup(self):
        peak_lr = 0.1
        warmup_steps = 100
        s = jax.jit(
            schedule.constant_with_linear_warmup(
                peak_lr=peak_lr,
                warmup_steps=warmup_steps,
            )
        )
        for step in range(0, 301, 50):
            value = s(step)
            if step < 100:
                # Test linear warmup.
                self.assertEqual(peak_lr * step / warmup_steps, value)
            else:
                # Test constant schedule.
                self.assertEqual(peak_lr, value)

    def test_linear_schedule_with_warmup(self):
        peak_lr = 0.1
        end_value = 0.1 * peak_lr
        max_step = 300
        warmup_steps = 100
        s = jax.jit(
            schedule.linear_schedule_with_warmup(
                peak_lr=peak_lr,
                max_step=max_step,
                warmup_steps=warmup_steps,
                end_value=end_value,
            )
        )
        for step in range(0, 301, 50):
            value = s(step)
            if step < 100:
                # Test linear warmup.
                self.assertEqual(peak_lr * step / warmup_steps, value)
            else:
                # Test linear decay schedule.
                frac = (step - warmup_steps) / (max_step - warmup_steps)
                frac = jnp.minimum(1.0, jnp.maximum(0.0, frac))
                linear_rate = peak_lr + frac**1 * (end_value - peak_lr)
                self.assertEqual(linear_rate, value)

    def test_adafactor_schedule(self):
        scale = 0.1
        warmup_steps, step_offset, max_step = 8, 1, 20
        effective_peak_lr = scale / jnp.sqrt(warmup_steps)
        s = jax.jit(
            schedule.adafactor(
                scale=scale,
                warmup_steps=warmup_steps,
                step_offset=step_offset,
            )
        )
        actual_peak_lr = -np.inf
        for step in range(max_step):
            value = s(step)
            if step < warmup_steps:
                # Test linear warmup.
                self.assertAlmostEqual(
                    effective_peak_lr * (step + step_offset) / warmup_steps, value
                )
            else:
                # Test inverse sqrt schedule.
                decay_lr = scale / jnp.sqrt(step + step_offset)
                self.assertAlmostEqual(decay_lr, value)
            actual_peak_lr = max(actual_peak_lr, value)
        self.assertAlmostEqual(actual_peak_lr, effective_peak_lr)

    def test_adafactor_decay_rate(self):
        fn = adafactor_decay_rate(step_offset=100)
        self.assertAlmostEqual(fn(1), 0)
        self.assertAlmostEqual(fn(100), 0)
        self.assertAlmostEqual(fn(200), 1 - (101) ** (-0.8))

    def test_ema_schedule(self):
        warmup_steps = 5
        s = jax.jit(
            schedule.ema_schedule(
                warmup_steps=warmup_steps,
            )
        )
        expected_warmup = [0.0, 1.0 / 2, 2.0 / 3, 3.0 / 4, 4.0 / 5]
        expected_decay = 0.9999
        for step in range(10):
            value = s(step)
            if step < warmup_steps:
                # Test warmup.
                self.assertAlmostEqual(expected_warmup[step], value)
            else:
                # Test inverse sqrt schedule.
                self.assertAlmostEqual(expected_decay, value)


if __name__ == "__main__":
    absltest.main()
