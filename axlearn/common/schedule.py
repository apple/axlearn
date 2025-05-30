# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Optimizer schedules."""
import math
from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp
from optax import constant_schedule, cosine_decay_schedule

from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.utils import Tensor

ScheduleFn = Callable[[Tensor], Tensor]
Schedule = Union[float, Tensor, ScheduleFn, InstantiableConfig]


def as_schedule_fn(s: Optional[Schedule]) -> ScheduleFn:
    if hasattr(s, "instantiate"):
        return s.instantiate()
    if s is None:
        s = 1.0
    if isinstance(s, (float, int)):
        return lambda step: float(s)
    assert callable(s), s
    return s


def polynomial(
    *,
    begin_step: int = 0,
    begin_value: float = 0,
    end_step: int = 1,
    end_value: float = 0,
    power: float = 1,
) -> ScheduleFn:
    """A polynomial (linear when power=1) schedule.

    Args:
        begin_step: The first step of polynomial schedule.
        begin_value: The begin value of polynomial schedule.
        end_step: The end step of polynomial schedule. Must be > begin_step.
        end_value: The end value of polynomial schedule.
        power: The polynomial power.

    Returns:
        A ScheduleFn according to the spec.

    Raises:
        ValueError: If begin_step >= end_step.
    """
    if begin_step >= end_step:
        raise ValueError(f"begin_step {begin_step} must be < end_step {end_step}.")

    def fn(step: Tensor) -> Tensor:
        frac = (step - begin_step) / (end_step - begin_step)
        frac = jnp.minimum(1.0, jnp.maximum(0.0, frac))
        return begin_value + (frac**power) * (end_value - begin_value)

    return fn


def exponential(
    *,
    begin_step: int = 0,
    begin_value: float = 0,
    end_step: int = 1,
    end_value: float = 0,
) -> ScheduleFn:
    """An exponential schedule.

    Args:
        begin_step: The first step of the schedule.
        begin_value: The begin value of the schedule.
        end_step: The end step of the schedule. Must be > begin_step.
        end_value: The end value of the schedule.

    Returns:
        A ScheduleFn according to the spec.

    Raises:
        ValueError: If begin_step >= end_step, or if either of begin_value or end_value are not
            positive.
    """
    if begin_step >= end_step:
        raise ValueError(f"begin_step {begin_step} must be < end_step {end_step}.")
    if begin_value <= 0 or end_value <= 0:
        raise ValueError(
            f"begin_value ({begin_value}) and end_value ({end_value}) must be both positive."
        )

    log_fn = polynomial(
        begin_step=begin_step,
        begin_value=math.log(begin_value),
        end_step=end_step,
        end_value=math.log(end_value),
    )

    def fn(step: Tensor) -> Tensor:
        return jnp.exp(log_fn(step))

    return fn


def inverse_sqrt(step: int, warmup_steps: int = 1) -> float:
    """Inverse sqrt schedule optionally with constant warmup, as seen in T5.

    Args:
        step: Current step of the schedule.
        warmup_steps: Number of warmup steps. Must be > 0.

    Returns:
        Current learning rate.

    Raises:
        ValueError: If warmup_steps is <= 0.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0.")
    return jnp.maximum(step, warmup_steps) ** -0.5


def adafactor(
    scale: float = 1.0,
    *,
    warmup_steps: int = 10_000,
    decay_power: float = -0.5,
    step_offset: int = 0,
) -> ScheduleFn:
    """Transformer learning rate scheduler.

    Note the effective peak_lr = scale * warmup_steps**decay_power.
    Attention Is All You Need. Eq.3. https://arxiv.org/pdf/1706.03762.pdf.
    Set step_offset = 1 to be equivalent to Lingvo implementation.
    https://github.com/tensorflow/lingvo/blob/253727f5bbedf4311e45ba62df46f2ee1009d12a/lingvo/core/schedule.py#L340
    """

    def fn(step):
        return scale * jnp.minimum(
            (step + step_offset) * (warmup_steps ** (decay_power - 1.0)),
            (step + step_offset) ** decay_power,
        )

    return fn


def adafactor_decay_rate(c: float = 0.8, step_offset: int = 0) -> ScheduleFn:
    """Returns the beta2 schedule described in section 7.2 of https://arxiv.org/abs/1804.04235.

    We assume that the first step is step_offset + 1.

    step = max(step - step_offset, 1)
    beta2 = 1 - step ** (-c)

    Args:
        c: The exponent.
        step_offset: The initial step number. If finetuning with some existing learner state, set
            offset to the number of steps to ensure that decay rate is reset.
            If `step` is less than or equal to `step_offset`, we clamp to `step - step_offset` to 1.

    Returns:
        The beta2 schedule.
    """

    def fn(step):
        step = jnp.maximum(step - step_offset, 1)
        return 1 - step ** (-c)

    return fn


def decay_bias_correction(decay: float) -> ScheduleFn:
    """Returns a ScheduleFn that applies bias correction to the given decay.

    Example usage:
        sch_fn = decay_bias_correction(0.9)
        ema = 0
        for step in range(1, num_steps + 1):
            # Note that steps start from 1. decay_t will always be 0 when step=1.
            decay_t = sch_fn(step)
            # ema will be a bias corrected exponential moving average of value(step), e.g.,
            # after step 1, ema = value(1);
            # after step 2, ema = (value(1) * decay + value(2)) / (decay + 1)
            ema = ema * decay_t + value(step) * (1 - decay_t)

    Reference:
    https://arxiv.org/pdf/1804.04235.pdf, section 7.1.
    https://github.com/tensorflow/lingvo/blob/3d16483b749a1181330ae9ce318688e7518d63c9/lingvo/jax/optimizers.py#L170-L191
    https://github.com/tensorflow/lingvo/blob/a527d2541abd7200cd4c2be0159e5bacc3cdeda1/lingvo/core/optimizer.py#L896
    """

    def fn(step):
        t = jnp.asarray(step, dtype=jnp.float32)
        return decay * (1.0 - jnp.power(decay, t - 1)) / (1.0 - jnp.power(decay, t))

    return fn


def stepwise(sub: list[Schedule], start_step: list[int]) -> ScheduleFn:
    """A composite schedule consisting of multiple sub-schedules.

    The first sub-schedule starts at step 0. For the rest of sub-schedules,
    sub[i] starts at start_step[i-1].

    The step passed to sub-schedule is the relative step from its start step,
    so that the values of each sub-schedule do not depend on other sub-schedules.

    Args:
        sub: a sequence of N sub-schedules.
        start_step: a sequence of N-1 integers. start_steps[i] represents the starting step
            of sub[i+1].
            0 <= start_step[i] <= start_step[i+1].

    Returns:
        A composite schedule.

    Raises:
        ValueError: If sub or start_step have invalid lengths, or if any start_step is negative.
    """
    if len(sub) != len(start_step) + 1:
        raise ValueError(f"Unexpected length: {len(sub)} != {len(start_step)} + 1")
    if not all(step >= 0 for step in start_step):
        raise ValueError(f"start_step must be >= 0: {start_step}")
    sub = [as_schedule_fn(s) for s in sub]
    all_start_steps = [0] + start_step
    all_limit_steps = start_step + [-1]

    def fn(step: Tensor) -> Tensor:
        values = [s(jnp.maximum(0, step - start)) for s, start in zip(sub, all_start_steps)]
        activations = [
            jnp.logical_and(
                jax.lax.le(start, step),
                jnp.logical_or(limit < 0, jax.lax.lt(step, limit)),
            ).astype(jnp.float32)
            for start, limit in zip(all_start_steps, all_limit_steps)
        ]
        return sum(value * activation for value, activation in zip(values, activations))

    return fn


def segment_wise(segments: list[Schedule], *, segment_steps: list[int]) -> ScheduleFn:
    """A composite schedule consisting of multiple segments, each with its own schedule.

    The step passed to sub-schedule always starts at 1, so that the values of each sub-schedule do
    not depend on other sub-schedules.

    Args:
        segments: a sequence of N sub-schedules.
        segment_steps: a sequence of N - 1 integers. segment_steps[i] represents the number of steps
            of segments[i]. Assuming that segments[N-1] = inf, segment[k] will be used for step
            [1 + sum(segment_steps[:k]), sum(segment_steps[:k + 1])].

    Returns:
        A composite schedule, s.t.
        * If step <= 0 or step > sum(segment_steps), the schedule returns 0;
        * If 1 + sum(segment_steps[:k]) <= step <= sum(segment_steps[:k + 1]), the schedule returns
          segments[k](step - sum(segment_steps[:k]));

    Raises:
        ValueError: If segments or segment_steps have incompatible lengths, or if any element of
            segment_steps is negative.
    """
    if len(segments) != len(segment_steps) + 1:
        raise ValueError(f"Unexpected length: {len(segments)=} != {len(segment_steps)=} + 1")
    if not all(num_steps >= 0 for num_steps in segment_steps):
        raise ValueError(f"segment_steps must be >= 0: {segment_steps}")
    segments = [as_schedule_fn(s) for s in segments]

    # segment[k] is used for steps in range [segment_offsets[k] + 1, segment_offsets[k + 1]].
    segment_offsets = [0]
    for num_steps in segment_steps:
        segment_offsets.append(segment_offsets[-1] + num_steps)

    def fn(step: Tensor) -> Tensor:
        values = [s(jnp.maximum(1, step - segment_offsets[k])) for k, s in enumerate(segments)]
        activations = [
            jnp.logical_and(
                jax.lax.le(segment_offsets[k] + 1, step),
                jax.lax.le(step, segment_offsets[k + 1]) if k + 1 < len(segment_offsets) else True,
            )
            for k in range(len(segments))
        ]
        return sum(value * activation for value, activation in zip(values, activations))

    return fn


def cosine_with_linear_warmup(
    peak_lr: float,
    *,
    max_step: int,
    warmup_steps: int = 500,
    begin_value: float = 0.0,
    alpha: float = 0.0,
    decay_begin_step: Optional[int] = None,
) -> ScheduleFn:
    """Cosine learning rate schedule with linear warm-up.

    Args:
        peak_lr: Peak value of the cosine learning rate.
        max_step: The total number of steps of the warm-up schedule and the cosine learning
            rate decay schedule.
        warmup_steps: The number of steps of the warm-up schedule. Skip warm-up if set to 0.
        begin_value: The begin value of the linear warm-up.
        alpha: The minimum value of the multiplier used to adjust the cosine learning rate.
        decay_begin_step: The step to begin cosine decay. The learning rate is kept constant
            in [warmup_steps, decay_begin_step). Ignored if decay_begin_step <= warmup_steps.

    Returns:
        A composite schedule.
    """
    segments, segment_steps = [], []
    if warmup_steps > 0:
        segments.append(
            config_for_function(polynomial).set(
                begin_step=0,
                begin_value=begin_value,
                end_step=warmup_steps,
                end_value=peak_lr,
            )
        )
        segment_steps.append(warmup_steps)
    if decay_begin_step is not None and decay_begin_step > warmup_steps:
        segments.append(
            config_for_function(polynomial).set(
                begin_step=0,
                begin_value=peak_lr,
                end_step=decay_begin_step - warmup_steps,
                end_value=peak_lr,
            )
        )
        segment_steps.append(decay_begin_step - warmup_steps)
    cosine_decay_steps = max_step - sum(segment_steps or [])
    segments.append(
        config_for_function(cosine_decay_schedule).set(
            init_value=peak_lr,
            decay_steps=cosine_decay_steps,
            alpha=alpha,
        )
    )
    return segment_wise(segments=segments, segment_steps=segment_steps)


def constant_with_linear_warmup(
    peak_lr: float,
    *,
    warmup_steps: int = 500,
    begin_value: float = 0.0,
) -> ScheduleFn:
    """Constant learning rate schedule with linear warm-up.

    Args:
        peak_lr: Value of the constant learning rate.
        warmup_steps: The number of steps of the warm-up schedule.
        begin_value: The begin value of the linear warm-up.

    Returns:
        A composite schedule.
    """
    return segment_wise(
        segments=[
            config_for_function(polynomial).set(
                begin_step=0,
                begin_value=begin_value,
                end_step=warmup_steps,
                end_value=peak_lr,
            ),
            config_for_function(constant_schedule).set(
                value=peak_lr,
            ),
        ],
        segment_steps=[warmup_steps],
    )


def linear_schedule_with_warmup(
    peak_lr: float,
    *,
    max_step: int,
    warmup_steps: int,
    begin_value: float = 0.0,
    end_value: float = 0.0,
) -> ScheduleFn:
    """Learning rate schedule with linear warm-up.

    Args:
        peak_lr: Peak value of the learning rate.
        max_step: The total number of steps of the warm-up schedule and the learning
            rate decay schedule.
        warmup_steps: The number of steps of the warm-up schedule.
        begin_value: The begin value of the linear warm-up.
        end_value: The end value of the linear decay.

    Returns:
        A composite schedule.
    """
    return segment_wise(
        segments=[
            config_for_function(polynomial).set(
                begin_step=0, begin_value=begin_value, end_step=warmup_steps, end_value=peak_lr
            ),
            config_for_function(polynomial).set(
                begin_step=0,
                begin_value=peak_lr,
                end_step=max_step - warmup_steps,
                end_value=end_value,
            ),
        ],
        segment_steps=[warmup_steps],
    )


def ema_schedule(decay: float = 0.9999, *, warmup_steps: int = 1) -> ScheduleFn:
    """Ema decay schedule with warm-up.

    The ema decay is 0, 1/2, 2/3, 3/4, 4/5, ... during warm-up, and then is constant at decay.

    Args:
        decay: ema decay.
        warmup_steps: The number of steps of the warm-up schedule.

    Returns:
        A ema decay schedule.

    Raises:
        ValueError: If warmup_steps <= 0.
    """
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be > 0.")

    def fn(step):
        return step / (1.0 + step) * (step < warmup_steps) + decay * (step >= warmup_steps)

    return fn
