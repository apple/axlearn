# Copyright © 2023 Apple Inc.

"""Tests tranducer layers."""
# pylint: disable=duplicate-code,invalid-name

import jax
import jaxlib
import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import absltest, parameterized
from jax import numpy as jnp
from jax.experimental import checkify

from axlearn.common.config import config_for_function
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestWithTemporaryCWD
from axlearn.common.transducer import (
    _NEG_INF,
    Transducer,
    _tilt,
    _untilt,
    apply_paddings,
    classic_logits_to_log_probs,
    hat_logits_to_log_probs,
    log_prob_alignments,
    log_prob_gradients,
    log_prob_prefix_alignments,
    log_prob_suffix_alignments,
    log_probs_from_blank_and_tokens,
)
from axlearn.common.utils import NestedTensor, Tensor, runtime_checks


def numpy_log_prob_prefix_alignments(
    log_prob_blank: np.ndarray, log_prob_y: np.ndarray
) -> np.ndarray:
    """A naive NumPy implementation of log_prob_prefix_alignments."""
    T, _ = log_prob_blank.shape
    _, U = log_prob_y.shape
    dtype = log_prob_blank.dtype
    assert log_prob_blank.shape == (T, U + 1), f"{log_prob_blank.shape} vs. {log_prob_y.shape}"
    assert log_prob_y.shape == (T + 1, U), f"{log_prob_blank.shape} vs. {log_prob_y.shape}"
    assert log_prob_blank.dtype == log_prob_y.dtype

    neg_inf = -np.inf
    log_prob_prefix = np.full([T + 1, U + 1], fill_value=neg_inf, dtype=dtype)
    for t in range(T + 1):
        for u in range(U + 1):
            if t > 0 or u > 0:
                log_prob_prefix[t, u] = np.logaddexp(
                    log_prob_prefix[t - 1, u] + log_prob_blank[t - 1, u] if t > 0 else neg_inf,
                    log_prob_prefix[t, u - 1] + log_prob_y[t, u - 1] if u > 0 else neg_inf,
                )
            else:
                log_prob_prefix[t, u] = 0
            assert log_prob_prefix[t, u] <= 0, f"{t}, {u}: {log_prob_prefix[t, u]}"
    return log_prob_prefix


def log_prob_prefix_alignments_numeric_gradient(
    log_prob_blank: np.ndarray,
    log_prob_y: np.ndarray,
    x_type: str,
    t: int,
    u: int,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Computes the gradient of `numpy_log_prob_prefix_alignments[-1, -1]` numerically.

    Please see log_prob_alignments for the descriptions of log_prob_blank and log_prob_y.

    Args:
        log_prob_blank: an array of shape [T, U + 1].
        log_prob_y: an array of shape [T + 1, U].
        x_type: a string value of "blank" or "y".
        t: an integer in range [0, T) if x_type == "blank" [0, T+1) if x_type == "y".
        u: an integer in range [0, U+1) if x_type == "blank" [0, U) if x_type == "y".
        epsilon: the delta of log_prob_{blank,y}[t, u].

    Returns:
        The numeric gradient of d(log_prob_full_alignments) / d(log_prob_{x_type}[t, u]), where
        log_prob_full_alignments represents log(prob(y|x)).

    Raises:
        ValueError: If x_type is unsupported.
    """
    log_prob_base = numpy_log_prob_prefix_alignments(log_prob_blank, log_prob_y)
    log_prob_blank_prime = log_prob_blank
    log_prob_y_prime = log_prob_y
    if x_type == "blank":
        log_prob_blank_prime[t, u] -= epsilon
    elif x_type == "y":
        log_prob_y_prime[t, u] -= epsilon
    else:
        raise ValueError(f"Invalid {x_type}")
    log_prob_prime = numpy_log_prob_prefix_alignments(log_prob_blank_prime, log_prob_y_prime)
    return (log_prob_prime[-1, -1] - log_prob_base[-1, -1]) / -epsilon


def assert_all_close(x, y, atol=1e-5, rtol=1e-5, err_msg=None):
    np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)


# pylint: disable=no-self-use
class AlignmentTest(TestWithTemporaryCWD, tf.test.TestCase):
    def test_single_route(self):
        U = T = 4
        self._test_prefix_probs(
            prob_blank=(
                np.arange(T).reshape((T, 1)) <= np.arange(U + 1).reshape((1, U + 1))
            ).astype(np.float32),
            prob_y=(np.arange(T + 1).reshape((T + 1, 1)) > np.arange(U).reshape((1, U))).astype(
                np.float32
            ),
        )

    @parameterized.parameters(
        (1, 1),
        (2, 1),
        (1, 2),
        (4, 4),
    )
    def test_random(self, T, U):
        prob_blank = np.random.uniform(0, 1, (T + 1, U + 1))
        prob_y = np.random.uniform(0, 1, (T + 1, U + 1)) * (1 - prob_blank)
        self._test_prefix_probs(prob_blank[:T, :], prob_y[:, :U])

    def _test_prefix_probs(self, prob_blank, prob_y):
        """Checks that the numpy and jax implementations compute the same log probs."""
        logging.info("prob_blank=%s", prob_blank)
        logging.info("prob_y=%s", prob_y)
        log_prob_blank, log_prob_y = np.log(prob_blank), np.log(prob_y)
        numpy_log_prob_prefix = numpy_log_prob_prefix_alignments(
            log_prob_blank=log_prob_blank, log_prob_y=log_prob_y
        )
        logging.info("numpy_prob_prefix=%s", np.exp(numpy_log_prob_prefix))
        self._verify_prefix_probs(
            prob_blank=prob_blank, prob_y=prob_y, prob_prefix=np.exp(numpy_log_prob_prefix)
        )
        jax_log_prob_prefix = log_prob_prefix_alignments(
            log_prob_blank=log_prob_blank, log_prob_y=log_prob_y
        )
        self._verify_prefix_probs(
            prob_blank=prob_blank, prob_y=prob_y, prob_prefix=np.exp(jax_log_prob_prefix)
        )
        assert_all_close(numpy_log_prob_prefix, jax_log_prob_prefix)

        jax_log_prob_suffix = log_prob_suffix_alignments(
            log_prob_blank=log_prob_blank, log_prob_y=log_prob_y
        )
        self._check_prefix_suffix_probs(jax_log_prob_prefix, jax_log_prob_suffix)
        self._check_gradients(
            log_prob_blank=log_prob_blank,
            log_prob_y=log_prob_y,
            log_prob_prefix=jax_log_prob_prefix,
            log_prob_suffix=jax_log_prob_suffix,
        )

    def _check_prefix_suffix_probs(self, jax_log_prob_prefix, jax_log_prob_suffix):
        assert_all_close(jax_log_prob_suffix[0, 0], jax_log_prob_prefix[-1, -1])
        R, C = jax_log_prob_prefix.shape
        diagonal_log_sums = []
        for diagonal in range(R + C - 1):
            log_sum = -np.inf
            for r in range(R):
                c = diagonal - r
                if 0 <= c < C:
                    log_sum = np.logaddexp(
                        log_sum, jax_log_prob_prefix[r, c] + jax_log_prob_suffix[r, c]
                    )
            diagonal_log_sums.append(log_sum)
            assert_all_close(log_sum, jax_log_prob_suffix[0, 0], err_msg=f"diagonal={diagonal}")
        logging.info("prob_prefix=%s", np.exp(jax_log_prob_prefix))
        logging.info("prob_suffix=%s", np.exp(jax_log_prob_suffix))
        logging.info("diagonals=%s", diagonal_log_sums)

    def _check_gradients(
        self,
        *,
        log_prob_blank: jnp.ndarray,
        log_prob_y: jnp.ndarray,
        log_prob_prefix: jnp.ndarray,
        log_prob_suffix: jnp.ndarray,
    ):
        prob_blank, prob_y = np.exp(log_prob_blank), np.exp(log_prob_y)
        T = prob_blank.shape[0]
        U = prob_y.shape[1]
        self.assertSequenceEqual(log_prob_prefix.shape, (T + 1, U + 1))
        self.assertSequenceEqual(log_prob_suffix.shape, (T + 1, U + 1))
        grad_blank, grad_y = log_prob_gradients(
            log_prob_prefix=log_prob_prefix,
            log_prob_suffix=log_prob_suffix,
            log_prob_blank=log_prob_blank,
            log_prob_y=log_prob_y,
        )
        self.assertSequenceEqual(grad_blank.shape, log_prob_blank.shape)
        self.assertSequenceEqual(grad_y.shape, log_prob_y.shape)

        custom_grad_blank, custom_grad_y = jax.grad(log_prob_alignments, argnums=(0, 1))(
            log_prob_blank, log_prob_y
        )
        self.assertSequenceEqual(custom_grad_blank.shape, log_prob_blank.shape)
        self.assertSequenceEqual(custom_grad_y.shape, log_prob_y.shape)
        for _ in range(20):
            x_type = np.random.choice(["blank", "y"], size=[])
            if x_type == "blank":
                t = np.random.randint(low=0, high=T)
                u = np.random.randint(low=0, high=U + 1)
            else:
                t = np.random.randint(low=0, high=T + 1)
                u = np.random.randint(low=0, high=U)
            numeric_grad = log_prob_prefix_alignments_numeric_gradient(
                np.log(prob_blank), np.log(prob_y), x_type, t, u
            )
            if x_type == "blank":
                assert_all_close(grad_blank[t, u], numeric_grad)
                assert_all_close(custom_grad_blank[t, u], numeric_grad)
            else:
                assert_all_close(grad_y[t, u], numeric_grad)
                assert_all_close(custom_grad_y[t, u], numeric_grad)

    def _verify_prefix_probs(self, prob_blank, prob_y, prob_prefix):
        T_plus_1, U_plus_1 = prob_prefix.shape
        for t in range(T_plus_1):
            for u in range(U_plus_1):
                if t == u == 0:
                    self.assertAlmostEqual(prob_prefix[t, u], 1)
                else:
                    assert_all_close(
                        prob_prefix[t, u],
                        (prob_prefix[t - 1, u] * prob_blank[t - 1, u] if t > 0 else 0)
                        + (prob_prefix[t, u - 1] * prob_y[t, u - 1] if u > 0 else 0),
                    )

    def test_tilting(self):
        R, C = 3, 2
        x = jnp.arange(R * C, dtype=jnp.float32).reshape(R, C) + 1
        pad_value = jnp.finfo(x.dtype).min
        y = _tilt(x, pad_value=pad_value)
        for i in range(R + C - 1):
            for j in range(C):
                r = i - j
                if 0 <= r < R:
                    self.assertEqual(y[i, j], x[r, j])
                else:
                    self.assertEqual(y[i, j], pad_value)
        np.testing.assert_array_equal(_untilt(y), x)

    def test_apply_paddings(self):
        am_max_seq_len, lm_max_seq_len = 6, 5
        T, U = 3, 2
        prob_blank = np.random.uniform(0, 1, (am_max_seq_len, lm_max_seq_len))
        prob_y = np.random.uniform(0, 1, (am_max_seq_len, lm_max_seq_len)) * (1 - prob_blank)
        am_paddings = (jnp.arange(am_max_seq_len) >= T).astype(jnp.float32)
        lm_paddings = (jnp.arange(lm_max_seq_len) > U).astype(jnp.float32)
        log_prob_blank, log_prob_y = jnp.log(prob_blank), jnp.log(prob_y)

        # Mask log_prob_blank[T-1, :U] to terminate at a blank step.
        log_prob_blank_mask = jnp.expand_dims(
            jnp.arange(am_max_seq_len) == T - 1, axis=-1
        ) * jnp.expand_dims(jnp.arange(lm_max_seq_len) < U, axis=0)
        log_prob_blank_stripped = (
            log_prob_blank * (1 - log_prob_blank_mask) + _NEG_INF * log_prob_blank_mask
        )

        # Mask log_prob_y[T, :U] to terminate at a blank step.
        log_prob_y_mask = jnp.expand_dims(
            jnp.arange(am_max_seq_len) == T, axis=-1
        ) * jnp.expand_dims(jnp.arange(lm_max_seq_len) < U, axis=0)
        log_prob_y_stripped = log_prob_y * (1 - log_prob_y_mask) + _NEG_INF * log_prob_y_mask

        log_prob_stripped = log_prob_alignments(
            log_prob_blank=log_prob_blank_stripped[:T, : U + 1],
            log_prob_y=log_prob_y_stripped[: T + 1, :U],
        )
        padded = apply_paddings(
            log_prob_blank=log_prob_blank,
            log_prob_y=log_prob_y,
            am_paddings=am_paddings,
            lm_paddings=lm_paddings,
        )
        for k, v in padded.items():
            logging.info("padded %s=\n%s", k, jnp.exp(v))
        log_prob_padded = log_prob_alignments(**padded)
        assert_all_close(log_prob_stripped, log_prob_padded)

    def test_apply_paddings_check(self):
        """Test that apply_paddings raises run_time errors when lm_paddings is all 1s."""
        batch_size, am_max_len, lm_max_len = 3, 8, 6
        am_lengths = jnp.asarray([0, 8, 3])
        lm_lengths = jnp.asarray([0, 5, 0])
        prob_blank = np.random.uniform(0, 1, (batch_size, am_max_len, lm_max_len))
        prob_y = np.random.uniform(0, 1, (batch_size, am_max_len, lm_max_len)) * (1 - prob_blank)
        am_paddings = jnp.expand_dims(jnp.arange(am_max_len), axis=0) >= jnp.expand_dims(
            am_lengths, axis=-1
        )
        lm_paddings = jnp.expand_dims(jnp.arange(lm_max_len), axis=0) >= jnp.expand_dims(
            lm_lengths, axis=-1
        )
        log_prob_blank, log_prob_y = jnp.log(prob_blank), jnp.log(prob_y)

        with runtime_checks():
            with self.assertRaisesRegex(
                jaxlib.xla_extension.XlaRuntimeError,
                "lm_paddings cannot be all 1s.",
            ):
                jax.jit(jax.vmap(apply_paddings))(
                    log_prob_blank=log_prob_blank,
                    log_prob_y=log_prob_y,
                    am_paddings=am_paddings,
                    lm_paddings=lm_paddings,
                )
        check_apply_paddings = checkify.checkify(apply_paddings, errors=checkify.user_checks)
        err, _ = jax.jit(jax.vmap(check_apply_paddings))(
            log_prob_blank=log_prob_blank,
            log_prob_y=log_prob_y,
            am_paddings=am_paddings,
            lm_paddings=lm_paddings,
        )
        with self.assertRaises(checkify.JaxRuntimeError):
            err.throw()

    @parameterized.product(
        logits_to_log_probs_cfg=(
            config_for_function(classic_logits_to_log_probs),
            config_for_function(hat_logits_to_log_probs),
        ),
        blank_id=(0, 5),
        blank_logit_bias=(0, -1, 1),
    )
    def test_logits_to_log_probs(self, logits_to_log_probs_cfg, blank_id, blank_logit_bias):
        """Validates the outputs of LogitsToLogProbFn."""
        batch_size, vocab_size = 2, 8
        logits = np.random.uniform(-1, 1, (batch_size, vocab_size))
        logits_to_log_probs_cfg.blank_id = blank_id
        logits_to_log_probs_cfg.blank_logit_bias = blank_logit_bias
        results = logits_to_log_probs_cfg.instantiate()(logits)
        self.assertCountEqual(("log_prob_blank", "log_prob_tokens"), results.keys())
        log_prob_blank = results["log_prob_blank"]
        log_prob_tokens = results["log_prob_tokens"]
        print(f"log_prob_blank={log_prob_blank}")
        print(f"log_prob_tokens={log_prob_tokens}")

        # The outputs of LogitsToLogProbFn have the right shapes.
        self.assertSequenceEqual((batch_size,), log_prob_blank.shape)
        self.assertSequenceEqual((batch_size, vocab_size), log_prob_tokens.shape)

        # log_prob_tokens[blank_id] =~ -inf.
        self.assertSequenceEqual(
            (log_prob_tokens[:, logits_to_log_probs_cfg.blank_id] < -1e3).tolist(),
            [True] * batch_size,
        )

        # sum(prob_blank + prob_tokens) =~ 1.
        prob_blank = jnp.exp(log_prob_blank)
        prob_tokens = jnp.exp(jax.nn.logsumexp(log_prob_tokens, axis=-1))
        assert_all_close(prob_tokens + prob_blank, jnp.ones([batch_size]))

        # Test that log_probs_from_blank_and_tokens correctly gets back to log probs.
        log_probs = log_probs_from_blank_and_tokens(
            log_prob_blank, log_prob_tokens, blank_id=blank_id
        )
        ref_log_probs = np.array(log_prob_tokens)
        ref_log_probs[..., blank_id] = np.array(log_prob_blank)
        assert_all_close(log_probs, ref_log_probs)

        # Incrementing blank_logit_bias increases log_prob_blank.
        cfg_with_higher_blank_logit_bias = logits_to_log_probs_cfg.clone()
        cfg_with_higher_blank_logit_bias.blank_logit_bias += 1
        results_with_blank_logit_bias = cfg_with_higher_blank_logit_bias.instantiate()(logits)
        print(f"results_with_blank_logit_bias={results_with_blank_logit_bias}")
        self.assertSequenceEqual(
            (results_with_blank_logit_bias["log_prob_blank"] > log_prob_blank).tolist(),
            [True] * batch_size,
        )

    @parameterized.parameters(2, 3, 4, 5)
    def test_log_probs_from_blank_and_tokens(self, ndim):
        batch_size, vocab_size = 4, 8
        rand_dim = np.random.randint(size=ndim - 1, low=2, high=16).tolist()
        blank_id = np.random.randint(low=0, high=8)
        logits = np.random.uniform(-5, 5, [batch_size] + rand_dim + [vocab_size])
        cfg = config_for_function(classic_logits_to_log_probs)
        cfg.blank_id = blank_id
        results = cfg.instantiate()(logits)

        log_probs = log_probs_from_blank_and_tokens(
            log_prob_blank=results["log_prob_blank"],
            log_prob_tokens=results["log_prob_tokens"],
            blank_id=blank_id,
        )
        ref_log_probs = np.array(results["log_prob_tokens"])
        ref_log_probs[..., blank_id] = np.array(results["log_prob_blank"])
        assert_all_close(log_probs, ref_log_probs)

    def _setup_transducer(self, input_dim, vocab_size, prng_key):
        cfg = Transducer.default_config().set(
            name="transducer", input_dim=input_dim, vocab_size=vocab_size
        )
        cfg.logits_to_log_probs.blank_id = 0
        layer = cfg.instantiate(parent=None)
        prng_key, init_key = jax.random.split(prng_key)
        layer_params = layer.initialize_parameters_recursively(init_key)
        return layer, layer_params, prng_key

    def _setup_random_data(
        self, batch_size, input_dim, vocab_size, am_max_len, lm_max_len, am_lengths, lm_lengths
    ):
        am_data = np.random.uniform(-1, 1, (batch_size, am_max_len, input_dim))
        lm_data = np.random.uniform(-1, 1, (batch_size, lm_max_len, input_dim))

        am_paddings = jnp.expand_dims(jnp.arange(am_max_len), axis=0) >= jnp.expand_dims(
            am_lengths, axis=-1
        )

        lm_paddings = jnp.expand_dims(jnp.arange(lm_max_len), axis=0) >= jnp.expand_dims(
            lm_lengths, axis=-1
        )
        # target_labels are never blank_id.
        target_labels = np.random.uniform(1, vocab_size, (batch_size, lm_max_len)).astype(np.int32)
        return am_data, am_paddings, lm_data, lm_paddings, target_labels

    def test_transducer(self):
        """Test batch prediction."""
        batch_size, input_dim, vocab_size = 4, 2, 3
        am_max_len, lm_max_len = 8, 6
        am_lengths = jnp.asarray([3, 8, 8, 0])
        lm_lengths = jnp.asarray([3, 6, 1, 6])
        prng_key = jax.random.PRNGKey(123)
        layer, layer_params, prng_key = self._setup_transducer(
            input_dim=input_dim, vocab_size=vocab_size, prng_key=prng_key
        )
        am_data, am_paddings, lm_data, lm_paddings, target_labels = self._setup_random_data(
            batch_size, input_dim, vocab_size, am_max_len, lm_max_len, am_lengths, lm_lengths
        )

        # Compute log_prob_{blank,tokens,y} by looping over (batch_size, am_max_len, lm_max_len).
        log_prob_blank = np.zeros([batch_size, am_max_len, lm_max_len])
        log_prob_tokens = np.zeros([batch_size, am_max_len, lm_max_len, vocab_size])
        log_prob_y = np.zeros([batch_size, am_max_len, lm_max_len])
        for i in range(batch_size):
            for t in range(am_max_len):
                for u in range(lm_max_len):
                    prediction, _ = F(
                        layer,
                        method="predict",
                        inputs=dict(
                            am_data=am_data[i, t : t + 1, :], lm_data=lm_data[i, u : u + 1, :]
                        ),
                        is_training=True,
                        state=layer_params,
                        prng_key=prng_key,
                    )
                    log_prob_blank[i, t, u] = prediction["log_prob_blank"][0, 0]
                    log_prob_tokens[i, t, u] = prediction["log_prob_tokens"][0, 0]
                    log_prob_y[i, t, u] = prediction["log_prob_tokens"][0, 0, target_labels[i, u]]

        # Batch prediction is the same as concatenating predictions for all (i, t, u).
        batch_prediction, _ = F(
            layer,
            method="predict",
            inputs=dict(am_data=am_data, lm_data=lm_data),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        assert_all_close(batch_prediction["log_prob_blank"], log_prob_blank)
        assert_all_close(batch_prediction["log_prob_tokens"], log_prob_tokens)

        # Apply paddings to log_prob_{blank,y} and compare the results to those returned by
        # forward().
        padded = jax.vmap(apply_paddings)(
            log_prob_blank=log_prob_blank,
            log_prob_y=log_prob_y,
            am_paddings=am_paddings,
            lm_paddings=lm_paddings,
        )
        # Check forward() results.
        (loss, per_example), _ = F(
            layer,
            inputs=dict(
                am_data=am_data,
                am_paddings=am_paddings,
                lm_data=lm_data,
                lm_paddings=lm_paddings,
                target_labels=target_labels,
            ),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        # forward() uses padded log_prob_{blank,y} to compute the losses.
        assert_all_close(per_example["log_prob_blank"], padded["log_prob_blank"])
        assert_all_close(per_example["log_prob_y"], padded["log_prob_y"])
        # A sequence is valid iff am_lengths > 0 and lm_lengths > 0.
        assert_all_close(
            per_example["is_valid_example"], jnp.logical_and(am_lengths > 0, lm_lengths > 0)
        )
        assert_all_close(per_example["loss"] > 0, per_example["is_valid_example"])
        # Only the first two example are valid and the loss is the mean of the losses on these
        # two examples.
        assert_all_close(loss, per_example["loss"][:3].mean())

    def test_transducer_loss_and_grad(self):
        """Test batch forward and gradient computation under padding."""
        batch_size, input_dim, vocab_size = 7, 2, 3
        am_max_len, lm_max_len = 8, 6
        # T.
        am_lengths = jnp.asarray([3, 8, 8, 8, 0, 0, 0])
        # U + 1.
        lm_lengths = jnp.asarray([3, 1, 5, 6, 1, 2, 6])
        prng_key = jax.random.PRNGKey(456)
        layer, layer_params, prng_key = self._setup_transducer(
            input_dim=input_dim, vocab_size=vocab_size, prng_key=prng_key
        )
        am_data, am_paddings, lm_data, lm_paddings, target_labels = self._setup_random_data(
            batch_size, input_dim, vocab_size, am_max_len, lm_max_len, am_lengths, lm_lengths
        )
        # [batch, 1, lm_max_len, vocab]
        labels_onehot = jax.nn.one_hot(target_labels, vocab_size)[:, None, :, :]

        def _loss(
            log_prob_blank: Tensor, log_prob_y: Tensor, am_pad: Tensor, lm_pad: Tensor
        ) -> tuple[Tensor, NestedTensor]:
            """Compute loss and grad from prediction outputs."""
            log_probs_mask = jax.vmap(apply_paddings)(
                log_prob_blank=log_prob_blank,
                log_prob_y=log_prob_y,
                am_paddings=am_pad,
                lm_paddings=lm_pad,
            )
            log_prob_seq = jax.vmap(log_prob_alignments)(**log_probs_mask)
            is_valid_example = jnp.logical_and((1 - lm_pad).sum(axis=-1), (1 - am_pad).sum(axis=-1))
            per_example_loss = -log_prob_seq * is_valid_example
            per_example = dict(
                log_prob_alignments=log_prob_seq,
                is_valid_example=is_valid_example,
                loss=per_example_loss,
                **log_probs_mask,
            )
            return per_example_loss.sum(), per_example

        prediction_batch, _ = F(
            layer,
            method="predict",
            inputs=dict(am_data=am_data, lm_data=lm_data),
            is_training=True,
            state=layer_params,
            prng_key=prng_key,
        )
        log_prob_blank_batch = prediction_batch["log_prob_blank"]
        log_prob_y_batch = (prediction_batch["log_prob_tokens"] * labels_onehot).sum(axis=-1)

        (loss_sum_batch, log_probs_batch), (grad_blank_batch, grad_y_batch) = jax.value_and_grad(
            _loss, argnums=(0, 1), has_aux=True
        )(
            log_prob_blank_batch,
            log_prob_y_batch,
            am_paddings,
            lm_paddings,
        )
        loss = 0

        def compare_fn(a, b):
            # Compare b's upper-left entries.
            assert_all_close(a, jax.lax.slice(b, [0] * a.ndim, a.shape), atol=1e-6, rtol=1e-6)

        for i in range(batch_size):
            T = am_lengths[i]
            U_plus_1 = lm_lengths[i]
            prediction_i, _ = F(
                layer,
                method="predict",
                inputs=dict(
                    am_data=am_data[i : i + 1, : jnp.maximum(T, 1)],
                    lm_data=lm_data[i : i + 1, :U_plus_1],
                ),
                is_training=True,
                state=layer_params,
                prng_key=prng_key,
            )
            log_prob_blank_i = prediction_i["log_prob_blank"]
            log_prob_y_i = (
                prediction_i["log_prob_tokens"] * labels_onehot[i : i + 1, :, :U_plus_1, :]
            ).sum(axis=-1)

            (loss_i, log_probs_i), (grad_blank_i, grad_y_i) = jax.value_and_grad(
                _loss, argnums=(0, 1), has_aux=True
            )(
                log_prob_blank_i,
                log_prob_y_i,
                am_paddings[i : i + 1, : jnp.maximum(T, 1)],
                lm_paddings[i : i + 1, :U_plus_1],
            )
            loss += loss_i

            # Batch forward with padded length is the same as forward with exact length.
            jax.tree.map(
                lambda x, y, n=i: compare_fn(x, y[n : n + 1]), log_probs_i, log_probs_batch
            )
            # Batch backward with padded length is the same as backward with exact length.
            compare_fn(grad_blank_i, grad_blank_batch[i : i + 1])
            compare_fn(grad_y_i, grad_y_batch[i : i + 1])

            # Gradient on the padded positions are all zero.
            valid_position_i = jnp.expand_dims(
                jnp.arange(am_max_len) >= am_lengths[i], axis=-1
            ) * jnp.expand_dims(jnp.arange(lm_max_len) >= lm_lengths[i], axis=0)
            self.assertAllEqual(
                grad_blank_batch[i : i + 1] * valid_position_i,
                jnp.zeros((1, am_max_len, lm_max_len)),
            )
            self.assertAllEqual(
                grad_y_batch[i : i + 1] * valid_position_i, jnp.zeros((1, am_max_len, lm_max_len))
            )
        assert_all_close(loss, loss_sum_batch, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    absltest.main()
