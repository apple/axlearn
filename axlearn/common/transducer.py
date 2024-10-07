# Copyright © 2023 Apple Inc.

"""An implementation of sequence transducer model.

Reference:
https://arxiv.org/abs/1211.3711
https://ieeexplore.ieee.org/document/8639690

Suppose we have three frames (T=3): F0, F1, F2 and two labels in the sequence (U=2): A, B.

The AM inputs to the transducer has T frames. There is an EXIT state FE at the end.
On the row of FE, only one state is reachable with a blank step.

The LM inputs will have U + 1 tokens, with a BOS token at the beginning.

Combining the T+1 states and U+1 tokens will give us a (T+1, U+1, vocab_size) tensor with a vector
of dim vocab_size at each point of the (T+1, U+1) matrix:

    BOS   A    B   (U = 2)
F0   @--->.--->.
     |    |    |
     v    v    v
F1   .--->.--->.
     |    |    |
     v    v    v
F2   .--->.--->.
     .    .    |
     .    .    v
FE   .    .    $


where
* log_prob_blank = log_prob_vocab[:T, :U + 1, blank]
* log_prob_y = log_prob_vocab[:T + 1, :U, y], where y = A/B, for u = 0/1, respectively.
"""


import jax
from chex import dataclass
from jax import numpy as jnp
from jax.experimental import checkify
from typing_extensions import Protocol

from axlearn.common.base_layer import BaseLayer
from axlearn.common.config import (
    REQUIRED,
    InstantiableConfig,
    Required,
    config_class,
    config_for_function,
)
from axlearn.common.layers import Linear, get_activation_fn
from axlearn.common.module import Module
from axlearn.common.utils import NestedTensor, Tensor

_NEG_INF = -1e30


class LogitsToLogProbFn(Protocol):
    """A function that consumes logits and produces modified logits."""

    def __call__(self, logits: Tensor):
        """Computes blank and token log_probs from the given logits.

        Args:
            logits: a Tensor of shape [..., vocab_size].

        Returns:
            A dict containing:
            - "log_prob_blank": a Tensor of shape [...].
            - "log_prob_tokens": a Tensor of shape [..., vocab_size].
            log_prob_blank and log_prob_tokens are properly normalized, such that
            sum(exp(log_prob_blank) + exp(log_prob_tokens)) =~ 1.
        """
        raise NotImplementedError(type(self))


def classic_logits_to_log_probs(
    *,
    blank_id: int,
    blank_logit_bias: float = 0,
) -> LogitsToLogProbFn:
    """Computes blank and token log_probs from the given logits.

    ... according to the classic (https://arxiv.org/abs/1211.3711) formulation.

    Args:
        blank_id: an int in range [0, vocab_size) representing the blank id.
        blank_logit_bias: a scalar bias to be applied on the blank logit before sigmoid.

    Returns:
        A LogitsToLogProbFn.
    """

    def fn(logits: Tensor):
        vocab_size = logits.shape[-1]
        blank_id_onehot = jax.nn.one_hot(blank_id, vocab_size)
        log_prob_tokens = jax.nn.log_softmax(logits + blank_logit_bias * blank_id_onehot)
        log_prob_blank = log_prob_tokens[..., blank_id]
        log_prob_tokens += _NEG_INF * blank_id_onehot
        return dict(log_prob_blank=log_prob_blank, log_prob_tokens=log_prob_tokens)

    return fn


def hat_logits_to_log_probs(*, blank_id: int, blank_logit_bias: float = 0):
    """Computes blank and token log_probs from the given logits.

    ... according to the HAT (https://arxiv.org/abs/2003.07705) formulation.

    Args:
        blank_id: an int in range [0, vocab_size) representing the blank id.
        blank_logit_bias: a scalar bias to be applied on the blank logit before sigmoid.

    Returns:
        A LogitsToLogProbFn.
    """

    def fn(logits: Tensor):
        blank_logits = logits[..., blank_id] + blank_logit_bias
        #   log_prob_blank
        # = log(sigmoid(blank_logits))
        # = log(1 / (1 + exp(-blank_logits)))
        # = -softplus(-blank_logits)
        log_prob_blank = -jax.nn.softplus(-blank_logits)
        # log_prob_not_blank = log(sigmoid(-blank_logits)) = -softplus(blank_logits).
        log_prob_not_blank = -jax.nn.softplus(blank_logits)
        # Set logits[blank_id] = -inf.
        vocab_size = logits.shape[-1]
        logits += _NEG_INF * jax.nn.one_hot(blank_id, vocab_size)
        log_prob_tokens = jax.nn.log_softmax(logits) + jnp.expand_dims(log_prob_not_blank, -1)
        return dict(log_prob_blank=log_prob_blank, log_prob_tokens=log_prob_tokens)

    return fn


def log_probs_from_blank_and_tokens(
    log_prob_blank: Tensor, log_prob_tokens: Tensor, *, blank_id: int
):
    """Computes full log_probs tensor from log_prob_blank and log_prob_tokens.

    Args:
        log_prob_blank: a Tensor of shape [...].
        log_prob_tokens: a Tensor of shape [..., vocab_size].
        blank_id: an int in range [0, vocab_size) representing the blank id.

    Returns:
        log_probs: a Tensor of shape [..., vocab_size].
            log_probs[..., id] = log_prob_blank if id == blank_id else log_prob_tokens.
    """
    vocab_size = log_prob_tokens.shape[-1]
    blank_id_onehot = jax.nn.one_hot(blank_id, vocab_size, dtype=jnp.int32)
    log_probs = log_prob_blank[..., None] * blank_id_onehot + log_prob_tokens * (
        1 - blank_id_onehot
    )
    return log_probs


class Transducer(BaseLayer):
    """A sequence transducer model.

    It outputs log probabilities of blank or label tokens.
    During training, Transducer takes encoded acoustic sequences and token sequences as
    inputs and computes the alignment probabilities and the losses.

    During inference, Transducer takes one encoded acoustic frame and the encoded previous
    token as inputs and predicts whether to move to the next acoustic frame ('log_prob_blank') or
    emit the next token ('log_prob_tokens').
    """

    @config_class
    class Config(BaseLayer.Config):
        # The acoustic and language input dims.
        input_dim: Required[int] = REQUIRED
        # The vocab size.
        vocab_size: Required[int] = REQUIRED
        activation_fn: str = "nn.tanh"
        proj: Linear.Config = Linear.default_config()
        # A config that instantiates to a LogitsToLogProbFn.
        logits_to_log_probs: InstantiableConfig = config_for_function(classic_logits_to_log_probs)

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg: Transducer.Config = self.config
        self._add_child("proj", cfg.proj.set(input_dim=cfg.input_dim, output_dim=cfg.vocab_size))
        self._logits_to_log_probs: LogitsToLogProbFn = cfg.logits_to_log_probs.instantiate()

    def forward(
        self,
        *,
        am_data: Tensor,
        am_paddings: Tensor,
        lm_data: Tensor,
        lm_paddings: Tensor,
        target_labels: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Computes the transducer loss (for training and evaluation).

        Args:
            am_data: a Tensor of shape [batch_size, am_max_len, input_dim].
            am_paddings: a 0/1 Tensor of shape [batch_size, am_max_len] of 0's followed 1's,
                where 1 represents a trailing padding frame.
            lm_data: a Tensor of shape [batch_size, lm_max_len, input_dim], usually computed
                from [BOS] + target_labels.
            lm_paddings: a 0/1 Tensor of shape [batch_size, lm_max_len] of 0's followed 1's,
                where 1 represents a padding token. lm_paddings should be 0 for BOS and the
                target labels. It has U+1 0 entries.
            target_labels: an int Tensor of shape [batch_size, lm_max_len]. Does not include
                BOS or EOS.

        Returns:
            (loss, aux), where `loss` is a scalar representing the transducer loss and `aux` is
            a dict containing the following keys:
            - "log_prob_blank": [batch_size, am_max_len, lm_max_len], log probability of blank.
            - "log_prob_y":  [batch_size, am_max_len, lm_max_len], log probability of the
              groundtruth token.
            - "log_prob_alignments": [batch_size], per-example log probability of alignments.
            - "is_valid_example": [batch_size], 1/0 representing whether the example is valid or
              padding.
            - "loss": [batch_size], per-example loss, 0 if it is a padding example.
        """
        cfg = self.config

        @dataclass
        class Seq:
            data: Tensor
            paddings: Tensor

        def example_fn(am_i: Seq, lm_i: Seq, labels_i: Tensor) -> NestedTensor:
            """Computes log_probs for one example."""
            # [lm_max_len, vocab_size].
            labels_i_onehot = jax.nn.one_hot(labels_i, cfg.vocab_size)

            # Below we use a map loop over am_max_len to compute log_prob_{blank,y}.
            #
            # The results are equivalent to self.predict(am_i.data, lm_i.data), but the loop is more
            # memory efficient since it avoids a tensor of shape [batch, am_len, lm_len, vocab].
            # The memory saving is possible since we don't need to keep logits over the entire
            # vocab, but only log_prob_{blank,y}. So we can compute softmax for each acoustic
            # frame and only keep two out of the vocab_size logits.
            #
            # Alternatively we can loop over labels (lm_max_len), but:
            # (1) looping over time can save a transpose, since our loss functions expects (T, U)
            #     matrices;
            # (2) we expect T and U to be approximately equal after acoustic subsampling.
            #
            # Yet another possibility is to use a chunk-wise loop to reduce number of iterations
            # in the map loop. We may consider this if this loop turns out to be the bottleneck
            # of transducer training.
            def map_fn(am_t: Seq) -> tuple[Tensor, Tensor]:
                # [1, ...].
                am_t = jax.tree.map(lambda x: jnp.expand_dims(x, 0), am_t)
                # [1, lm_max_len, ...].
                prediction_t = self.predict(am_t.data, lm_i.data)
                # [1, lm_max_len].
                log_prob_y = (prediction_t["log_prob_tokens"] * labels_i_onehot).sum(axis=-1)
                return prediction_t["log_prob_blank"][0], log_prob_y[0]

            # Loop over time (am_max_len) to compute log_probs of shape [am_max_len, lm_max_len].
            log_prob_blank, log_prob_y = jax.lax.map(map_fn, xs=am_i)
            _, log_probs = checkify.checkify(apply_paddings, errors=checkify.user_checks)(
                log_prob_blank=log_prob_blank,
                log_prob_y=log_prob_y,
                am_paddings=am_i.paddings,
                lm_paddings=lm_i.paddings,
            )
            return dict(log_prob_alignments=log_prob_alignments(**log_probs), **log_probs)

        # [batch_size].
        per_example = jax.vmap(example_fn)(
            Seq(data=am_data, paddings=am_paddings),
            Seq(data=lm_data, paddings=lm_paddings),
            target_labels,
        )
        # [batch_size].
        is_valid_example = jnp.logical_and(
            (1 - lm_paddings).sum(axis=-1), (1 - am_paddings).sum(axis=-1)
        )
        per_example["is_valid_example"] = is_valid_example
        per_example["loss"] = -per_example["log_prob_alignments"] * is_valid_example
        loss = per_example["loss"].sum() / jnp.maximum(1, is_valid_example.sum())
        return loss, per_example

    def predict(self, am_data: Tensor, lm_data: Tensor):
        """Computes the log probabilities of blank and token outputs.

        During inference, am_max_len and lm_max_len should be both 1.

        Args:
            am_data: a Tensor of shape [..., am_max_len, input_dim].
            lm_data: a Tensor of shape [..., lm_max_len, input_dim].

        Returns:
            A dict containing:
            - "log_prob_blank": a Tensor of shape [..., am_max_len, lm_max_len].
            - "log_prob_tokens": a Tensor of shape [..., am_max_len, lm_max_len, vocab_size].
        """
        cfg: Transducer.Config = self.config
        # [..., am_max_len, lm_max_len, input_dim].
        hidden = get_activation_fn(cfg.activation_fn)(
            jnp.expand_dims(am_data, -2) + jnp.expand_dims(lm_data, -3)
        )
        # [..., am_max_len, lm_max_len, vocab_size].
        logits = self.proj(hidden)
        return self._logits_to_log_probs(logits)


def _tilt(x: Tensor, pad_value: Tensor) -> Tensor:
    """Tilts `x` by 45 degrees clockwise.

    Args:
        x: an array of shape (R, C).

    Returns:
        y, an array of shape (R + C - 1, C), s.t.
            * y[i, j] == x[i - j, j] if 0 <= i - j < R.
            * y[i, j] == pad_value if i - j < 0 or i - j >= R.
        Or, in other words, x[i, j] is placed in y[i + j, j].
    """
    r, c = x.shape
    # [R + C, C].
    x = jnp.pad(x, ((0, c), (0, 0)), constant_values=pad_value)
    # [C, R + C].
    x = jnp.transpose(x, (1, 0))
    # [C * (R + C)].
    x = x.reshape(-1)
    # [C * (R + C - 1)].
    x = x[:-c]
    # [C, (R + C - 1)].
    x = x.reshape((c, r + c - 1))
    # [R + C - 1, C].
    return jnp.transpose(x, (1, 0))


def _untilt(y: Tensor) -> Tensor:
    """Untilts `y`, i.e., undo _tilt().

    Args:
        y: an array of shape (R + C - 1, C).

    Returns:
        x, an array of shape (R, C), s.t.
            * y[i, j] == x[i - j, j] if 0 <= i - j < R.
            * y[i, j] == 0 if i - j < 0 or i - j >= R.
        Or, in other words, x[i, j] is placed in y[i + j, j].
    """
    r_c_1, c = y.shape
    r = r_c_1 + 1 - c
    # [C, R + C - 1].
    y = y.transpose(1, 0)
    # [C * (R + C - 1)].
    y = y.reshape(-1)
    # [C * (R + C)].
    y = jnp.pad(y, (0, c))
    # [C, R + C].
    y = y.reshape((c, r + c))
    # [C, R].
    y = y[:, :r]
    # [R, C].
    return y.transpose(1, 0)


def log_prob_prefix_alignments(log_prob_blank: Tensor, log_prob_y: Tensor) -> Tensor:
    """Computes log(probability) of alignments between each prefix pair.

    Given input x[0:T] and labels y[0:U], computes the log sum probability of all forward alignments
    of prefixes x[0:t] and y[0:u].

    This is log(alpha[t, u]) in https://arxiv.org/abs/1211.3711.

    This implementation follows
    'Efficient Implementation of Recurrent Neural Network Transducer in Tensorflow',
    by T. Bagby et al. (2018), https://ieeexplore.ieee.org/document/8639690.

    Args:
        log_prob_blank: an array of shape [T, U + 1], where
            log_prob_blank[t, u] = log(prob(blank | x[0:t] aligns with y[0:u])),
            where 0 <= t < T and 0 <= u <= U.
        log_prob_y: an array of shape [T + 1, U], where
            log_prob_blank[t, u] = log(prob(y_u | x[0:t] aligns with y[0:u])),
            where 0 <= t <= T and 0 <= u < U.

    Returns:
        log_prob_prefix of shape [T + 1, U + 1], where
        log_prob_prefix[t, u] = log(prob(y[0:u] | x[0:t])) where 0 <= t <= T and 0 <= u <= U.
    """
    _, u = log_prob_y.shape
    dtype = log_prob_blank.dtype
    pad_value = _NEG_INF

    # Tilt lob_prob_* s.t. tilted_log_prob_*[t + u, u] = log_prob_*[t, u].
    # [T + U, U + 1].
    tilted_log_prob_blank = _tilt(log_prob_blank, pad_value=pad_value)
    # [T + U, U].
    tilted_log_prob_y = _tilt(log_prob_y, pad_value=pad_value)

    # Compute alpha[t, u] one diagonal at a time: for 0 <= k <= T + U, where k = t + u.
    # Each row in 'xs' represents a diagonal in log_prob_*.
    xs = (tilted_log_prob_blank, tilted_log_prob_y)
    # [U + 1]. carry0[0] = 0, carry0[i] = -inf for 0 < i <= U.
    carry0 = jnp.log(jax.nn.one_hot(0, u + 1, dtype=dtype))

    def scan_fn(carry: Tensor, xs: Tensor):
        """Computes the next diagonal of alpha[t, u].

        Args:
            carry: the k'th diagonal of log_alpha[t, u] where t + u == k, 0 <= k < T + U,
                carry[i] = log_alpha[k - i, i] for 0 <= i <= U.
            xs: (b, y), represents the k'th diagonal of log_prob_blank and log_prob_y, respectively.
                b[i] = log_prob_blank[k - i, i], y[i] = log_prob_y[k - i, i] for 0 <= i <= U.

        Returns:
            (carry', carry'), where carry' represents the (k+1)'th diagonal of alpha[t, u]:
            carry'[i] = log_alpha[k + 1 - i, i] for 0 <= i <= U.

            carry'[i] can be computed from carry, b, and y as follows:
              carry'[i]
            = log_alpha[k + 1 - i, i]
            = logsumexp(log_alpha[k - i, i] + log_prob_blank[k - i, i],
                        log_alpha[k + 1 - i, i - 1] + log_prob_y[k + 1 - i, i])
            = logsumexp(carry[i] + b[i], carry[i - 1] + y[i - 1])

            (We assume that carry[-1] + y[-1] == -inf.)
        """
        b, y = xs
        carry = jnp.logaddexp(b + carry, jnp.pad(y + carry[:-1], (1, 0), constant_values=-jnp.inf))
        return carry, carry

    # ys.shape = [T + U, U + 1].
    _, ys = jax.lax.scan(scan_fn, carry0, xs)
    # [T + U + 1, U + 1].
    tilted_log_prob_prefix = jnp.concatenate([carry0.reshape(1, -1), ys], axis=0)
    # [T + 1, U + 1].
    log_prob_prefix = _untilt(tilted_log_prob_prefix)
    return log_prob_prefix


def log_prob_suffix_alignments(log_prob_blank: Tensor, log_prob_y: Tensor) -> Tensor:
    """Computes log(beta(t,u)) in https://arxiv.org/abs/1211.3711.

    See log_prob_prefix_alignments for descriptions of args.
    """
    log_prob_blank_reversed = log_prob_blank[::-1, ::-1]
    log_prob_y_reversed = log_prob_y[::-1, ::-1]
    log_prob_suffix_reversed = log_prob_prefix_alignments(
        log_prob_blank_reversed, log_prob_y_reversed
    )
    return log_prob_suffix_reversed[::-1, ::-1]


def log_prob_gradients(
    *,
    log_prob_blank: Tensor,
    log_prob_y: Tensor,
    log_prob_prefix: Tensor,
    log_prob_suffix: Tensor,
):
    """Computes gradients of d(log(prob(y|x))) / d(log_prob_blank) and d(log_prob_y).

    Args:
        log_prob_blank: an array of shape [T, U + 1], where
            log_prob_blank[t, u] = log(prob(blank | x[0:t] aligns with y[0:u])),
            where 0 <= t < T and 0 <= u <= U.
        log_prob_y: an array of shape [T + 1, U], where
            log_prob_blank[t, u] = log(prob(y_u | x[0:t] aligns with y[0:u])),
            where 0 <= t <= T and 0 <= u < U.
        log_prob_prefix: an array of shape [T + 1, U + 1], where
            log_prob_prefix[t, u] = log(prob(y[:u] | x[:t])) where 0 <= t <= T and 0 <= u <= U.
        log_prob_suffix: an array of shape [T + 1, U + 1], where
            log_prob_prefix[t, u] = log(prob(y[u:] | x[t:])) where 0 <= t <= T and 0 <= u <= U.

    Returns:
        (grad_blank, grad_y), where
        - grad_blank has the same shape as log_log_prob_blank and represents
          d(log(prob(y|x))) / d(log_prob_blank);
        - grad_y has the same shape as log_log_prob_y and represents
          d(log(prob(y|x))) / d(log_prob_y);
    """
    log_prob_full_alignments = log_prob_suffix[0, 0]
    # Let P=prob(y|x):
    # log_d_p_blank[t, u] = log(d(P) / d(prob_blank[t, u])) = log(alpha[t, u] * beta[t + 1, u])
    # log_d_p_y[t, u] = log(d(P) / d(p_y[t, u])) = log(alpha[t, u] * beta[t, u + 1])
    log_d_p_blank = log_prob_prefix[:-1] + log_prob_suffix[1:]
    log_d_p_y = log_prob_prefix[:, :-1] + log_prob_suffix[:, 1:]
    #   d(log(P)) / d(log(p_blank))
    # = d(log(P)) / d(P) * d(P) / d(p_blank) * d(p_blank) / d(log(p_blank))
    #
    # Since:
    # 1. d(log(P)) / d(P) = 1/P = exp(-log_prob_full_alignments).
    # 2. d(P) / d(p_blank) = exp(log_d_p_blank)
    # 3. d(p_blank) / d(log(p_blank)) = p_blank = exp(log_prob_blank).
    #
    # d(log(P)) / d(log(p_blank)) = exp(-log_prob_full_alignments + log_d_p_blank + log_prob_blank)
    return (
        jnp.exp(-log_prob_full_alignments + log_d_p_blank + log_prob_blank),
        jnp.exp(-log_prob_full_alignments + log_d_p_y + log_prob_y),
    )


@jax.custom_vjp
def log_prob_alignments(log_prob_blank: Tensor, log_prob_y: Tensor):
    """Computes log(probability) of transducer alignment.

    Given input x[0:T] and labels y[0:U], computes the log sum probability of all alignments
    of x and y.

    This is log(alpha(T, U)) in https://arxiv.org/abs/1211.3711.

    This implementation follows
    'Efficient Implementation of Recurrent Neural Network Transducer in Tensorflow',
    by T. Bagby et al. (2018), https://ieeexplore.ieee.org/document/8639690.

    Args:
        log_prob_blank: an array of shape [T, U + 1], where
            log_prob_blank[t, u] = log(prob(blank | x[0:t] aligns with y[0:u])),
            where 0 <= t < T and 0 <= u <= U.
        log_prob_y: an array of shape [T + 1, U], where
            log_prob_blank[t, u] = log(prob(y_u | x[0:t] aligns with y[0:u])),
            where 0 <= t <= T and 0 <= u < U.

    Returns:
        A scalar, representing log(prob(y | x)).
    """
    log_prob_prefix = log_prob_prefix_alignments(log_prob_blank, log_prob_y)
    return log_prob_prefix[-1, -1]


def _log_prob_alignments_fwd(log_prob_blank: Tensor, log_prob_y: Tensor):
    """The forward part of custom_vjp of log_prob_alignments.

    Please see log_prob_alignments for the descriptions of log_prob_blank and log_prob_y.

    Returns:
        (log_prob_alignments, res), where `log_prob_alignments` represents the log(prob(y|x)) and
        `res` is a tuple used by the backward function `_log_prob_alignments_bwd`.
    """
    log_prob_prefix = log_prob_prefix_alignments(log_prob_blank, log_prob_y)
    return (log_prob_prefix[-1, -1], (log_prob_blank, log_prob_y, log_prob_prefix))


def _log_prob_alignments_bwd(
    res: tuple[Tensor, Tensor, Tensor], g: Tensor
) -> tuple[Tensor, Tensor]:
    """The backward part of custom_vjp of log_prob_alignments.

    Args:
        res: The intermediate data produced by _log_prob_alignments_fwd.
        g: The gradient on log_prob_alignments.

    Returns:
        (grad_blank, grad_y), where
        - grad_blank has the same shape as log_prob_blank and represents
          g * d(log(prob(y|x))) / d(log_prob_blank);
        - grad_y has the same shape as log_prob_y and represents
          g * d(log(prob(y|x))) / d(log_prob_y);
    """
    log_prob_blank, log_prob_y, log_prob_prefix = res
    log_prob_suffix = log_prob_suffix_alignments(log_prob_blank, log_prob_y)
    grad_blank, grad_y = log_prob_gradients(
        log_prob_blank=log_prob_blank,
        log_prob_y=log_prob_y,
        log_prob_prefix=log_prob_prefix,
        log_prob_suffix=log_prob_suffix,
    )
    return (g * grad_blank, g * grad_y)


# Jacobian vector products for log_prob_alignments.
#
# Reference:
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
log_prob_alignments.defvjp(_log_prob_alignments_fwd, _log_prob_alignments_bwd)


def apply_paddings(
    log_prob_blank: Tensor, log_prob_y: Tensor, am_paddings: Tensor, lm_paddings: Tensor
) -> dict[str, Tensor]:
    """Applies paddings to log_prob_{blank, y}.

    Since sequences in a batch may have different lengths, we pad them to fixed sizes, e.g.,
    from T=3 and U=2 to am_max_seq_length=6 and lm_max_seq_length=5:

        BOS   A    B   PAD  PAD
    F0   @--->.--->.    .    .
         |    |    |
         v    v    v
    F1   .--->.--->.    .    .
         |    |    |
         v    v    v
    F2   .--->.--->.    .    .
                   |
                   v
    FE   .    .    $    .    .
                   #
                   v
    PAD  .    .    .    .    .
                   #
                   v
    PAD  .    .    .###>.###>.

    The missing edges means that we set the corresponding log_prob to -inf, while '#' means that we
    set the log_prob to 0.

    Args:
        log_prob_blank: an array of shape [am_max_seq_len, lm_max_seq_len], where only the
            upper-left (T, U + 1) entries will be used.
        log_prob_y: an array of shape [am_max_seq_len, lm_max_seq_len], where only the
            upper-left (T + 1, U) entries will be used.
        am_paddings: a 0/1 array of shape [am_max_seq_len], where the first T entries are 0,
            the rest are 1.
        lm_paddings: a 0/1 array of shape [lm_max_seq_len], where the first U + 1 entries are 0,
            the rest are 1.

    Returns:
        A dict containing "log_prob_blank" and "log_prob_y" of shape
          [am_max_seq_len, lm_max_seq_len] and [am_max_seq_len + 1, lm_max_seq_len - 1],
        respectively and
          log_prob_alignments(log_prob_blank, log_prob_y) ==
          log_prob_alignments(log_prob_blank[:T, :U + 1], log_prob_y[:T + 1, :U])

    Raises:
        ValueError: if lm_paddings are all 1s and called with checkify.
    """
    checkify.check(jnp.sum(1 - lm_paddings) > 0, "lm_paddings cannot be all 1s.")
    am_max_seq_len = am_paddings.shape[0]
    lm_max_seq_len = lm_paddings.shape[0]
    lm_eos_index = ((1 - lm_paddings).sum() - 1).astype(jnp.int32)  # U
    am_eos_index = ((1 - am_paddings).sum()).astype(jnp.int32)  # T
    # log_prob_blank_mask[i, j] = 1 iff (i < T - 1 and j <= U) or (i == T - 1 and j == U),
    # 0 otherwise.
    log_prob_blank_mask = jnp.expand_dims(
        jnp.arange(am_max_seq_len) < am_eos_index - 1, axis=-1
    ) * jnp.expand_dims(jnp.arange(lm_max_seq_len) <= lm_eos_index, axis=0)
    # The terminal state reached by a blank step.
    log_prob_blank_mask = log_prob_blank_mask.at[am_eos_index - 1, lm_eos_index].set(1)
    # Set to NEG_INF if mask==0.
    log_prob_blank = log_prob_blank * log_prob_blank_mask + _NEG_INF * (1 - log_prob_blank_mask)

    # 1 where we want to set log_prob_blank to 0, 0 otherwise.
    log_prob_blank_zero = jnp.expand_dims(
        jnp.arange(am_max_seq_len) >= am_eos_index, axis=-1
    ) * jnp.expand_dims(jnp.arange(lm_max_seq_len) == lm_eos_index, axis=0)
    # log_prob_blank[T:, U] = 0
    log_prob_blank *= 1 - log_prob_blank_zero

    # Pad one FE row for log_prob_y to handle examples of length am_max_seq_len.
    # [am_max_seq_len + 1, lm_max_seq_len].
    log_prob_y = jnp.pad(log_prob_y, ((0, 1), (0, 0)), constant_values=_NEG_INF)
    # log_prob_y_mask[i, j] = 1 iff (i < T and j < U), 0 otherwise.
    log_prob_y_mask = jnp.expand_dims(
        jnp.arange(am_max_seq_len + 1) < am_eos_index, axis=-1
    ) * jnp.expand_dims(jnp.arange(lm_max_seq_len) < lm_eos_index, axis=0)
    # Set to NEG_INF if mask==0.
    log_prob_y = log_prob_y * log_prob_y_mask + _NEG_INF * (1 - log_prob_y_mask)
    # 1 where we want to set log_prob_y to 0, 0 otherwise.
    log_prob_y_zero = jnp.expand_dims(
        jnp.arange(am_max_seq_len + 1) == am_max_seq_len, axis=-1
    ) * jnp.expand_dims(jnp.arange(lm_max_seq_len) >= lm_eos_index, axis=0)
    # log_prob_y[-1, U:] = 0
    log_prob_y *= 1 - log_prob_y_zero
    return dict(log_prob_blank=log_prob_blank, log_prob_y=log_prob_y[:, :-1])
