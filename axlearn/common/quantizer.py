# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# google/praxis:
# Copyright 2022 Google LLC.
# Licensed under the Apache License, Version 2.0 (the "License").

# facebookresearch/fairseq:
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

"""Vector quantization layer and metrics.

Reference:
vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations.
https://arxiv.org/pdf/1910.05453.pdf.

Self-supervised Learning with Random-projection Quantizer for Speech Recognition.
https://arxiv.org/pdf/2202.01855.pdf.

Neural Discrete Representation Learning.
https://arxiv.org/pdf/1711.00937.pdf.

https://github.com/google/praxis/blob/179774fb688aa8fe048307d2184c9f2b338e935f/praxis/layers/quantizer.py
https://github.com/google/praxis/blob/179774fb688aa8fe048307d2184c9f2b338e935f/praxis/layers/quantizer_objectives.py
https://github.com/facebookresearch/fairseq/blob/d871f6169f8185837d1c11fb28da56abfd83841c/fairseq/modules/gumbel_vector_quantizer.py
"""
from enum import Enum, unique
from typing import NamedTuple, Optional

import jax.nn
import jax.numpy as jnp

from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Linear, MultiLinear
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import InvocationContext, Module, current_context
from axlearn.common.normalize import l2_normalize
from axlearn.common.param_init import (
    DefaultInitializer,
    FanAxes,
    GaussianInitializer,
    WeightInitializer,
    constant_initializer,
)
from axlearn.common.schedule import as_schedule_fn
from axlearn.common.utils import Nested, NestedTensor, Tensor

_einsum_dims = "abcdefwxyz"


def compute_code_histogram(onehots: Tensor, paddings: Tensor) -> Tensor:
    """Computes histograms of the quantized codes over the codebook vocabulary.

    Args:
        onehots: Quantized onehots. Tensor of shape [..., num_codebooks, codebook_size].
        paddings: paddings of the quantized codes. Tensor of shape [...].

    Returns:
        Histogram of the quantized codes of shape [num_codebooks, codebook_size].
    """
    onehots = onehots * (1 - paddings)[..., None, None]
    # [num_codebooks, codebook_size].
    histogram = jnp.sum(onehots, axis=tuple(range(onehots.ndim - 2)))
    return histogram


def compute_code_pplx(onehots: Tensor, paddings: Tensor) -> tuple[Tensor, Tensor]:
    """Computes pplx and entropy of the quantized codes distribution."""
    histogram = compute_code_histogram(onehots, paddings)
    normalizer = jnp.sum(1 - paddings)
    # [num_codebooks, codebook_size].
    probs = histogram / jnp.maximum(normalizer, 1.0)
    log_probs = jnp.log(jnp.maximum(1.0e-30, probs))
    # [num_codebooks].
    sum_plogp = jnp.sum(log_probs * probs, axis=-1)
    pplx = jnp.mean(jnp.exp(-sum_plogp))
    entropy = jnp.log(pplx)
    return pplx, entropy


def compute_code_coverage(onehots: Tensor, paddings: Tensor) -> Tensor:
    """Computes codebook coverage."""
    codebook_size = onehots.shape[-1]
    # [num_codebooks, codebook_size].
    histogram = compute_code_histogram(onehots, paddings)
    avg_num_covered_words = jnp.mean(jnp.sum((histogram > 0).astype(jnp.float32), axis=-1))
    return avg_num_covered_words / codebook_size


class BaseQuantizer(BaseLayer):
    """An abstract class to define the common interface of vector quantizer layers."""

    @config_class
    class Config(BaseLayer.Config):
        # Dim of each codebook.
        codebook_dim: Required[int] = REQUIRED
        # Vocabulary size of each codebook.
        codebook_size: Required[int] = REQUIRED
        # Number of codebook groups.
        num_codebooks: Required[int] = REQUIRED

    class Output(NamedTuple):
        # [..., num_codebooks].
        ids: Tensor
        # [..., num_codebooks, codebook_dim].
        quantized_vectors: Tensor
        # Scalar of quantizer loss.
        loss: Optional[Tensor] = None

    @classmethod
    def default_config(cls):
        cfg: BaseQuantizer.Config = super().default_config()
        # Do not shard the codebook.
        # ToDo(zhiyunlu): investigate the codebook partition_spec.
        cfg.param_partition_spec = (None, None, None)
        cfg.param_init = REQUIRED
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            codebook=ParameterSpec(
                shape=(cfg.codebook_size, cfg.num_codebooks, cfg.codebook_dim),
            )
        )
        return params

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> Output:
        """Quantizes input sequences.

        Args:
            inputs: Tensor of shape [batch_size, seq_len, input_dim].
            paddings: Tensor of shape [batch_size, seq_len].

        Returns:
            BaseQuantizer.Output.
            * ids: Tensor [..., num_codebooks].
            * quantized_vectors: Tensor [..., num_codebooks, codebook_dim].
        """
        raise NotImplementedError(type(self))

    def lookup(self, ids: Tensor) -> Output:
        """Codebook look up with ids.

        Args:
            ids: integer tensor of shape [..., num_codebooks] with values
                in range [0, codebook_size).

        Returns:
            BaseQuantizer.Output
            * ids: Tensor [..., num_codebooks].
            * quantized_vectors: Tensor [..., num_codebooks, codebook_dim].

        Raises:
            NotImplementedError: if ids.ndim > 11.
        """
        return _lookup(ids=ids, codebook=self.parameters["codebook"])


def _lookup(*, ids: Tensor, codebook: Tensor) -> BaseQuantizer.Output:
    """Codebook look up with ids.

    Args:
        ids: integer tensor of shape [..., num_codebooks] with values
            in range [0, codebook_size).
        codebook: Tensor of shape [codebook_size, num_codebooks, codebook_dim].

    Returns:
        BaseQuantizer.Output
        * ids: Tensor [..., num_codebooks].
        * quantized_vectors: Tensor [..., num_codebooks, codebook_dim].

    Raises:
        NotImplementedError: if ids.ndim > 11.
    """
    if ids.ndim - 1 > len(_einsum_dims):
        raise NotImplementedError(ids.shape)

    # [..., num_codebooks]
    g_index = jnp.expand_dims(jnp.arange(ids.shape[-1]), axis=tuple(range(ids.ndim - 1)))
    # codebook: [codebook_size, num_codebooks, codebook_dim], ids: [..., num_codebooks]
    # -> [..., num_codebooks, codebook_dim]
    quantized_vectors = codebook[ids, g_index]
    return BaseQuantizer.Output(
        ids=ids,
        quantized_vectors=quantized_vectors,
    )


@unique
class SimilarityMetric(Enum):
    L2_DISTANCE = 0
    DOT_PRODUCT = 1


def quantize_by_nearest_neighbor(
    inputs: Tensor, *, codebook: Tensor, metric: SimilarityMetric
) -> BaseQuantizer.Output:
    """Quantizes inputs by the nearest neighbor look-up in the codebook.

    This is used in both RandomVectorQuantizer and KmeansVectorQuantizer.

    Args:
        inputs: Tensor of shape [..., num_codebooks, codebook_dim].
        codebook: Tensor of shape [codebook_size, num_codebooks, codebook_dim].
        metric: similarity metric to rank the codebook. Choose from
            L2_DISTANCE or DOT_PRODUCT.

    Returns:
        BaseQuantizer.Output.

    Raises:
        ValueError: if last two dimensions of inputs and codebook do not match.
        ValueError: if metric is neither L2_DISTANCE nor DOT_PRODUCT.
        NotImplementedError: if inputs.ndim > 12.
    """
    if codebook.shape[-2:] != inputs.shape[-2:]:
        raise ValueError(
            "Last two dimensions of inputs should match with [num_codebooks, codebook_dim]."
            f"{inputs.shape[-2:]} != {codebook.shape[-2:]}."
        )

    # Compute similarity between inputs and codebook.
    if inputs.ndim - 2 > len(_einsum_dims):
        raise NotImplementedError(inputs.shape)
    batch_dims = _einsum_dims[: inputs.ndim - 2]
    distance = -2 * jnp.einsum(f"{batch_dims}gh,vgh->{batch_dims}vg", inputs, codebook)
    # l2_dist = (inputs - codebook) ** 2 = inputs ** 2 - 2 * input * codebook + codebook ** 2.
    # Since we do not compare distances across input vectors, we can drop the `input ** 2` term.
    # [..., vocab_size, num_codebooks].
    if metric == SimilarityMetric.L2_DISTANCE:
        distance += jnp.sum(codebook**2, axis=-1)
    elif metric != SimilarityMetric.DOT_PRODUCT:
        raise ValueError(f"Expect DOT_PRODUCT metric, but got {metric}.")
    # [..., num_codebooks].
    ids = jnp.argmin(distance, axis=-2)
    # Note if the codebook is normalized, quantized_vectors is also normalized.
    return _lookup(ids=ids, codebook=codebook)


def _apply_paddings(*, outputs: BaseQuantizer.Output, paddings: Tensor) -> BaseQuantizer.Output:
    """Applies paddings to quantizer outputs.

    ids are padded with -1. onehots and quantized_vectors are padded with 0s. loss
    is copied over.

    Args:
        outputs: BaseQuantizer.Output.
        paddings: 0/1 tensor of shape [batch_size, seq_len], where 0 is valid position.

    Returns:
        padded_outputs: BaseQuantizer.Output.
    """

    # ids are padded with -1.
    ids_paddings = paddings[:, :, None].astype(outputs.ids.dtype)
    ids = outputs.ids * (1 - ids_paddings) + (-1) * ids_paddings
    quantized_vectors = outputs.quantized_vectors * (1 - paddings)[:, :, None, None]
    return BaseQuantizer.Output(
        ids=ids,
        quantized_vectors=quantized_vectors,
        loss=outputs.loss,
    )


def _ids_to_onehots(ids: Tensor, *, codebook_size: int, dtype: jnp.dtype) -> Tensor:
    # [..., num_codebooks, codebook_size].
    return jax.nn.one_hot(ids, num_classes=codebook_size, axis=-1, dtype=dtype)


def _add_codebook_summaries(*, context: InvocationContext, onehots: Tensor, paddings: Tensor):
    """Helper function to compute codebook distribution statistics and add to summaries.

    The statistics are from all frames, not only on those masked frames in self-supervised training.
    # ToDo(zhiyunlu): Add support to take an additional mask input.

    Args:
        context: Module invocation context to add summaries to.
        onehots: onehot of BaseQuantizer.Output.ids.
        paddings: 0/1 tensor of shape [batch_size, seq_len], where 0 is valid position.
    """
    coverage = compute_code_coverage(onehots=onehots, paddings=paddings)
    pplx, entropy = compute_code_pplx(onehots=onehots, paddings=paddings)
    batch_size = paddings.shape[0]

    num_frames = jnp.sum(1 - paddings)
    context.add_summary(
        "codebook/num_frames",
        WeightedScalar(num_frames.astype(jnp.float32) / batch_size, batch_size),
    )
    # Mean coverage of all codebooks.
    context.add_summary(
        "codebook/coverage",
        WeightedScalar(coverage, jnp.maximum(1, num_frames)),
    )
    # Mean perplexity of all codebooks.
    context.add_summary(
        "codebook/pplx",
        WeightedScalar(pplx, jnp.maximum(1, num_frames)),
    )
    # Mean entropy of all codebooks.
    context.add_summary(
        "codebook/entropy",
        WeightedScalar(entropy, jnp.maximum(1, num_frames)),
    )


class RandomVectorQuantizer(BaseQuantizer):
    """Random-projection Quantizer.

    Best-RQ: Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition.
    https://arxiv.org/pdf/2202.01855.pdf.

    This layer performs random projection, and vector quantization with a
    random codebook. We do not back-propagate any error signal through this layer. The gradients
    w.r.t. the layer params (codebook, random_proj) and inputs are all zeros.
    """

    @config_class
    class Config(BaseQuantizer.Config):
        input_dim: Required[int] = REQUIRED
        rand_proj: Linear.Config = Linear.default_config().set(bias=False)
        normalize_codebook: bool = True
        normalize_inputs: bool = True

    @classmethod
    def default_config(cls):
        cfg: RandomVectorQuantizer.Config = super().default_config()
        # Sect 3.1 https://arxiv.org/pdf/2202.01855.pdf.
        # Codebook uses standard Gaussian initialization.
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={".*codebook$": GaussianInitializer.default_config().set(std=1.0)}
        )
        return cfg

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Random projection uses Xavier initialization.
        self._add_child(
            "rand_proj",
            cfg.rand_proj.set(
                input_dim=cfg.input_dim, output_dim=cfg.num_codebooks * cfg.codebook_dim
            ),
        )

    def initialize_parameters_recursively(
        self, prng_key: Tensor, *, prebuilt: Optional[Nested[Optional[ParameterSpec]]] = None
    ) -> NestedTensor:
        params = super().initialize_parameters_recursively(prng_key=prng_key, prebuilt=prebuilt)
        # In RandomVectorQuantizer, we freeze codebook throughout training. So we can
        # normalize codebook only once at the initialization.
        if self.config.normalize_codebook:
            params["codebook"] = l2_normalize(params["codebook"], axis=-1, eps=1e-12)
        return params

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> BaseQuantizer.Output:
        """Computes random projection and quantization.

        Args:
            inputs: Tensor of shape [batch_size, seq_len, input_dim].
            paddings: 0/1 Tensor of shape [batch_size, seq_len].

        Returns:
            BaseQuantizer.Output.
        """
        cfg = self.config

        # [batch_size, seq_len, num_codebooks * codebook_dim].
        inputs = self.rand_proj(inputs)
        inputs_by_group = jnp.reshape(
            inputs, list(inputs.shape[:2]) + [cfg.num_codebooks, cfg.codebook_dim]
        )

        if cfg.normalize_inputs:
            # [..., num_codebooks, codebook_dim].
            inputs_by_group = l2_normalize(inputs_by_group, axis=-1, eps=1e-12)

        # When codebook is normalized, dot_product is equivalent to l2_distance.
        metric = (
            SimilarityMetric.DOT_PRODUCT if cfg.normalize_codebook else SimilarityMetric.L2_DISTANCE
        )
        q_outputs = quantize_by_nearest_neighbor(
            inputs=inputs_by_group,
            codebook=self.parameters["codebook"],
            metric=metric,
        )
        q_outputs = _apply_paddings(outputs=q_outputs, paddings=paddings)
        # Best-rq freezes the codebook.
        ids = jax.lax.stop_gradient(q_outputs.ids)
        quantized_vectors = jax.lax.stop_gradient(q_outputs.quantized_vectors)

        outputs = self.Output(
            # [batch_size, seq_len, num_codebooks].
            ids=ids,
            # [batch_size, seq_len, num_codebooks, codebook_dim].
            quantized_vectors=quantized_vectors,
        )

        onehots = _ids_to_onehots(outputs.ids, codebook_size=cfg.codebook_size, dtype=jnp.int32)
        _add_codebook_summaries(context=current_context(), onehots=onehots, paddings=paddings)
        return outputs


class KmeansVectorQuantizer(BaseQuantizer):
    """Vector quantizer with mse loss.

    The code is selected based on the l2 distance between inputs and code embeddings.
    The codebook is learnt with l2 loss. Gradients w.r.t inputs is computed using the
    straight-through estimator.

    VQ-VAE: Neural Discrete Representation Learning.
    https://arxiv.org/pdf/1711.00937.pdf.
    vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations.
    https://arxiv.org/pdf/1910.05453.pdf.
    https://github.com/google/praxis/blob/179774fb688aa8fe048307d2184c9f2b338e935f/praxis/layers/quantizer.py#L342
    Current implementation does not apply inputs or codebook normalization.
    """

    @config_class
    class Config(BaseQuantizer.Config):
        # Scale of the commitment loss.
        beta: Required[float] = REQUIRED
        normalize_codebook: bool = False
        normalize_inputs: bool = False

    @classmethod
    def default_config(cls) -> Config:
        cfg: KmeansVectorQuantizer.Config = super().default_config()
        # The original VQ-VAE implementation initializes codebook by uniform(1/sqrt(dim)).
        # https://github.com/google-deepmind/sonnet/blob/cd5b5fa48e15e4d020f744968f5209949ebe750f/sonnet/python/modules/nets/vqvae.py#L62C24-L64
        # Note: fan_out of the codebook is `num_codebooks * codebook_dim`.
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                ".*codebook$": WeightInitializer.default_config().set(
                    scale=1.0, fan="fan_out", distribution="uniform"
                )
            }
        )
        return cfg

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "codebook":
            if len(parameter_spec.shape) != 3:
                raise ValueError(f"Unexpected parameter spec {parameter_spec}")
            # Axis 0, 1, 2 is codebook vocab_size, num_codebooks, and codebook_dim respectively.
            return FanAxes(in_axis=0, out_axis=(1, 2))
        else:
            return None

    def forward(self, inputs: Tensor, *, paddings: Tensor) -> BaseQuantizer.Output:
        """Quantization with mse loss.

        Gradients for the codebook are computed only from MSE loss, while the gradients for
        the inputs are computed from both MSE and decoder losses approximated using
        the straight-through estimator.

        Args:
            inputs: Tensor of shape [batch_size, seq_len, input_dim]. input_dim must equal to
                cfg.num_codebooks * cfg.codebook_dim.
            paddings: 0/1 Tensor of shape [batch_size, seq_len].

        Returns:
            BaseQuantizer.Output.
            module_outputs contains kmeans_loss and commitment_loss.

        Raises:
            ValueError: if inputs' last dimension does not match with the codebook dimensions.
        """
        cfg = self.config
        input_dim = cfg.num_codebooks * cfg.codebook_dim
        batch_size, seq_len = inputs.shape[:2]
        if inputs.shape[-1] != input_dim:
            raise ValueError(
                "inputs feature dimension should match with dims from all codebooks."
                f"{inputs.shape[-1]} != {cfg.num_codebooks} x {cfg.codebook_dim}."
            )
        inputs_by_group = jnp.reshape(
            inputs, [batch_size, seq_len, cfg.num_codebooks, cfg.codebook_dim]
        )
        quantized_inputs = quantize_by_nearest_neighbor(
            inputs=inputs_by_group,
            codebook=self.parameters["codebook"],
            metric=(
                SimilarityMetric.DOT_PRODUCT
                if cfg.normalize_codebook
                else SimilarityMetric.L2_DISTANCE
            ),
        )
        if cfg.normalize_inputs:
            inputs_by_group = l2_normalize(inputs_by_group, axis=-1, eps=1e-12)
        if cfg.normalize_codebook:
            self.parameters["codebook"] = l2_normalize(
                self.parameters["codebook"], axis=-1, eps=1e-12
            )
        quantized_inputs = _apply_paddings(outputs=quantized_inputs, paddings=paddings)

        # [batch_size, seq_len, input_dim].
        q_vecs = jnp.reshape(quantized_inputs.quantized_vectors, [batch_size, seq_len, input_dim])

        # Compute mean squared errors between q_vecs and inputs on non-padded frames.
        # Number of valid frames * input_dim.
        num_frames = jnp.sum(1 - paddings)
        denominator = jnp.maximum(num_frames * input_dim, 1)
        # Eq.3 of VQ-VAE paper https://arxiv.org/pdf/1711.00937.pdf.
        # The codebook is optimized by kmeans_loss only.
        inputs_to_loss = (
            jnp.reshape(inputs_by_group, [batch_size, seq_len, -1])
            if cfg.normalize_inputs
            else inputs
        )
        kmeans_loss = (
            jnp.sum(
                (q_vecs - jax.lax.stop_gradient(inputs_to_loss)) ** 2 * (1 - paddings)[:, :, None]
            )
            / denominator
        )
        # The inputs receive gradients from commitment_loss.
        commitment_loss = (
            jnp.sum(
                (inputs_to_loss - jax.lax.stop_gradient(q_vecs)) ** 2 * (1 - paddings)[:, :, None]
            )
            / denominator
        )
        total_loss = kmeans_loss + cfg.beta * commitment_loss

        self.add_module_output("kmeans_loss", kmeans_loss)
        self.add_module_output("commitment_loss", commitment_loss)

        # Straight-through estimator such that dL/inputs = dL/q_outputs.
        # Note that gradient on quantized_vectors is not propagated to the codebook.
        quantized_vectors = inputs + jax.lax.stop_gradient(q_vecs - inputs)
        # We need this to stop gradients on the padded inputs.
        quantized_vectors = quantized_vectors * (1 - paddings)[:, :, None]

        outputs = self.Output(
            # [batch_size, seq_len, num_codebooks].
            ids=quantized_inputs.ids,
            # [batch_size, seq_len, num_codebooks, codebook_dim].
            quantized_vectors=jnp.reshape(
                quantized_vectors, [batch_size, seq_len, cfg.num_codebooks, cfg.codebook_dim]
            ),
            loss=total_loss,
        )
        onehots = _ids_to_onehots(outputs.ids, codebook_size=cfg.codebook_size, dtype=jnp.int32)
        _add_codebook_summaries(context=current_context(), onehots=onehots, paddings=paddings)
        return outputs


class GumbelSoftmaxVectorQuantizer(BaseQuantizer):
    """Vector quantizer with Gumbel softmax trick.

    The code is selected based on the largest index of inputs. Gradients w.r.t inputs is computed
    using Gumbel softmax straight-through estimator.

    Categorical Reparameterization with Gumbel-Softmax. https://arxiv.org/pdf/1611.01144.pdf.
    https://github.com/facebookresearch/fairseq/blob/d871f6169f8185837d1c11fb28da56abfd83841c/fairseq/modules/gumbel_vector_quantizer.py.
    """

    @config_class
    class Config(BaseQuantizer.Config):
        input_dim: Required[int] = REQUIRED
        input_proj: MultiLinear.Config = MultiLinear.default_config()
        temperature_schedule: Required[InstantiableConfig] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "input_proj",
            cfg.input_proj.set(
                input_dim=cfg.input_dim,
                num_outputs=cfg.num_codebooks,
                output_dim=cfg.codebook_size,
            ),
        )
        self.temperature_schedule = as_schedule_fn(cfg.temperature_schedule)

    @classmethod
    def default_config(cls) -> Config:
        cfg: GumbelSoftmaxVectorQuantizer.Config = super().default_config()
        cfg.param_init = DefaultInitializer.default_config().set(
            init_by_param_name={
                ".*codebook$": WeightInitializer.default_config().set(
                    scale=1.0, fan=None, distribution="uniform"
                )
            }
        )
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        params = super()._create_layer_parameter_specs()
        params["step"] = ParameterSpec(
            shape=[],
            dtype=jnp.float32,
            mesh_axes=(None,),
            initializer=constant_initializer(0.0),
            weight_decay_scale=0,
        )
        return params

    def forward(
        self, inputs: Tensor, *, paddings: Tensor
    ) -> tuple[BaseQuantizer.Output, dict[str, Tensor]]:
        """Quantization using Gumbel softmax trick.

        The code is selected based on the largest index of inputs. Inputs Gradients is computed
        with Gumbel softmax straight-through estimator.

        Args:
            inputs: tensor of shape [batch_size, seq_len, input_dim].
            paddings: 0/1 Tensor of shape [batch_size, seq_len].

        Returns:
            BaseQuantizer.Output.
            module_outputs contains temperature `tau` and prediction probability `probs`.
        """
        cfg = self.config
        # [batch_size, seq_len, num_codebooks, vocab_size].
        logits = self.input_proj(inputs=inputs)

        if self.is_training:
            tau = self.temperature_schedule(self.parameters["step"])
            logits = (
                logits + jax.random.gumbel(self.prng_key, shape=logits.shape, dtype=logits.dtype)
            ) / tau
            self.add_state_update("step", self.parameters["step"] + 1)

        # [batch_size, seq_len, num_codebooks].
        ids = jnp.argmax(logits, axis=-1)

        if not self.is_training:
            outputs = self.lookup(ids=ids)
            outputs = _apply_paddings(outputs=outputs, paddings=paddings)
        else:
            # [batch_size, seq_len, 1].
            mask = (1 - paddings)[:, :, None].astype(ids.dtype)
            ids = ids * mask + (-1) * (1 - mask)
            # TODO(dhwang2): optimize memory by scan for long context training.
            # [batch_size, seq_len, num_codebooks, vocab_size].
            onehots = _ids_to_onehots(ids, codebook_size=cfg.codebook_size, dtype=inputs.dtype)
            # We need this to stop gradients on the padded frames.
            mask = mask.astype(inputs.dtype)
            onehots = onehots * mask[:, :, :, None]
            # [batch_size, seq_len, num_codebooks, vocab_size].
            y_soft = jax.nn.softmax(logits, axis=-1)
            y_soft = y_soft * mask[:, :, :, None]

            # Straight-through estimator such that dL/y_soft = dL/onehots.
            onehots = y_soft + jax.lax.stop_gradient(onehots - y_soft)
            batch_dims = _einsum_dims[: onehots.ndim - 2]
            quantized_vectors = jnp.einsum(
                f"{batch_dims}gv,vgh->{batch_dims}gh", onehots, self.parameters["codebook"]
            )
            quantized_vectors = quantized_vectors * mask[:, :, :, None]
            outputs = self.Output(
                # [batch_size, seq_len, num_codebooks].
                ids=ids,
                # [batch_size, seq_len, num_codebooks, codebook_dim].
                quantized_vectors=quantized_vectors,
            )

        onehots = _ids_to_onehots(outputs.ids, codebook_size=cfg.codebook_size, dtype=jnp.int32)
        _add_codebook_summaries(context=current_context(), onehots=onehots, paddings=paddings)
        if self.is_training:
            self.add_module_output("probs", y_soft)
            self.add_summary("codebook/temperature_schedule_step", self.parameters["step"])
            self.add_summary("codebook/temperature", tau)
        return outputs
