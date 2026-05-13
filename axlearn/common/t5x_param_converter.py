# Copyright © 2023 Apple Inc.

"""T5X → AXLearn parameter converters.

Imports T5X (Flax/JAX) parameter dictionaries into AXLearn T5 model state. No
PyTorch dependency. Originally lived in `axlearn/common/param_converter.py`,
extracted when that module was deleted as part of axlearn #2469 (PyTorch test
removal).
"""

import jax
import jax.numpy as jnp

from axlearn.common.attention import (
    BaseMultiheadLinear,
    FusedQKVLinear,
    QKVLinear,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerAttentionLayer,
    TransformerLayer,
)
from axlearn.common.t5 import T5Decoder, T5Encoder, T5EncoderDecoderModel
from axlearn.common.utils import NestedTensor, VDict, as_tensor


def _parameters_from_t5x_ff(src: NestedTensor, *, src_norm: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X Mlp param dict into AXLearn TransformerFeedForwardLayer.

    Corresponding T5X layer is layer.MlpBlock.

    Args:
        src: Params corresponding to the T5X Feed Forward param dict.
        src_norm: Params corresponding to T5X pre-norm for feed forward.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Feed Forward.
    """
    dst = dict(dropout1={}, dropout2={}, stochastic_depth={})

    for layer, params in src.items():
        if layer == "wi_0":
            dst["linear1_0"] = _parameters_from_t5x_linear_like(params)
        elif layer == "wi_1":
            dst["linear1_1"] = _parameters_from_t5x_linear_like(params)
        elif layer == "wo":
            dst["linear2"] = _parameters_from_t5x_linear_like(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")
    dst["norm"] = _parameters_from_t5x_layer_norm(src_norm)

    return as_tensor(dst)


def _parameters_from_t5x_attn_proj(
    src: NestedTensor, *, dst_proj: str, dst_layer: BaseMultiheadLinear
) -> NestedTensor:
    """Imports params from T5X DenseGeneral param dict into AXLearn BaseMultiheadLinear model
    state.

    Corresponding T5X layer is just layer.DenseGeneral.

    Args:
        src: Params corresponding to the T5X Attention Projection param dict.
        dst_proj: AXLearn linear projection to transfer params to. Should be one of {q,k,v,o}_proj.
        dst_layer: AXLearn BaseMultiheadLinear model.
            This is usually MultiheadInputLinear or MultiheadOutputLinear.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If projection name is invalid.
    """
    if dst_proj not in [f"{x}_proj" for x in "qkvo"]:
        raise ValueError(f"Unsupported projection type. Got {dst_proj}")

    num_heads = dst_layer.config.num_heads
    per_head_dim = dst_layer.config.per_head_dim

    kernel = src["kernel"]
    if dst_proj[0] == "o":
        kernel = kernel.transpose(1, 0)
    # T5X bakes the projection scaling into initialization
    if dst_proj[0] == "q":
        kernel = kernel * per_head_dim**0.5

    # Note: T5X linear layers do not use bias.
    dst = dict(weight=kernel.reshape(-1, num_heads, per_head_dim))
    return as_tensor(dst)


def _parameters_from_t5x_lm_head(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X LM Head param dict into AXLearn LmHead model state.

    Args:
        src: Params corresponding to the T5X LM Head param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.
    """
    dst = dict(weight=src["kernel"].transpose(1, 0))
    return as_tensor(dst)


def _parameters_from_t5x_rel_pos_emb(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X Relative Position Embeddings param dict
        into AXLearn T5RelativePositionalEmbedding model state.

    Corresponding T5X module is layers.RelativePositionBiases

    Args:
        src: Params corresponding to the T5X Rel Pos Emb param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.
    """
    dst = dict(emb=dict(weight=src["rel_embedding"].transpose(1, 0)))
    return as_tensor(dst)


def _parameters_from_t5x_attention(
    src: NestedTensor, *, src_norm: NestedTensor, dst_layer: TransformerAttentionLayer
) -> NestedTensor:
    """Imports params from T5X Attention param dict into AXLearn TransformerAttentionLayer model
    state.

    Corresponding T5X layer is layer.MultiHeadDotProductAttention.

    Args:
        src: Params corresponding to the T5X Attention param dict.
        src_norm: Params corresponding to T5X pre-norm for attention.
        dst_layer: AXLearn TransformerAttentionLayer model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Attention.
        NotImplementedError: If dst_layer.attention.i_proj is not supported.
    """
    dst = dict(dropout={}, stochastic_depth={}, attention=dict(i_proj={}, dropout={}))
    i_proj = {}

    for layer, params in src.items():
        if layer not in {"query", "key", "value", "out"}:
            raise ValueError(f"Unsupported layer: {layer}")

        # query, key, value, out => q_proj, k_proj, v_proj, o_proj
        proj_key = f"{layer[0]}_proj"

        if proj_key.startswith("o"):
            dst["attention"][proj_key] = _parameters_from_t5x_attn_proj(
                params, dst_proj=proj_key, dst_layer=dst_layer.attention.o_proj
            )
            continue

        if isinstance(dst_layer.attention.i_proj, QKVLinear):
            proj_layer = getattr(dst_layer.attention.i_proj, proj_key)
        elif isinstance(dst_layer.attention.i_proj, FusedQKVLinear):
            proj_layer = dst_layer.attention.i_proj.qkv_proj
        else:
            raise NotImplementedError(type(dst_layer.attention.i_proj))
        i_proj[proj_key] = _parameters_from_t5x_attn_proj(
            params, dst_proj=proj_key, dst_layer=proj_layer
        )

    if isinstance(dst_layer.attention.i_proj, QKVLinear):
        for proj_key in ("q_proj", "k_proj", "v_proj"):
            dst["attention"]["i_proj"][proj_key] = i_proj[proj_key]
    elif isinstance(dst_layer.attention.i_proj, FusedQKVLinear):
        dst["attention"]["i_proj"] = VDict(
            qkv_proj=dict(
                jax.tree.map(
                    lambda *inputs: jnp.stack(inputs),
                    *[i_proj[proj_key] for proj_key in ("q_proj", "k_proj", "v_proj")],
                )
            ),
        )
    else:
        raise NotImplementedError(type(dst_layer.attention.i_proj))

    dst["norm"] = _parameters_from_t5x_layer_norm(src_norm)
    return as_tensor(dst)


def _parameters_from_t5x_transformer_layer(
    src: NestedTensor, dst_layer: TransformerLayer
) -> NestedTensor:
    """Imports parameters from T5X Transformer param dict into AXLearn TransformerLayer model state.

    Corresponding T5X layer is network.EncoderLayer and network.DecoderLayer.

    Args:
        src: Params corresponding to the T5X Transformer param dict.
        dst_layer: AXLearn TransformerLayer model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Transformer.
    """
    dst = {}

    for layer_name in dst_layer.children:
        if layer_name == "cross_attention":
            dst["cross_attention"] = _parameters_from_t5x_attention(
                src["encoder_decoder_attention"],
                src_norm=src["pre_cross_attention_layer_norm"],
                dst_layer=dst_layer.cross_attention,
            )
        elif layer_name == "self_attention":
            dst["self_attention"] = _parameters_from_t5x_attention(
                src.get("self_attention") or src.get("attention"),
                src_norm=src.get("pre_self_attention_layer_norm")
                or src.get("pre_attention_layer_norm"),
                dst_layer=dst_layer.self_attention,
            )
        elif layer_name == "feed_forward":
            dst["feed_forward"] = _parameters_from_t5x_ff(
                src["mlp"], src_norm=src["pre_mlp_layer_norm"]
            )
        else:
            raise ValueError(f"Unsupported layer: {layer_name}")

    return as_tensor(dst)


# pylint: disable-next=too-many-branches
def _parameters_from_t5x_decoder(src: NestedTensor, dst_layer: T5Decoder) -> NestedTensor:
    """Imports parameters from T5X Decoder param dict into AXLearn T5Decoder model state.

    Args:
        src: Params corresponding to the T5X Decoder param dict.
        dst_layer: AXLearn T5Decoder model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Decoder.
        NotImplementedError: If dst_layer.transformer is not supported.
    """
    dst = dict(
        transformer={}, emb=dict(dropout={}, token_emb={}), attention_mask={}, output_dropout={}
    )
    transformer_layers = {}

    for layer, params in src.items():
        if layer.startswith("layers_"):
            # T5X layers are named "layers_{i}".
            layer_index = int(layer.split("_")[-1])
            if isinstance(dst_layer.transformer, StackedTransformerLayer):
                # AXLearn stacked layers are named "layer{i}".
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, getattr(dst_layer.transformer, f"layer{layer_index}")
                )
            elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, dst_layer.transformer.repeat.layer
                )
        elif layer == "decoder_norm":
            dst["output_norm"] = _parameters_from_t5x_layer_norm(params)
        elif layer == "logits_dense":
            dst["lm_head"] = _parameters_from_t5x_lm_head(params)
        elif layer == "relpos_bias":
            dst["relative_pos_emb"] = _parameters_from_t5x_rel_pos_emb(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    num_axlearn_decoder_layers = dst_layer.transformer.config.num_layers
    num_t5x_decoder_layers = len(transformer_layers)
    if num_axlearn_decoder_layers != num_t5x_decoder_layers:
        raise ValueError(
            f"Number of decoder layers does not match. T5X: {num_t5x_decoder_layers}, "
            f"AXLearn: {num_axlearn_decoder_layers}."
        )

    # Traverse transformer_layers by index, to ensure that layers are ordered even if the source
    # layers are visited out-of-order.
    if isinstance(dst_layer.transformer, StackedTransformerLayer):
        for i in range(num_axlearn_decoder_layers):
            dst["transformer"][f"layer{i}"] = transformer_layers[i]
    elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
        dst["transformer"]["repeat"] = VDict(
            # pylint: disable-next=no-value-for-parameter
            layer=jax.tree.map(
                lambda *inputs: jnp.stack(inputs),
                *[transformer_layers[i] for i in range(num_axlearn_decoder_layers)],
            ),
        )
    else:
        raise NotImplementedError(type(dst_layer.transformer))

    return as_tensor(dst)


def _parameters_from_t5x_encoder(src: NestedTensor, dst_layer: T5Encoder) -> NestedTensor:
    """Imports parameters from T5X Encoder param dict into AXLearn T5Encoder model state.

    Args:
        src: Params corresponding to the T5X Encoder param dict.
        dst_layer: AXLearn T5Encoder model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Encoder.
        NotImplementedError: If dst_layer.transformer is not supported.
    """
    dst = dict(transformer={}, emb=dict(dropout={}, token_emb={}), attention_mask={})
    transformer_layers = {}

    for layer, params in src.items():
        if layer.startswith("layers_"):
            # T5X layers are named "layers_{i}".
            layer_index = int(layer.split("_")[-1])
            if isinstance(dst_layer.transformer, StackedTransformerLayer):
                # AXLearn stacked layers are named "layer{i}".
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, getattr(dst_layer.transformer, f"layer{layer_index}")
                )
            elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
                transformer_layers[layer_index] = _parameters_from_t5x_transformer_layer(
                    params, dst_layer.transformer.repeat.layer
                )
        elif layer == "encoder_norm":
            dst["output"] = dict(norm=_parameters_from_t5x_layer_norm(params), dropout={})
        elif layer == "relpos_bias":
            dst["relative_pos_emb"] = _parameters_from_t5x_rel_pos_emb(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    num_axlearn_encoder_layers = dst_layer.transformer.config.num_layers
    num_t5x_encoder_layers = len(transformer_layers)
    if num_axlearn_encoder_layers != num_t5x_encoder_layers:
        raise ValueError(
            f"Number of encoder layers does not match. T5X: {num_t5x_encoder_layers}, "
            f"AXLearn: {num_axlearn_encoder_layers}."
        )

    # Traverse transformer_layers by index, to ensure that layers are ordered even if the source
    # layers are visited out-of-order.
    if isinstance(dst_layer.transformer, StackedTransformerLayer):
        for i in range(num_axlearn_encoder_layers):
            dst["transformer"][f"layer{i}"] = transformer_layers[i]
    elif isinstance(dst_layer.transformer, RepeatedTransformerLayer):
        dst["transformer"]["repeat"] = VDict(
            # pylint: disable-next=no-value-for-parameter
            layer=jax.tree.map(
                lambda *inputs: jnp.stack(inputs),
                *[transformer_layers[i] for i in range(num_axlearn_encoder_layers)],
            ),
        )
    else:
        raise NotImplementedError(type(dst_layer.transformer))

    return as_tensor(dst)


def _parameters_from_t5x_layer_norm(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X Layer Norm param dict into AXLearn RMSNorm model state.

    Corresponding T5X module is layers.LayerNorm

    Args:
        src: Params corresponding to the T5X LayerNorm param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Encoder.
    """
    dst = {}

    for layer, params in src.items():
        if layer == "scale":
            dst["scale"] = params
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    return as_tensor(dst)


def _parameters_from_t5x_linear_like(src: NestedTensor) -> NestedTensor:
    """Imports parameters from T5X linear param dict into AXLearn Embedding or Linear model state.

    Corresponding T5X modules are layers.Embed and layers.DenseGeneral.

    Args:
        src: Params corresponding to the T5X LayerNorm param dict.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X Encoder.
    """
    dst = {}

    for layer, params in src.items():
        if layer in {"embedding", "kernel"}:
            dst["weight"] = params
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    return as_tensor(dst)


def parameters_from_t5x_encoder_decoder(
    src: NestedTensor, dst_layer: T5EncoderDecoderModel
) -> NestedTensor:
    """Imports parameters from T5X EncoderDecoder param dict into AXLearn T5EncoderDecoder model
    state.

    The corresponding T5X Flax model is network.Transformer.

    Args:
        src: Params corresponding to the T5X EncoderDecoder param dict.
        dst_layer: AXLearn T5EncoderDecoder model.

    Returns:
        NestedTensor containing corresponding AXLearn parameters for T5.

    Raises:
        ValueError: If source model is not T5X EncoderDecoder.
    """
    dst = {}

    for layer, params in src.items():
        if layer == "decoder":
            dst["decoder"] = _parameters_from_t5x_decoder(params, dst_layer.decoder)
        elif layer == "encoder":
            dst["encoder"] = _parameters_from_t5x_encoder(params, dst_layer.encoder)
        elif layer == "token_embedder":
            dst["shared_token_emb"] = _parameters_from_t5x_linear_like(params)
        else:
            raise ValueError(f"Unsupported layer: {layer}")

    return as_tensor(dst)
