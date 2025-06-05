# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# rwightman/efficientnet-jax:
# Copyright 2020 Ross Wightman.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# huggingface/pytorch-image-models:
# Copyright 2019 Ross Wightman.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Implementation of mobile networks containing MobileNetV3,
EfficientNet, EfficientNetV2 and EfficientNet-Lite.

References:
https://arxiv.org/abs/1905.02244
https://arxiv.org/abs/1905.11946
https://arxiv.org/abs/2104.00298

Implementation adapted from:
https://github.com/rwightman/efficientnet-jax/
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
"""
import contextlib
import enum
import math
import re
from collections.abc import Iterator
from copy import deepcopy
from typing import Any, Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.convolution import Conv2D
from axlearn.common.layers import (
    BatchNorm,
    Linear,
    SqueezeExcitation,
    StochasticDepth,
    get_activation_fn,
)
from axlearn.common.module import Module, Tensor
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.vision.mobilenets_blocks import (
    ConvBnAct,
    MobileBlock,
    MobileBlockType,
    SamePaddingType,
    SeReduceReference,
    round_features,
)

EFFICIENTNETVARIANTS = {
    # (feat_multiplier, depth_multiplier, resolution, dropout_rate)
    "test": (0.1, 1.0, 32, 0.2),
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.0, 1.1, 240, 0.2),
    "b2": (1.1, 1.2, 260, 0.3),
    "b3": (1.2, 1.4, 300, 0.3),
    "b4": (1.4, 1.8, 380, 0.4),
    "b5": (1.6, 2.2, 456, 0.4),
    "b6": (1.8, 2.6, 528, 0.5),
    "b7": (2.0, 3.1, 600, 0.5),
    "b8": (2.2, 3.6, 672, 0.5),
    "l2": (4.3, 5.3, 800, 0.5),
    "litetest": (0.1, 1.0, 32, 0.2),
    "lite0": (1.0, 1.0, 224, 0.2),
    "lite1": (1.0, 1.1, 240, 0.2),
    "lite2": (1.1, 1.2, 260, 0.3),
    "lite3": (1.2, 1.4, 280, 0.3),
    "lite4": (1.4, 1.8, 300, 0.3),
}


class ModelNames(str, enum.Enum):
    MOBILENETV3 = "mobilenetv3"
    EFFICIENTNET = "efficientnet"


class EndpointsMode(str, enum.Enum):
    """Endpoint mode for MobileNets / EfficientNets.

    This enum holds the modes which determine the endpoints
    returned by MobileNets / EfficientNets.

    DEFAULT: All hidden states after each block are considered endpoints.
    LASTBLOCKS: Hidden states after the last block of a stage are considered endpoints,
        if they are followed by a block with stride > 1. In addition, the very last block
        is considered a feature and the stem if followed by a convolution with stride > 1.
        This EndpointsMode is required when accessing the hidden states for another model: e.g.,
        EfficientDet uses EndpointsMode.LASTBLOCKS to access features on different levels.
    """

    DEFAULT = "default"
    LASTBLOCKS = "last_blocks"


def _mobilenet_conv2d() -> Conv2D.Config:
    # Weight initialization settings follow
    # https://github.com/rwightman/efficientnet-jax/blob/a65811fbf63cb90b9ad0724792040ce93b749303/jeffnet/linen/efficientnet_linen.py#L21-L22
    return Conv2D.default_config().set(
        param_init=DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    scale=2.0, fan="fan_out", distribution="normal"
                )
            }
        )
    )


def _mobilenet_linear() -> Linear.Config:
    # Weight initialization settings follow
    # https://github.com/rwightman/efficientnet-jax/blob/a65811fbf63cb90b9ad0724792040ce93b749303/jeffnet/linen/efficientnet_linen.py#L21-L22
    return Linear.default_config().set(
        param_init=DefaultInitializer.default_config().set(
            init_by_param_name={
                PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                    scale=1.0 / 3,
                    fan="fan_out",
                    distribution="uniform",
                )
            }
        )
    )


def _mobilenet_batchnorm2d() -> BatchNorm.Config:
    # BatchNorm settings follow the PyTorch default,
    # https://github.com/rwightman/efficientnet-jax/blob/20e1b87d04f3588dbc4470701b551ae680ea6177/jeffnet/common/constants.py#L11-L12
    return BatchNorm.default_config().set(decay=0.9, eps=1e-3)


class MobileNetV3Embedding(BaseLayer):
    """Embedding computation for MobileNetV3."""

    @config_class
    class Config(BaseLayer.Config):
        # Input dim.
        input_dim: Required[int] = REQUIRED
        # Output dim; number of features.
        output_dim: Required[int] = REQUIRED
        # Activation function.
        activation: str = "nn.relu"
        # Convolution layer.
        conv_layer: InstantiableConfig = _mobilenet_conv2d()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child(
            "conv_pw",
            cfg.conv_layer.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Computes image embedding.

        Args:
            x: A float tensor of shape [batch_size, height, width, input_dim].

        Returns:
            A float tensor of shape [batch_size, cfg.output_dim].
        """
        cfg = self.config

        x = x.mean((1, 2), keepdims=True)
        x = self.conv_pw(x)
        x = get_activation_fn(cfg.activation)(x)
        x = x.reshape(x.shape[0], cfg.output_dim)
        return x


class EfficientNetEmbedding(BaseLayer):
    """Embedding computation for EfficientNet."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures EfficientNetEmbedding."""

        # Input dim.
        input_dim: Required[int] = REQUIRED
        # Output dim; number of features.
        output_dim: Required[int] = REQUIRED
        # Activation function.
        activation: str = "nn.relu"

        # Convolution layer.
        conv_layer: InstantiableConfig = _mobilenet_conv2d()
        # Normalization layer.
        norm_layer: InstantiableConfig = _mobilenet_batchnorm2d()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        self._add_child(
            "conv_pw",
            cfg.conv_layer.set(input_dim=cfg.input_dim, output_dim=cfg.output_dim, bias=False),
        )

        self._add_child(
            "bn",
            cfg.norm_layer.set(
                input_dim=cfg.output_dim,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Computes image embedding.

        Args:
            x: A float tensor of shape [batch_size, height, width, input_dim].

        Returns:
            A float tensor of shape [batch_size, output_dim].
        """
        cfg = self.config
        x = self.conv_pw(x)
        x = self.bn(x)
        x = get_activation_fn(cfg.activation)(x)

        x = x.mean((1, 2), keepdims=True)
        x = x.reshape(x.shape[0], cfg.output_dim)
        return x


class MobileNets(BaseModel):
    """Mobile Networks."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures MobileNets."""

        # Convolution followed by BatchNorm followed by activation.
        conv_bn_act: InstantiableConfig = ConvBnAct.default_config()
        # Depthwise separable convolution.
        depthwise_separable: InstantiableConfig = MobileBlock.default_config().set(
            mobile_block_type=MobileBlockType.DEPTHWISE_SEPARABLE
        )
        # Inverted bottleneck block.
        inverted_bottleneck: InstantiableConfig = MobileBlock.default_config().set(
            mobile_block_type=MobileBlockType.INVERTED_BOTTLENECK
        )
        # Fused inverted bottleneck block.
        fused_inverted_bottleneck: InstantiableConfig = MobileBlock.default_config().set(
            mobile_block_type=MobileBlockType.FUSED_INVERTED_BOTTLENECK
        )

        # Convolution layer.
        conv_layer: InstantiableConfig = _mobilenet_conv2d()
        # Linear layer.
        linear_layer: InstantiableConfig = _mobilenet_linear()
        # Normalization layer.
        norm_layer: InstantiableConfig = _mobilenet_batchnorm2d()
        # Sqeeze excitation layer.
        se_layer: InstantiableConfig = SqueezeExcitation.default_config().set(
            gating="nn.hard_sigmoid"
        )
        # Drop path regularization.
        drop_path: InstantiableConfig = StochasticDepth.default_config()
        # Activation function applied to all blocks unless overwritten by block-specific args.
        activation: str = "nn.relu"

        # Output dim; number of features.
        output_dim: Required[int] = REQUIRED
        # Number of features produced by the stem.
        stem_size: int = 16
        # If True scale stem with feat_multiplier, else use stem_size.
        scale_stem: bool = True
        # Nested list (stages and blocks per stage) of MobileBlockType
        # and config args to instantiate the block class.
        block_defs: list[list[tuple[MobileBlockType, dict[str, Any]]]] = None
        # Width multiplier to scale the number of features.
        feat_multiplier: float = 1.0
        # Minimum factor for rounding down the number of features.
        round_limit: float = 0.9
        drop_path_rate: float = 0.0

        # Embedding layer config.
        embedding_layer: InstantiableConfig = MobileNetV3Embedding.default_config()

        # Padding type for the network.
        padding_type: SamePaddingType = SamePaddingType.DEFAULT

        # Endpoints mode, i.e., which intermediate representations to return.
        endpoints_mode: EndpointsMode = EndpointsMode.DEFAULT

        # If specified return only a subset of the intermediate representations.
        endpoints_names: Optional[set[str]] = None

    def __init__(self, cfg: Config, *, parent: Module):
        @contextlib.contextmanager
        def _register_endpoint() -> Iterator:
            # TODO(pdufter) unify endpoint computation for mobilenet, efficientnet, other backbones.
            # Register endpoints.
            if cfg.endpoints_mode == EndpointsMode.DEFAULT:
                # Default endpoints index starts from 2.
                self._endpoints_dims[str(endpoint_idx + 1)] = block_args["output_dim"]
                self._block_to_endpoint_name[f"{stage_idx}-{block_idx}"] = str(endpoint_idx + 1)
                yield True
            elif cfg.endpoints_mode == EndpointsMode.LASTBLOCKS:
                if block_idx + 1 == len(stage_defs) and (
                    stage_idx + 1 >= len(cfg.block_defs)
                    or cfg.block_defs[stage_idx + 1][0][1]["stride"] > 1
                ):
                    self._endpoints_dims[str(endpoint_idx)] = block_args["output_dim"]
                    self._block_to_endpoint_name[f"{stage_idx}-{block_idx}"] = str(endpoint_idx)
                    yield True
                else:
                    yield False
            else:
                yield False

        super().__init__(cfg, parent=parent)
        cfg = self.config

        global_block_args = dict(
            conv_layer=cfg.conv_layer,
            norm_layer=cfg.norm_layer,
            padding_type=cfg.padding_type,
        )

        input_dim = self._compute_stem_size()

        self._add_child(
            "stem",
            cfg.conv_bn_act.clone(
                input_dim=3,
                output_dim=input_dim,
                kernel_size=3,
                stride=2,
                activation=cfg.activation,
                **global_block_args,
            ),
        )

        self.vlog(3, f"Building model trunk with {len(cfg.block_defs)} stages...")
        num_blocks = sum(len(x) for x in cfg.block_defs)
        # Start counting endpoint index from 1.
        flat_idx, endpoint_idx = 0, 1
        self._stages = []

        # Store the mapping from stage and block idx to feature name.
        self._block_to_endpoint_name, self._endpoints_dims = {}, {}

        # Outer list of block_args defines the stacks.
        for stage_idx, stage_defs in enumerate(cfg.block_defs):
            self.vlog(3, f"Stack #{stage_idx}")
            blocks = []
            # Each stage contains a list of block types and arguments.
            for block_idx, (block_type, block_args) in enumerate(stage_defs):
                # Check and adjust stride.
                if block_args["stride"] not in (1, 2):
                    raise ValueError(f'Invalid value for stride: {block_args["stride"]}.')
                # Only the first block in any stack can have a stride > 1.
                if block_idx >= 1:
                    block_args["stride"] = 1
                # Store stride value for endpoint registration.
                current_stride = block_args["stride"]

                # Update block args.
                block_args["input_dim"] = input_dim
                block_args = self._update_block_args(
                    block_args=block_args,
                    global_block_args=global_block_args,
                    block_type=block_type,
                    drop_path_multiplier=flat_idx / num_blocks,
                )
                # Set input_dim of next block to be output_dim of current block.
                input_dim = block_args["output_dim"]

                # Create and add the block.
                self.vlog(
                    3,
                    f"\tBlock {block_idx},{flat_idx}/{num_blocks}, {block_type}, {str(block_args)}",
                )
                block_cfg = getattr(cfg, block_type).clone(**block_args)
                blocks.append(
                    self._add_child(f"blocks_{stage_idx}_{block_idx}", block_cfg),
                )
                # Increment flattened block idx (across all stages)
                flat_idx += 1

                # Check if we add the stem to the feature maps.
                if (
                    cfg.endpoints_mode == EndpointsMode.LASTBLOCKS
                    and stage_idx == block_idx == 0
                    and current_stride > 1
                ):
                    self._endpoints_dims["stem"] = block_args["input_dim"]
                    self._block_to_endpoint_name["stem"] = "stem"

                with _register_endpoint() as registered_endpoint:
                    endpoint_idx += int(registered_endpoint)

            self._stages.append(blocks)

        if cfg.embedding_layer is not None:
            embedding_layer_cfg = cfg.embedding_layer.set(
                input_dim=input_dim,
                output_dim=cfg.output_dim,
            )
            if hasattr(embedding_layer_cfg, "conv_layer"):
                embedding_layer_cfg.conv_layer = cfg.conv_layer
            if hasattr(embedding_layer_cfg, "activation"):
                embedding_layer_cfg.activation = cfg.activation
            if hasattr(embedding_layer_cfg, "norm_layer"):
                embedding_layer_cfg.norm_layer = cfg.norm_layer
            self._add_child("embedding_layer", embedding_layer_cfg)
            # The embedding layer after global average pooling and the final point-wise convolution.
            self._endpoints_dims["embedding"] = cfg.output_dim
            self._block_to_endpoint_name["embedding"] = "embedding"

        if cfg.endpoints_names is not None:
            self._filter_endpoints(endpoints_names=cfg.endpoints_names)

    def _compute_stem_size(self) -> int:
        cfg = self.config
        if cfg.scale_stem:
            return round_features(
                cfg.stem_size, multiplier=cfg.feat_multiplier, round_limit=cfg.round_limit
            )
        else:
            return cfg.stem_size

    def _update_block_args(
        self,
        *,
        block_args: dict[str, Any],
        global_block_args: dict[str, Any],
        block_type: MobileBlockType,
        drop_path_multiplier: float,
    ) -> dict[str, Any]:
        """Updates the block args by verifying/changing values and adding configs.

        Args:
            block_args: Block args which will be updated.
            global_block_args: Global block args that will overwrite block_args.
            block_type: The block type.
            drop_path_multiplier: Multiplier for the drop_path.

        Returns:
            The updated block args dictionary.
        """
        cfg = self.config
        block_args.update(global_block_args)

        # Set output dimension.
        block_args["output_dim"] = round_features(
            block_args["output_dim"], multiplier=cfg.feat_multiplier, round_limit=cfg.round_limit
        )

        # Block activation function overrides the model default.
        if not block_args["activation"]:
            block_args["activation"] = cfg.activation

        if block_type in {
            MobileBlockType.DEPTHWISE_SEPARABLE,
            MobileBlockType.INVERTED_BOTTLENECK,
            MobileBlockType.FUSED_INVERTED_BOTTLENECK,
        }:
            se_ratio = block_args.pop("se_ratio", 0.0)
            if se_ratio > 0.0:
                block_args["se_layer"] = cfg.se_layer.clone(se_ratio=se_ratio)
            if cfg.drop_path_rate > 0.0:
                block_args["drop_path"] = cfg.drop_path.clone(
                    rate=cfg.drop_path_rate * drop_path_multiplier
                )
        return block_args

    def _filter_endpoints(self, *, endpoints_names: set[str]):
        self._endpoints_dims = {k: self._endpoints_dims[k] for k in endpoints_names}
        self._block_to_endpoint_name = {
            k: v for (k, v) in self._block_to_endpoint_name.items() if v in endpoints_names
        }

    def forward(self, input_batch: Tensor) -> dict[str, Tensor]:
        """Compute prediction on an input batch.

        Args:
            input_batch: A float Tensor with value of shape (batch, height, width, 3).

        Returns:
            endpoints: A dict of float Tensors with values of shape
                       (batch, height_i, width_i, channels_i).
                       Keys represent the layer indices (or "embedding" for the final output),
                       values are the corresponding intermediate features of the model.
        """
        cfg = self.config
        x = self.stem(input_batch)

        endpoints = {}

        if "stem" in self._block_to_endpoint_name:
            endpoints[self._block_to_endpoint_name["stem"]] = x

        for stage_idx, stage in enumerate(self._stages):
            for block_idx, block in enumerate(stage):
                x = block(x)
                if f"{stage_idx}-{block_idx}" in self._block_to_endpoint_name:
                    endpoints[self._block_to_endpoint_name[f"{stage_idx}-{block_idx}"]] = x

        if cfg.embedding_layer is not None:
            endpoints["embedding"] = self.embedding_layer(x)
        return endpoints

    @property
    def endpoints_dims(self) -> dict[str, int]:
        """A dict of {endpoint: dim} specifies dimension of intermediate representations."""
        return self._endpoints_dims


def _decode_block_str(block_str: str) -> tuple[MobileBlockType, dict[str, Any], int]:
    """Decode block definition string.
    E.g. ib_r3_k5_s2_e3_c40_se0.25_nre

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type.
        ib = INVERTED_BOTTLENECK.
        fib = FUSED_INVERTED_BOTTLENECK.
        ds = DEPTHWISE_SEPARABLE.
        cn = CONV_BN_ACT.
    r - number of repeat blocks.
    k - kernel size.
    s - strides (1-9).
    e - expansion ratio.
    c - output channels.
    se - squeeze/excitation ratio.
    n - activation fn ('re').
    skip - use a skip connection.

    Reference:
    https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/common/arch_defs.py

    Args:
        block_str: a string representation of block arguments.

    Returns:
        A tuple of the block type (MobileBlockType), block arguments (dict),
        number of repetitions (int).

    Raises:
        NotImplementedError: If block_type is unknown.
    """
    assert isinstance(block_str, str)
    ops = block_str.split("_")
    # Take the block type off the front.
    block_type = {
        "ib": MobileBlockType.INVERTED_BOTTLENECK,
        "fib": MobileBlockType.FUSED_INVERTED_BOTTLENECK,
        "ds": MobileBlockType.DEPTHWISE_SEPARABLE,
        "cn": MobileBlockType.CONV_BN_ACT,
    }[ops[0]]
    use_skip_connection_if_possible = False
    options = {}
    for op in ops[1:]:
        # String options being checked on individual basis.
        if op == "skip":
            # Use a skip connection if the block architecture allows for it.
            use_skip_connection_if_possible = True
        elif op == "nre":
            # Activation function.
            options["n"] = "nn.relu"
        else:
            # All numeric options.
            split = re.split(r"(\d.*)", op)
            key, value = split[:2]
            options[key] = value

    # If activation is None, the model default (passed to model init) will be used.
    activation = options.get("n", None)
    se_ratio = float(options.get("se", 0.0))
    num_repeat = int(options["r"])

    # Each type of block has different valid arguments, fill accordingly.
    if block_type in {
        MobileBlockType.INVERTED_BOTTLENECK,
        MobileBlockType.FUSED_INVERTED_BOTTLENECK,
    }:
        block_args = dict(
            activation=activation,
            output_dim=int(options["c"]),
            kernel_size=int(options["k"]),
            stride=int(options["s"]),
            exp_ratio=float(options["e"]),
            se_ratio=se_ratio,
        )
    elif block_type == MobileBlockType.DEPTHWISE_SEPARABLE:
        block_args = dict(
            activation=activation,
            output_dim=int(options["c"]),
            kernel_size=int(options["k"]),
            stride=int(options["s"]),
            se_ratio=se_ratio,
        )
    elif block_type == MobileBlockType.CONV_BN_ACT:
        block_args = dict(
            activation=activation,
            output_dim=int(options["c"]),
            kernel_size=int(options["k"]),
            stride=int(options["s"]),
            use_skip_connection_if_possible=use_skip_connection_if_possible,
        )
    else:
        raise NotImplementedError(f"Unknown block type: {block_type}.")

    return block_type, block_args, num_repeat


def _decode_arch_def(
    arch_def: list[list[str]],
    depth_multiplier: float = 1.0,
    depth_trunc: str = "ceil",
    fix_first_last: bool = False,
) -> list[list[tuple[MobileBlockType, dict[str, Any]]]]:
    """Decodes model architecture definition into corresponding block types and block arguments.

    Args:
        arch_def: A representation of the model architecture as a nested list of strings.
        depth_multiplier: Multiplier to scale the depth of stages.
        depth_trunc: Truncation policy for computing the depth of stages.
        fix_first_last: If true, prefer to have smaller depth in first and last stage.

    Returns:
        A nested list (stages and blocks per stage) of the types and arguments of each block.
    """
    arch_args = []
    for stage_idx, block_strings in enumerate(arch_def):
        stage_args = []
        repeats = []
        for block_str in block_strings:
            block_type, block_args, num_repeat = _decode_block_str(block_str)
            stage_args += [deepcopy((block_type, block_args)) for _ in range(num_repeat)]
            repeats.append(num_repeat)
        if fix_first_last and (stage_idx in (0, len(arch_def) - 1)):
            arch_args.append(_scale_stage_depth(stage_args, repeats, 1.0, depth_trunc))
        else:
            arch_args.append(_scale_stage_depth(stage_args, repeats, depth_multiplier, depth_trunc))
    return arch_args


def _arch_mobilenet_v3(variant: str) -> dict[str, Any]:
    """Defines different MobileNetV3 model architectures.

    Reference:
    https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/common/arch_defs.py

    Args:
        variant: the model name. Variant naming follows f"MobileNetV3-{scale}-{feat_multiplier}",
            where scale can be "smaller" or "large" and "feat_multiplier" is a number between 1 to
            100, representing the feat scaling percentage. e.g. MobileNetV3-large-100.

    Returns:
        A dictionary specifying the model definition (e.g. block structure).

    Raises:
        ValueError: undefined variant name.
    """
    if "small" in variant:
        output_dim = 1024
        if "minimal" in variant:
            activation = "nn.relu"
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s2_e1_c16"],
                # stage 1, 56x56 in
                ["ib_r1_k3_s2_e4.5_c24", "ib_r1_k3_s1_e3.67_c24"],
                # stage 2, 28x28 in
                ["ib_r1_k3_s2_e4_c40", "ib_r2_k3_s1_e6_c40"],
                # stage 3, 14x14 in
                ["ib_r2_k3_s1_e3_c48"],
                # stage 4, 14x14in
                ["ib_r3_k3_s2_e6_c96"],
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c576"],
            ]
        else:
            activation = "nn.hard_swish"
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s2_e1_c16_se0.25_nre"],  # relu
                # stage 1, 56x56 in
                ["ib_r1_k3_s2_e4.5_c24_nre", "ib_r1_k3_s1_e3.67_c24_nre"],  # relu
                # stage 2, 28x28 in
                ["ib_r1_k5_s2_e4_c40_se0.25", "ib_r2_k5_s1_e6_c40_se0.25"],  # hard-swish
                # stage 3, 14x14 in
                ["ib_r2_k5_s1_e3_c48_se0.25"],  # hard-swish
                # stage 4, 14x14in
                ["ib_r3_k5_s2_e6_c96_se0.25"],  # hard-swish
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c576"],  # hard-swish
            ]
    elif "large" in variant:
        output_dim = 1280
        if "minimal" in variant:
            activation = "nn.relu"
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s1_e1_c16"],
                # stage 1, 112x112 in
                ["ib_r1_k3_s2_e4_c24", "ib_r1_k3_s1_e3_c24"],
                # stage 2, 56x56 in
                ["ib_r3_k3_s2_e3_c40"],
                # stage 3, 28x28 in
                ["ib_r1_k3_s2_e6_c80", "ib_r1_k3_s1_e2.5_c80", "ib_r2_k3_s1_e2.3_c80"],
                # stage 4, 14x14in
                ["ib_r2_k3_s1_e6_c112"],
                # stage 5, 14x14in
                ["ib_r3_k3_s2_e6_c160"],
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c960"],
            ]
        else:
            activation = "nn.hard_swish"
            arch_def = [
                # stage 0, 112x112 in
                ["ds_r1_k3_s1_e1_c16_nre"],  # relu
                # stage 1, 112x112 in
                ["ib_r1_k3_s2_e4_c24_nre", "ib_r1_k3_s1_e3_c24_nre"],  # relu
                # stage 2, 56x56 in
                ["ib_r3_k5_s2_e3_c40_se0.25_nre"],  # relu
                # stage 3, 28x28 in
                [
                    "ib_r1_k3_s2_e6_c80",
                    "ib_r1_k3_s1_e2.5_c80",
                    "ib_r2_k3_s1_e2.3_c80",
                ],  # hard-swish
                # stage 4, 14x14in
                ["ib_r2_k3_s1_e6_c112_se0.25"],  # hard-swish
                # stage 5, 14x14in
                ["ib_r3_k5_s2_e6_c160_se0.25"],  # hard-swish
                # stage 6, 7x7 in
                ["cn_r1_k1_s1_c960"],  # hard-swish
            ]
    else:
        raise ValueError(f"Undefined MobileNet variant {variant}.")

    model_kwargs = dict(
        block_defs=_decode_arch_def(arch_def),
        output_dim=output_dim,
        feat_multiplier=float(variant.split("-")[-1]) * 0.01,
        activation=activation,
    )
    return model_kwargs


def _scale_stage_depth(
    stage_defs: list[tuple[MobileBlockType, dict]],
    repeats: list[int],
    depth_multiplier: float = 1.0,
    depth_trunc: str = "ceil",
) -> list[tuple[MobileBlockType, dict]]:
    """Per-stage depth scaling.

    Scales the block repeats in each stage.

    Args:
        stage_defs: Architecture definitions for stages.
        repeats: Number of repeats for each stages.
        depth_multiplier: Multiplier to scale the depth of stages.
        depth_trunc: Truncation policy for computing the depth of stages.

    Returns:
        A nested list (stages and blocks per stage) of the types and arguments of each block.
    """

    # We scale the total repeat count for each stage, there may be multiple
    # block arg defs per stage so we need to sum.
    num_repeat = sum(repeats)
    if depth_trunc == "round":
        # Truncating to int by rounding allows stages with few repeats to remain
        # proportionally smaller for longer. This is a good choice when stage definitions
        # include single repeat stages that we'd prefer to keep that way as long as possible.
        num_repeat_scaled = max(1, round(num_repeat * depth_multiplier))
    else:
        # The default for EfficientNet truncates repeats to int via 'ceil'.
        # Any multiplier > 1.0 will result in an increased depth for every stage.
        num_repeat_scaled = int(math.ceil(num_repeat * depth_multiplier))

    # Proportionally distribute repeat count scaling to each block definition in the stage.
    # Allocation is done in reverse as it results in the first block being less likely to be scaled.
    # The first block makes less sense to repeat in most of the arch definitions.
    repeats_scaled = []
    for repeat in repeats[::-1]:
        repeat_scaled = max(1, round(repeat / num_repeat * num_repeat_scaled))
        repeats_scaled.append(repeat_scaled)
        num_repeat -= repeat
        num_repeat_scaled -= repeat_scaled
    repeats_scaled = repeats_scaled[::-1]

    # Apply the calculated scaling to each block def in the stage.
    defs_scaled = []
    for stage_def, repeat_scaled in zip(stage_defs, repeats_scaled):
        defs_scaled.extend([deepcopy(stage_def) for _ in range(repeat_scaled)])
    return defs_scaled


def _arch_efficientnet(variant: str, version: str = "V1", **kwargs) -> dict[str, Any]:
    """Defines EfficientNet model architectures (both V1 and V2).

    References:
    https://github.com/rwightman/efficientnet-jax/blob/master/jeffnet/common/arch_defs.py
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py

    Papers:
    https://arxiv.org/abs/1905.11946
    https://arxiv.org/abs/2104.00298

    Args:
        variant: The desired EfficientNet variant, e.g., "b0", "lite0".
        version: The desired EfficientNet version, i.e.,
            "V1" (https://arxiv.org/abs/1905.11946) or "V2" (https://arxiv.org/abs/2104.00298).

    Returns:
        A dictionary specifying the model definition.
    """

    variant = variant.lower()
    # For EfficientNetV1, we ONLY support the variants in EFFICIENTNETVARIANTS.
    # For EfficientNetV2, we ADDITIONALLY support the S, M, L variants from the original paper.
    if variant in EFFICIENTNETVARIANTS:
        feat_multiplier, depth_multiplier, _, _ = EFFICIENTNETVARIANTS[variant]
    elif version == "V2":
        feat_multiplier, depth_multiplier = 1.0, 1.0
    else:
        raise ValueError(f"Invalid EfficientNetV1 variant: {variant}.")
    round_limit = 0.9 if version == "V1" else 0.0

    if variant.startswith("lite"):
        fix_first_last = True
        output_dim = 1280
        se_cfg = ""
        activation = "nn.relu6"
        scale_stem = False
    else:
        fix_first_last = False
        output_dim = round_features(1280, multiplier=feat_multiplier, round_limit=round_limit)
        se_ratio = 0.25
        se_cfg = f"_se{se_ratio}"
        activation = "nn.swish"
        scale_stem = True

    if version == "V1":
        stem_size = 32
        arch_def = [
            [f"ds_r1_k3_s1_e1_c16{se_cfg}"],
            [f"ib_r2_k3_s2_e6_c24{se_cfg}"],
            [f"ib_r2_k5_s2_e6_c40{se_cfg}"],
            [f"ib_r3_k3_s2_e6_c80{se_cfg}"],
            [f"ib_r3_k5_s1_e6_c112{se_cfg}"],
            [f"ib_r4_k5_s2_e6_c192{se_cfg}"],
            [f"ib_r1_k3_s1_e6_c320{se_cfg}"],
        ]
    elif version == "V2":
        if variant == "s":
            stem_size = 24
            arch_def = [
                ["cn_r2_k3_s1_e1_c24_skip"],
                ["fib_r4_k3_s2_e4_c48"],
                ["fib_r4_k3_s2_e4_c64"],
                ["ib_r6_k3_s2_e4_c128_se0.25"],
                ["ib_r9_k3_s1_e6_c160_se0.25"],
                ["ib_r15_k3_s2_e6_c256_se0.25"],
            ]
        elif variant == "m":
            stem_size = 24
            arch_def = [
                ["cn_r3_k3_s1_e1_c24_skip"],
                ["fib_r5_k3_s2_e4_c48"],
                ["fib_r5_k3_s2_e4_c80"],
                ["ib_r7_k3_s2_e4_c160_se0.25"],
                ["ib_r14_k3_s1_e6_c176_se0.25"],
                ["ib_r18_k3_s2_e6_c304_se0.25"],
                ["ib_r5_k3_s1_e6_c512_se0.25"],
            ]
        elif variant == "l":
            stem_size = 32
            arch_def = [
                ["cn_r4_k3_s1_e1_c32_skip"],
                ["fib_r7_k3_s2_e4_c64"],
                ["fib_r7_k3_s2_e4_c96"],
                ["ib_r10_k3_s2_e4_c192_se0.25"],
                ["ib_r19_k3_s1_e6_c224_se0.25"],
                ["ib_r25_k3_s2_e6_c384_se0.25"],
                ["ib_r7_k3_s1_e6_c640_se0.25"],
            ]
        elif variant in EFFICIENTNETVARIANTS:
            stem_size = 32
            arch_def = [
                ["cn_r1_k3_s1_e1_c16_skip"],
                ["fib_r2_k3_s2_e4_c32"],
                ["fib_r2_k3_s2_e4_c48"],
                [f"ib_r3_k3_s2_e4_c96{se_cfg}"],
                [f"ib_r5_k3_s1_e6_c112{se_cfg}"],
                [f"ib_r8_k3_s2_e6_c192{se_cfg}"],
            ]
        else:
            raise ValueError(f"Invalid EfficientNetV2 variant: {variant}.")
    else:
        raise ValueError('EfficientNet version must be either "V1" or "V2".')

    model_kwargs = dict(
        block_defs=_decode_arch_def(arch_def, depth_multiplier, fix_first_last=fix_first_last),
        output_dim=output_dim,
        stem_size=stem_size,
        feat_multiplier=feat_multiplier,
        round_limit=round_limit,
        activation=activation,
        se_layer=SqueezeExcitation.default_config().set(
            gating="nn.sigmoid",
            activation="nn.swish",
        ),
        depthwise_separable=MobileBlock.default_config().set(
            mobile_block_type=MobileBlockType.DEPTHWISE_SEPARABLE,
            se_reduction_divisor=1,
        ),
        inverted_bottleneck=MobileBlock.default_config().set(
            mobile_block_type=MobileBlockType.INVERTED_BOTTLENECK,
            se_reduce_ref=SeReduceReference.INPUT_DIM,
            se_reduction_divisor=1,
        ),
        fused_inverted_bottleneck=MobileBlock.default_config().set(
            mobile_block_type=MobileBlockType.FUSED_INVERTED_BOTTLENECK,
            se_reduction_divisor=1,
        ),
        embedding_layer=EfficientNetEmbedding.default_config(),
        drop_path_rate=0.2,
        scale_stem=scale_stem,
        **kwargs,
    )
    # Remove repeated blocks in test models.
    # TODO(xianzhi): use a better way determine a variant is only used for testing,
    # e.g. introducing a helper function or an additional flag to EFFICIENTNETVARIANTS.
    if "test" in variant:
        model_kwargs["block_defs"] = [[x[0]] for x in model_kwargs["block_defs"]]
    return model_kwargs


def named_model_configs(
    model_name: ModelNames,
    variant: str,
    *,
    efficientnet_version: Optional[str] = "V1",
    extra_settings: Optional[dict[str, Any]] = None,
) -> InstantiableConfig:
    if model_name == ModelNames.EFFICIENTNET:
        model_settings = _arch_efficientnet(variant, efficientnet_version)
    elif model_name == ModelNames.MOBILENETV3:
        model_settings = _arch_mobilenet_v3(variant)
    else:
        raise ValueError(f"Unsupported model: {model_name=}.")
    if extra_settings:
        model_settings.update(extra_settings)
    cfg = MobileNets.default_config().set(**model_settings)
    return cfg
