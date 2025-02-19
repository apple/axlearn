# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# microsoft/lora:
# Copyright (c) Microsoft Corporation.

"""Low-Rank Adaptation (LoRA) for Large Language Models.

RefA: original LoRA paper https://arxiv.org/abs/2106.09685
RefB: generalized LoRA https://arxiv.org/pdf/2110.04366
RefC: Pytorch implementation https://github.com/microsoft/lora
"""
from typing import Optional

import jax.numpy as jnp

from axlearn.common.attention import (
    BaseMultiheadLinear,
    BaseQKVLinear,
    FusedQKVLinear,
    MultiheadOutputLinear,
)
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import REQUIRED, InstantiableConfig, Required, config_class
from axlearn.common.layers import Dropout, Linear
from axlearn.common.module import Module
from axlearn.common.param_init import (
    PARAM_REGEXP_WEIGHT,
    ConstantInitializer,
    DefaultInitializer,
    FanAxes,
)
from axlearn.common.utils import Tensor


class _BaseLoraAdapter(BaseLayer):
    """An abstract class to define the common interface of all LoRA Adapter."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures _BaseLoraAdapter."""

        # The input feature dimension.
        input_dim: Required[int] = REQUIRED

        # The output feature dimension.
        output_dim: Required[int] = REQUIRED

        # The rank of the LoRA. In order to reduce the number of trainable
        # parameters, rank should be greatly smaller than the input_dim and output_dim.
        # It is required that rank > 0.
        rank: Required[int] = REQUIRED

        # Value alpha/rank is used to scale the output of LoRA Adapter.
        alpha: Required[float] = REQUIRED

        # Drop out input for LoRA layer. This dropout rate can be set differently from
        # the original layer.
        # We set `rate` explicitly so that users can call
        # `set_dropout_rate_recursively(..., set_only_if_none=True)` without overriding
        # the LoRA dropout rate.
        dropout: Dropout.Config = Dropout.default_config().set(rate=0.0)

        # Linear layer that projects the input feature from input_dim to rank.
        # That is, the lora_A matrix in RefA.
        lora_down: InstantiableConfig = Linear.default_config()

        # Linear layer that projects intermediate representations from rank to output_dim.
        # That is, the lora_B matrix in RefA.
        lora_up: InstantiableConfig = Linear.default_config()

    @property
    def scaling(self):
        cfg = self.config
        if cfg.rank > 0:
            return cfg.alpha / cfg.rank
        else:
            return 1

    def _is_valid_config(self):
        cfg = self.config
        cls_name = self.path()
        # Check whether rank is valid.
        if cfg.rank == 0:
            raise ValueError(f"{cls_name}'s rank should not be set as 0.")
        if cfg.rank > cfg.input_dim:
            raise ValueError(
                f"{cls_name}'s down-rank {cfg.rank} is greater than input_dim {cfg.input_dim}."
            )
        if cfg.rank > cfg.output_dim:
            raise ValueError(
                f"{cls_name}'s up-rank {cfg.rank} is greater than output_dim {cfg.output_dim}."
            )

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._is_valid_config()
        # Add on LoRA dropout layer.
        # Drop out input randomly before passing to LoRA layers.
        self._add_child("lora_dropout", cfg.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.lora_down(self.lora_dropout(inputs))
        outputs = self.lora_up(outputs)
        return outputs * self.scaling


class LoraLinearAdapter(_BaseLoraAdapter):
    """Adapter to be used in parallel to Linear layers."""

    def __init__(self, cfg: _BaseLoraAdapter.Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        # Initialize the LoRA-A weight matrix in RefA.
        self._add_child(
            "lora_down",
            cfg.lora_down.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.rank,
                bias=False,
            ),
        )
        # Initialize the LoRA-B weight matrix in RefA.
        # According to RefA, this weight is initiated as 0.
        self._add_child(
            "lora_up",
            cfg.lora_up.set(
                input_dim=cfg.rank,
                output_dim=cfg.output_dim,
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: ConstantInitializer.default_config().set(value=0.0)
                    }
                ),
                bias=False,
            ),
        )


class LoraDownFusedLinear(BaseLayer):
    """The linear layer used for fused LoRA-down projection in LoraFusedQKVAdapter.

    It uses einsum for efficient computation on TPU to avoid reshaping.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures LoraDownFusedLinear."""

        input_dim: Required[int] = REQUIRED  # Feature dim.
        num_enabled: Required[int] = REQUIRED  # Number of enabled LoRA projections.
        output_dim: Required[int] = REQUIRED  # Output feature dim.

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        cfg.param_partition_spec = (None, None, "model")
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.input_dim, cfg.num_enabled, cfg.output_dim),
                mesh_axes=cfg.param_partition_spec,
            )
        )
        return params

    def forward(self, inputs: Tensor) -> Tensor:
        params = self.parameters
        # Output shape: [num_enabled, batch, seq_len, LoRA_rank].
        return jnp.einsum("btd,dsr->sbtr", inputs, params["weight"])

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            return FanAxes(in_axis=0, out_axis=(1, 2))
        else:
            return None


class LoraUpFusedLinear(BaseLayer):
    """The linear layer used for fused LoRA's up-projection in LoraFusedQKVAdapter.

    It uses einsum for efficient computation on TPU to avoid reshaping.
    """

    @config_class
    class Config(BaseLayer.Config):
        """Configures LoraUpFusedLinear."""

        input_dim: Required[int] = REQUIRED  # Feature dim.
        num_enabled: Required[int] = REQUIRED  # Number of attention heads.
        num_heads: Required[int] = REQUIRED  # Dimension per head.
        per_head_dim: Required[int] = REQUIRED  # Dimension per head.

    @classmethod
    def default_config(cls) -> Config:
        cfg = super().default_config()
        # Shard the 'num_heads' axis by the 'model' dim of the mesh.
        cfg.param_partition_spec = (None, None, "model", None)
        return cfg

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            weight=ParameterSpec(
                shape=(cfg.num_enabled, cfg.input_dim, cfg.num_heads, cfg.per_head_dim),
                mesh_axes=cfg.param_partition_spec,
            )
        )
        return params

    def forward(self, inputs: Tensor) -> Tensor:
        params = self.parameters
        # Output shape: [num_enabled, batch, seq_len, num_heads, per_head_dim].
        return jnp.einsum("sbtr,srnh->sbtnh", inputs, params["weight"])

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            return FanAxes(in_axis=1, out_axis=(2, 3))
        else:
            return None


class LoraDownMultiheadLinear(BaseMultiheadLinear):
    """The linear layer used for LoRA's down-projection in LoraMultiheadOutputAdapter."""

    @property
    def _einsum_expr(self):
        return "btnh,rnh->btr"

    @property
    def _bias_spec(self):
        cfg = self.config
        return ParameterSpec(
            shape=(cfg.model_dim,),
            mesh_axes=cfg.param_partition_spec[:1],
        )

    # pylint: disable-next=no-self-use
    def _compute_fan_axes(self, name: str, parameter_spec: ParameterSpec) -> Optional[FanAxes]:
        if name == "weight":
            return FanAxes(in_axis=(1, 2), out_axis=0)
        else:
            return None


class LoraMultiheadOutputAdapter(_BaseLoraAdapter):
    """Adapter to be used in parallel to MultiheadOutputLinear layers."""

    @config_class
    class Config(_BaseLoraAdapter.Config):
        """Configures LoraMultiheadOutputAdapter."""

        num_heads: Required[int] = REQUIRED

    @classmethod
    def default_config(cls) -> Config:
        cfg: _BaseLoraAdapter.Config = super().default_config()
        cfg.lora_down = LoraDownMultiheadLinear.default_config()
        return cfg

    @property
    def _per_head_dim(self):
        cfg = self.config
        return cfg.output_dim // cfg.num_heads

    def _is_valid_config(self):
        super()._is_valid_config()
        cfg = self.config
        if cfg.output_dim != cfg.input_dim:
            raise ValueError(
                f"In MultiheadOutputLinear, the input dim {cfg.input_dim} should be"
                f" the same as output dim {cfg.output_dim}."
            )
        if cfg.output_dim % cfg.num_heads != 0:
            raise ValueError(
                f"Output dim {cfg.output_dim} is not divisible by num_heads {cfg.num_heads}."
            )

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "lora_down",
            cfg.lora_down.set(
                model_dim=cfg.rank,
                num_heads=cfg.num_heads,
                per_head_dim=self._per_head_dim,
                bias=False,
            ),
        )
        self._add_child(
            "lora_up",
            cfg.lora_up.set(
                input_dim=cfg.rank,
                output_dim=cfg.output_dim,
                bias=False,
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: ConstantInitializer.default_config().set(value=0.0)
                    }
                ),
            ),
        )


class LoraFusedQKVAdapter(_BaseLoraAdapter):
    """Adapter to be used in parallel to FusedQKVLinear layers.

    The implementation reproduces the one in RefC.

    In the FusedQKVLinear layer, query, key, and value are concatenated together
    before applying the projection. However, it is not necessary to have LoRA
    adapters for all three Q, K, V projections.

    This adapter first projects the fused inputs from dimension input_dim
    to (rank * s), where 0 <= s <= 3 is the number of needed LoRA adapters, then
    projects it back to (s * output_dim), where output_dim = num_heads * per_head_dim.
    """

    @config_class
    class Config(_BaseLoraAdapter.Config):
        """Configures LoraFusedQKVAdapter."""

        # Whether to use LoRA for query, key, value projections.
        # Keys of enable_lora must be query, key, value.
        enable_lora: Required[dict[str, bool]] = REQUIRED
        num_heads: Required[int] = REQUIRED

    @classmethod
    def default_config(cls) -> Config:
        cfg: _BaseLoraAdapter.Config = super().default_config()
        cfg.lora_down = LoraDownFusedLinear.default_config()
        cfg.lora_up = LoraUpFusedLinear.default_config()
        return cfg

    @property
    def _per_head_dim(self):
        cfg = self.config
        return cfg.output_dim // cfg.num_heads

    @property
    def _num_enabled(self):
        cfg = self.config
        return sum(cfg.enable_lora.values())

    def _is_valid_config(self):
        super()._is_valid_config()
        # Check whether enable_lora is valid if exists.
        cfg = self.config
        if set(cfg.enable_lora.keys()) != {"query", "key", "value"}:
            raise ValueError(
                f"Keys of enable_lora must be query, key, value, saw: {cfg.enable_lora}."
            )
        if not all(isinstance(val, bool) for val in cfg.enable_lora.values()):
            raise ValueError(f"Values of enable_lora must be boolean, saw: {cfg.enable_lora}.")
        if cfg.output_dim % cfg.num_heads != 0:
            raise ValueError(
                f"Output dim {cfg.output_dim} is not divisible by num_heads {cfg.num_heads}."
            )

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        self._add_child(
            "lora_down",
            cfg.lora_down.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.rank,
                num_enabled=self._num_enabled,
            ),
        )
        self._add_child(
            "lora_up",
            cfg.lora_up.set(
                input_dim=cfg.rank,
                num_heads=cfg.num_heads,
                per_head_dim=self._per_head_dim,
                num_enabled=self._num_enabled,
                param_init=DefaultInitializer.default_config().set(
                    init_by_param_name={
                        PARAM_REGEXP_WEIGHT: ConstantInitializer.default_config().set(value=0.0)
                    }
                ),
            ),
        )


class LoraLinear(BaseLayer):
    """LoRA's replacement of the Linear layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures LoraLinear."""

        # Input feature dim.
        input_dim: Required[int] = REQUIRED
        # Output feature dim.
        output_dim: Required[int] = REQUIRED
        # The original linear layer config.
        layer: Linear.Config = Linear.default_config()
        # The adapter config for LoRA.
        adapter: LoraLinearAdapter.Config = LoraLinearAdapter.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)  # Initiate the original linear layer.
        cfg = self.config
        self._add_child(
            "layer",
            cfg.layer.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )
        self._add_child(
            "adapter",
            cfg.adapter.set(
                input_dim=cfg.input_dim,
                output_dim=cfg.output_dim,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x) + self.adapter(x)


class LoraMultiheadOutputLinear(BaseLayer):
    """LoRA's replacement of the MultiheadOutputLinear layer."""

    @config_class
    class Config(BaseLayer.Config):
        """Configures LoraMultiheadOutputLinear."""

        model_dim: Required[int] = REQUIRED  # Feature dim.
        num_heads: Required[int] = REQUIRED  # Number of attention heads.
        per_head_dim: Required[int] = REQUIRED  # Dimension per head.
        # The original linear layer config.
        layer: MultiheadOutputLinear.Config = MultiheadOutputLinear.default_config()
        # The adapter config for LoRA.
        adapter: LoraMultiheadOutputAdapter.Config = LoraMultiheadOutputAdapter.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)  # Initiate the original linear layer.
        cfg = self.config
        self._add_child(
            "layer",
            cfg.layer.set(
                model_dim=cfg.model_dim, num_heads=cfg.num_heads, per_head_dim=cfg.per_head_dim
            ),
        )
        self._add_child(
            "adapter",
            cfg.adapter.set(
                input_dim=cfg.model_dim, output_dim=cfg.model_dim, num_heads=cfg.num_heads
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x) + self.adapter(x)


class LoraFusedQKVLinear(BaseQKVLinear):
    """Fused LoRA replacement for non-grouped QKVLinear layers.

    N.B. Only supports cases where (1) key and value are None; (2) query, key,
    and value all have the same shape; (3) model_dim = num_heads * per_head_dim.
    """

    @config_class
    class Config(BaseQKVLinear.Config):
        """Configures LoraFusedQKVLinear."""

        # The original QKVLinear layer config.
        layer: BaseQKVLinear.Config = FusedQKVLinear.default_config()
        # The adapter config for LoRA.
        adapter: LoraFusedQKVAdapter.Config = LoraFusedQKVAdapter.default_config()

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        cfg = self.config
        if not cfg.query_dim == cfg.key_dim == cfg.value_dim:
            raise ValueError(
                f"All projection dims must be equal for {type(self)}, saw: "
                f"query:{cfg.query_dim}, key:{cfg.key_dim}, value:{cfg.value_dim}"
            )
        qkv_proj_cfg = cfg.layer
        qkv_proj_cfg.query_dim = cfg.query_dim
        qkv_proj_cfg.key_dim = cfg.key_dim
        qkv_proj_cfg.value_dim = cfg.value_dim
        qkv_proj_cfg.num_heads = cfg.num_heads
        qkv_proj_cfg.per_head_dim = cfg.per_head_dim
        qkv_proj_cfg.cache_dtype = cfg.cache_dtype
        self._add_child("layer", qkv_proj_cfg)
        self._add_child(
            "adapter",
            cfg.adapter.set(
                input_dim=cfg.query_dim,
                output_dim=cfg.num_heads * cfg.per_head_dim,
                num_heads=cfg.num_heads,
            ),
        )

    @property
    def num_kv_heads(self):
        return self.layer.num_kv_heads

    def forward(
        self,
        query: Tensor,
        *,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        kv_state: Optional[Tensor] = None,
        query_positions: Optional[Tensor] = None,
    ) -> BaseQKVLinear.Output:
        cfg = self.config
        if key is None and value is None:
            inputs = query
        else:
            raise ValueError("Key and value should be both None in LoraFusedQKVLinear.")

        q_proj, k_proj, v_proj = self.layer(
            query, key=key, value=value, kv_state=kv_state, query_positions=query_positions
        )
        adapter_outputs = self.adapter(inputs)

        index = 0
        if cfg.adapter.enable_lora["query"]:
            q_proj = q_proj + adapter_outputs[index]
            index = index + 1

        if cfg.adapter.enable_lora["key"]:
            k_proj = k_proj + adapter_outputs[index]
            index = index + 1

        if cfg.adapter.enable_lora["value"]:
            v_proj = v_proj + adapter_outputs[index]

        return self.Output(query=q_proj, key=k_proj, value=v_proj)
