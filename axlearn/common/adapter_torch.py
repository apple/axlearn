# Copyright © 2023 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# bzhangGo/rmsnorm:
# Copyright (c) 2019, Biao Zhang. All rights reserved.
# Licensed under the BSD 3-Clause License.
#
# ofirpress/attention_with_linear_biases:
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT license.

"""This file is intended to be a standalone PyTorch re-implementation of a subset
of the layers + models available in AXLearn. As it is standalone, it should only
depend on PyTorch and freely available Python libraries.

The goal is to provide a straightforward way for developers to 1. initialize an equivalent
model architecture in PyTorch, and 2. load AXLearn weights into it.

N.B. Not all AXLearn functionality is replicated here, although what is replicated
is tested. For your particular mapping usecase, you may need to add features
and/or layers.
"""
# pylint: disable=no-self-use,duplicate-code,too-many-lines
import copy
import itertools
import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, NamedTuple, Optional, Protocol, Union

import numpy as np
import torch
from torch import nn

NEG_INF = -1e15


# pylint: disable-next=abstract-method
class TorchModule(torch.nn.Module):
    """Equips a PyTorch Module with a mechanism to load a flattened AXLearn weight state dict.

    All PyTorch layers must have this as a mixin to facilitate loading state of AXLearn origin.
    """

    def load_axlearn_state_dict(
        self, state_dict: OrderedDict[str, torch.Tensor], strict: bool = True
    ):
        """Load state from state dict for all sub-modules recursively.

        Args:
            state_dict: A flattened version of AXLearn state, with '.' separators and
                weights as torch Tensors.
            strict: If True, complain if any TorchModule state did not find
                a counterpart in the input state dict.
        """

        def load(module, prefix=""):
            # pylint: disable-next=protected-access
            module._local_load(state_dict, prefix=prefix, strict=strict)
            # pylint: disable-next=protected-access
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self)

    def _local_load(self, state_dict: dict[str, Any], prefix: str, strict: bool):
        """Load state for this local Module only.

        Args:
            state_dict: A flattened version of AXLearn state, with '.' separators and
                weights as torch Tensors.
            prefix: A prefix filter on keys in state-dict, describing the ancestry
                of this Module.
            strict: If True, complain if any TorchModule state did not find
                a counterpart in the input state dict.

        Raises:
            RuntimeError: If strict and local state didn't find a counterpart in
                state dict.
            ValueError: If the parameter's shapes don't match.
        """
        persistent_buffers = {
            k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
        local_state = {k: v for k, v in local_name_params if v is not None}
        not_found = []
        for name, param in local_state.items():
            axlearn_name = self._axlearn_name_mapper(weight_name=name)
            key = prefix + axlearn_name
            if key in state_dict:
                input_param = self._axlearn_weight_mapper(
                    weight_name=axlearn_name, weight=state_dict[key]
                )
                # torch.Tensor.copy_ allows broadcasting, resulting in loading a linear(in=5, out=1)
                # weights into a linear(in=5, out=10) layer quietly.
                if input_param.shape != param.shape:
                    raise ValueError(
                        f"shapes not same for {key=}, "
                        f"expected: {param.shape}, received:{input_param.shape}"
                    )
                with torch.no_grad():
                    param.copy_(input_param)
            else:
                not_found.append(key)

        if not_found and strict:
            raise RuntimeError(f"Didn't load state for {not_found}.")

    # pylint: disable-next=unused-argument,no-self-use
    def _axlearn_weight_mapper(self, weight_name: str, weight: torch.Tensor) -> torch.Tensor:
        """Given a module type, weight name, and AXLearn weight tensor returns a torch tensor
            of the correct shape for that module's named weight.

        Args:
            weight_name: Name of the weight within the module.
            weight: The AXLearn weight that we want to massage into a suitable format.

        Returns:
            A torch weight suitable for loading into the named weight slot for the module.
        """
        return weight

    # pylint: disable-next=no-self-use
    def _axlearn_name_mapper(self, weight_name: str) -> str:
        """Given a module type and a weight name for that module, return the equivalent AXLearn
        weight name.

        Args:
            weight_name: Name of the weight within the module.

        Returns:
            The name corresponding to this PyTorch weight in AXLearn world.
        """
        return weight_name

    @property
    def device(self) -> torch.device:
        """Returns the device of the first available parameter, if no parameters return 'cpu'."""
        device = torch.device("cpu")
        for param in self.parameters():
            device = param.device
            break
        return device


# Torch utils:


def gelu(x):
    # Matches default jax.nn.gelu.
    sqrt_2_over_pi = torch.as_tensor([np.sqrt(2 / np.pi)]).type(x.dtype).to(x.device)
    cdf = 0.5 * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * (x**3))))
    return x * cdf


class Conv2d(torch.nn.Conv2d, TorchModule):
    def _axlearn_weight_mapper(self, weight_name: str, weight: torch.Tensor) -> torch.Tensor:
        # Conv2d.weight uses layout (output, input, H, W), AXLearn assumes (H, W, input, output).
        if weight_name == "weight":
            return weight.permute(3, 2, 0, 1)
        return weight


class Dropout(nn.Dropout, TorchModule):
    pass


class Embedding(nn.Embedding, TorchModule):
    def _axlearn_weight_mapper(self, weight_name: str, weight: torch.Tensor) -> torch.Tensor:
        # For positional embedding weights, AXLearn adds a leading axis.
        if weight_name == "weight":
            return weight.squeeze()
        return weight


class LayerNorm(nn.LayerNorm, TorchModule):
    def __init__(self, *args, eps=1e-6, **kwargs):
        super().__init__(*args, eps=eps, **kwargs)

    def _axlearn_name_mapper(self, weight_name: str) -> str:
        # LayerNorm uses "weight" where AXLearn uses "scale".
        if weight_name == "weight":
            return "scale"
        return weight_name


class Linear(nn.Linear, TorchModule):
    def _axlearn_weight_mapper(self, weight_name: str, weight: torch.Tensor) -> torch.Tensor:
        # Linear applies dot(W^T, x), AXLearn applies dot(W, x).
        if weight_name == "weight":
            return weight.t()
        return weight


class L2Norm(TorchModule):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, eps=self.eps)


class RMSNorm(TorchModule):
    """Torch implementation of RMSNorm <https://github.com/bzhangGo/rmsnorm>"""

    def __init__(self, input_dim: int, eps: float = 1e-8):
        super().__init__()
        self._eps = eps
        self.scale = torch.nn.Parameter(torch.ones((input_dim,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        if x_dtype != torch.float32:
            x = x.type(torch.float32)
        moment2 = (x * x).mean(axis=-1, keepdims=True)
        x = x * torch.rsqrt(moment2 + self._eps)
        x = x * self.scale
        x = x.type(x_dtype)
        return x


def _build_normalization_module(
    norm_type: str, *, input_dim: Optional[int] = None, eps: float = 1e-6
) -> torch.nn.Module:
    """Build normalization module given name and input dimension."""
    norm_types_needing_input_dim = ["layernorm", "rmsnorm"]
    if norm_type in norm_types_needing_input_dim:
        if input_dim is None:
            raise ValueError(
                f"Need to provide input_dim if creating {norm_types_needing_input_dim}"
            )
    if norm_type == "layernorm":
        return LayerNorm(input_dim, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(input_dim, eps=eps)
    elif norm_type == "l2norm":
        return L2Norm(eps=eps)
    else:
        raise NotImplementedError(f"Don't know how to build normalization type: {norm_type}")


# Replication of AXLearn's attention modules:


def _torch_activation_fn(axlearn_activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Convert AXLearn activation to equivalent Torch version."""
    if axlearn_activation == "nn.relu":
        return torch.nn.functional.relu
    if axlearn_activation == "nn.gelu":
        return gelu
    elif axlearn_activation == "nn.silu":
        return torch.nn.functional.silu
    elif axlearn_activation == "linear":
        return lambda x: x
    else:
        raise NotImplementedError(f"Don't have a torch equivalent for {axlearn_activation}.")


class TransformerFeedForwardLayer(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: Union[Callable, tuple[Callable, Callable]] = gelu,
        dropout: float = 0.1,
        structure: str = "prenorm",
        linear_biases: bool = True,
        norm: str = "layernorm",
    ):
        super().__init__()
        if isinstance(activation, tuple):
            assert len(activation) == 2, activation
            for i in range(len(activation)):
                setattr(self, f"linear1_{i}", Linear(input_dim, hidden_dim, bias=linear_biases))
        else:
            self.linear1 = Linear(input_dim, hidden_dim, bias=linear_biases)
        self.linear2 = Linear(hidden_dim, input_dim, bias=linear_biases)
        self.dropout = Dropout(dropout)
        self.norm = _build_normalization_module(norm, input_dim=input_dim)
        self.structure = structure
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.structure == "prenorm":
            x = self.norm(inputs)
            x = self._linear1_activation(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.dropout(x)
            x += inputs
        elif self.structure == "postnorm":
            x = self._linear1_activation(inputs)
            x = self.linear2(x)
            x = self.dropout(x)
            x = self.norm(x + inputs)
        else:
            raise NotImplementedError(self.structure)
        return x

    def _linear1_activation(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.activation, tuple):
            activations = []
            for i, act_fn in enumerate(self.activation):
                activations.append(act_fn(getattr(self, f"linear1_{i}")(x)))
            return activations[0] * activations[1]
        else:
            return self.activation(self.linear1(x))


class MultiheadLinearBase(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(self, model_dim: int, num_heads: int, per_head_dim: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(model_dim, num_heads, per_head_dim) * 0.02)
        self._model_dim = model_dim
        self._num_heads = num_heads
        self._per_head_dim = per_head_dim
        if bias:
            self.bias = self._build_bias()

    def _build_bias(self) -> torch.Tensor:
        raise NotImplementedError(type(self))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.einsum(self._einsum_expr, inputs, self.weight)
        if hasattr(self, "bias"):
            outputs += self.bias
        return outputs


class MultiheadLinearInput(MultiheadLinearBase):
    """See AXLearn's module of the same name for docs."""

    @property
    def _einsum_expr(self):
        return "btd,dnh->btnh"

    def _build_bias(self) -> torch.Tensor:
        return torch.nn.Parameter(torch.zeros(self._num_heads, self._per_head_dim))


class FusedQKVMultiheadLinearInput(TorchModule):
    """Like three MultiheadLinearInputs applied in parallel."""

    def __init__(self, model_dim: int, num_heads: int, per_head_dim: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, model_dim, num_heads, per_head_dim) * 0.02)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(3, num_heads, per_head_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            einsum_expr = "btd,pdnh->pbtnh"
        elif inputs.ndim == 4:
            einsum_expr = "pbtd,pdnh->pbtnh"
        else:
            raise NotImplementedError(f"Don't know how to handle inputs of shape {inputs.shape}")
        outputs = torch.einsum(einsum_expr, inputs, self.weight)
        if self.bias is not None:
            outputs = outputs + self.bias[:, None, None, :, :]
        return outputs


class MultiheadLinearOutput(MultiheadLinearBase):
    """See AXLearn's module of the same name for docs."""

    @property
    def _einsum_expr(self):
        return "btnh,dnh->btd"

    def _build_bias(self) -> torch.Tensor:
        return torch.nn.Parameter(
            torch.zeros(
                self._model_dim,
            )
        )


class _QKVLinearBase(TorchModule):
    """Base module for QKV projection layers."""

    def __init__(
        self,
        query_dim: int,
        *,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        per_head_dim: int,
        linear_biases: bool,
    ):
        del query_dim, key_dim, value_dim, num_heads, per_head_dim, linear_biases
        super().__init__()

    class Output(NamedTuple):
        query: torch.Tensor
        key: torch.Tensor
        value: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> Output:
        raise NotImplementedError(type(self))


class QKVLinear(_QKVLinearBase):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        query_dim: int,
        *,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        per_head_dim: int,
        linear_biases: bool,
    ):
        super().__init__(
            query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            linear_biases=linear_biases,
        )
        self.q_proj = MultiheadLinearInput(
            query_dim, num_heads=num_heads, per_head_dim=per_head_dim, bias=linear_biases
        )
        self.k_proj = MultiheadLinearInput(
            key_dim, num_heads=num_heads, per_head_dim=per_head_dim, bias=linear_biases
        )
        self.v_proj = MultiheadLinearInput(
            value_dim, num_heads=num_heads, per_head_dim=per_head_dim, bias=linear_biases
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> _QKVLinearBase.Output:
        key = query if key is None else key
        value = query if value is None else value
        q_proj = self.q_proj(query)
        k_proj = self.k_proj(key)
        v_proj = self.v_proj(value)
        return self.Output(query=q_proj, key=k_proj, value=v_proj)


class FusedQKVLinear(_QKVLinearBase):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        query_dim: int,
        *,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        per_head_dim: int,
        linear_biases: bool,
    ):
        assert query_dim == key_dim == value_dim, "Only supports shared query/key/value dimensions."
        super().__init__(
            query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            per_head_dim=per_head_dim,
            linear_biases=linear_biases,
        )
        self.qkv_proj = FusedQKVMultiheadLinearInput(
            query_dim, num_heads=num_heads, per_head_dim=per_head_dim, bias=linear_biases
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> _QKVLinearBase.Output:
        if key is None and value is None:
            inputs = query
        elif key is not None and value is not None:
            inputs = torch.stack([query, key, value], axis=0)
        else:
            raise ValueError("Key and value should be both None or both set.")
        q_proj, k_proj, v_proj = self.qkv_proj(inputs)
        return self.Output(query=q_proj, key=k_proj, value=v_proj)


def apply_attention_logit_biases(
    logits: torch.Tensor, *, attention_logit_biases: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Applies `attention_logit_biases` on `logits`.

    Args:
        logits: a float Tensor.
        attention_logit_biases: a float Tensor. If None, assume all zeros.

    Returns:
        logits + attention_logit_biases, in logits.dtype.
    """

    if attention_logit_biases is None:
        return logits
    return logits + attention_logit_biases.to(dtype=logits.dtype, device=logits.device)


def softmax_with_biases(
    logits: torch.Tensor, attention_logit_biases: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Computes softmax with optional masking.

    Args:
        logits: a torch.Tensor of any shape.
        attention_logit_biases: a mask torch.Tensor that is broadcastable with logits.

    Returns:
        A torch.Tensor of same shape and dtype as logits.
    """
    logits = apply_attention_logit_biases(logits, attention_logit_biases=attention_logit_biases)
    logits_dtype = logits.dtype
    if logits_dtype in (torch.float16, torch.bfloat16):
        # Avoid computing softmax in 16-bit floats.
        logits = logits.to(torch.float32)
    probs = torch.softmax(logits, axis=-1)
    if probs.dtype != logits_dtype:
        probs = probs.to(logits_dtype)
    return probs


def alibi_get_slopes(num_heads: int) -> list:
    """Get the slopes for different attention heads defined in ALiBi paper.

    This is a direct copy from ALiBi codebase.
    <https://github.com/ofirpress/attention_with_linear_biases/tree/3b7c2eca/fairseq/models/transformer.py#L742-L752>

    Args:
        num_heads: The number of attention heads.

    Returns:
        A tensor of slopes with shape of [num_heads]. Each value represents
        a slope for one attention head.
    """

    def get_slopes_power_of_2(n: int) -> list:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + alibi_get_slopes(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
        )


class MultiheadAttention(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        qkv_linear_cls: type[_QKVLinearBase] = QKVLinear,
        linear_biases: bool = True,
    ):
        super().__init__()
        self._per_head_dim = (hidden_dim or query_dim) // num_heads
        self._num_heads = num_heads
        self.i_proj: TorchModule = qkv_linear_cls(
            query_dim,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            per_head_dim=self._per_head_dim,
            linear_biases=linear_biases,
        )
        self.o_proj = MultiheadLinearOutput(
            (output_dim or query_dim),
            num_heads=num_heads,
            per_head_dim=self._per_head_dim,
            bias=linear_biases,
        )
        self.dropout = Dropout(dropout)

    class Output(NamedTuple):
        data: torch.Tensor
        probs: torch.Tensor

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> Output:
        q_proj, k_proj, v_proj = self.i_proj(query, key=key, value=value)
        q_proj = self._scale_query(q_proj)
        logits = torch.einsum("btnh,bsnh->bnts", q_proj, k_proj)
        if attention_logit_biases is not None and attention_logit_biases.ndim == 3:
            attention_logit_biases = attention_logit_biases[:, None, :, :]
        probs = softmax_with_biases(logits, attention_logit_biases=attention_logit_biases)
        probs = self.dropout(probs)
        context = torch.einsum("bnts,bsnh->btnh", probs, v_proj).to(v_proj.dtype)
        outputs = self.o_proj(context)
        return self.Output(data=outputs, probs=probs)

    @dataclass
    class CacheState:
        # Decoding step.
        step: torch.Tensor
        # Cached key state.
        key: torch.Tensor
        # Cached value state.
        value: torch.Tensor

    def init_state(self, *, target_batch_size: int, target_max_len: int) -> CacheState:
        """Cached state for autoregressive decoding."""
        dtype = self.o_proj.weight.dtype
        device = self.o_proj.weight.device
        cache = self.CacheState(
            step=torch.as_tensor(0, dtype=torch.int32, device=device),
            key=torch.zeros(
                (target_batch_size, target_max_len, self._num_heads, self._per_head_dim),
                dtype=dtype,
                device=device,
            ),
            value=torch.zeros(
                (target_batch_size, target_max_len, self._num_heads, self._per_head_dim),
                dtype=dtype,
                device=device,
            ),
        )
        return cache

    def extend(
        self,
        cached_state: CacheState,
        query: torch.Tensor,
        *,
        attention_logit_biases: torch.Tensor,
    ) -> tuple[CacheState, Output]:
        """For autoregressive decoding."""
        # Each has shape [B, query_len, N, H].
        q_proj, k_proj, v_proj = self.i_proj(query, key=query, value=query)
        q_proj = self._scale_query(q_proj)
        query_len = q_proj.shape[1]
        cached_state.key[:, cached_state.step : cached_state.step + query_len, ...] = k_proj
        cached_state.value[:, cached_state.step : cached_state.step + query_len, ...] = v_proj
        cached_state.step = cached_state.step + query_len
        logits = torch.einsum(
            "btnh,bsnh->bnts", q_proj, cached_state.key[:, : cached_state.step, ...]
        )
        if attention_logit_biases is not None and attention_logit_biases.ndim == 3:
            attention_logit_biases = attention_logit_biases[:, None, :, :]
        probs = softmax_with_biases(logits, attention_logit_biases=attention_logit_biases)
        probs = self.dropout(probs)
        context = torch.einsum(
            "bnts,bsnh->btnh", probs, cached_state.value[:, : cached_state.step, ...]
        ).to(v_proj.dtype)
        outputs = self.o_proj(context)
        return cached_state, self.Output(data=outputs, probs=probs)

    def _scale_query(self, q_proj: torch.Tensor) -> torch.Tensor:
        return q_proj * (self._per_head_dim**-0.5)


class TransformerAttentionLayer(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        target_dim: int,
        source_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        structure: str = "prenorm",
        norm: str = "layernorm",
        qkv_linear_cls: type[_QKVLinearBase] = QKVLinear,
        linear_biases: bool = True,
    ):
        super().__init__()
        self.norm = _build_normalization_module(norm, input_dim=target_dim)
        self.attention = MultiheadAttention(
            query_dim=target_dim,
            key_dim=source_dim,
            value_dim=source_dim,
            output_dim=target_dim,
            num_heads=num_heads,
            dropout=dropout,
            qkv_linear_cls=qkv_linear_cls,
            linear_biases=linear_biases,
        )
        self.dropout = Dropout(dropout)
        self.structure = structure

    class Output(NamedTuple):
        data: torch.Tensor
        probs: torch.Tensor

    def forward(
        self,
        *,
        target: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        attention_logit_biases: Optional[torch.Tensor] = None,
    ):
        if self.structure == "prenorm":
            skip_input = target
            norm_target = self.norm(target)
            atten_output = self.attention(
                query=norm_target,
                key=source,
                value=source,
                attention_logit_biases=attention_logit_biases,
            )
            data = skip_input + self.dropout(atten_output.data)
        elif self.structure == "postnorm":
            atten_output = self.attention(
                query=target,
                key=source,
                value=source,
                attention_logit_biases=attention_logit_biases,
            )
            data = self.norm(target + self.dropout(atten_output.data))
        else:
            raise NotImplementedError(self.structure)
        return self.Output(data=data, probs=atten_output.probs)

    @dataclass
    class CacheState:
        # Multihead attention state.
        attention: MultiheadAttention.CacheState

    def init_state(self, *, target_batch_size: int, target_max_len: int) -> CacheState:
        """Cached state for autoregressive decoding."""
        return self.CacheState(
            attention=self.attention.init_state(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def extend(
        self,
        cached_state: CacheState,
        target: torch.Tensor,
        *,
        attention_logit_biases: torch.Tensor,
    ) -> tuple[CacheState, Output]:
        """For autoregressive decoding."""
        if self.structure == "prenorm":
            skip_input = target
            norm_target = self.norm(target)
            updated_attention_state, attention_output = self.attention.extend(
                cached_state=cached_state.attention,
                query=norm_target,
                attention_logit_biases=attention_logit_biases,
            )
            data = skip_input + self.dropout(attention_output.data)
        elif self.structure == "postnorm":
            updated_attention_state, attention_output = self.attention.extend(
                cached_state=cached_state.attention,
                query=target,
                attention_logit_biases=attention_logit_biases,
            )
            data = self.norm(target + self.dropout(attention_output.data))
        else:
            raise NotImplementedError(self.structure)
        cached_state.attention = updated_attention_state
        return cached_state, self.Output(data=data, probs=attention_output.probs)


class BaseTransformerLayer(TorchModule):
    """See AXLearn's module of the same name for docs."""

    class Output(NamedTuple):
        data: torch.Tensor
        self_attention_probs: torch.Tensor
        cross_attention_probs: torch.Tensor

    @dataclass
    class CacheState:
        # Self-attention state.
        self_attention: TransformerAttentionLayer.CacheState

    def forward(
        self,
        data: torch.Tensor,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> Output:
        raise NotImplementedError

    def init_state(self, *, target_batch_size: int, target_max_len: int) -> Output:
        raise NotImplementedError


class TransformerLayer(BaseTransformerLayer):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        self_attention: TransformerAttentionLayer,
        feed_forward: TransformerFeedForwardLayer,
        cross_attention: Optional[TransformerAttentionLayer] = None,
    ):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        if cross_attention is not None:
            self.cross_attention = cross_attention

    def forward(
        self,
        data: torch.Tensor,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> BaseTransformerLayer.Output:
        self_atten_outputs = self.self_attention(
            target=data, attention_logit_biases=self_attention_logit_biases
        )
        data = self_atten_outputs.data
        if cross_attention_data is not None:
            cross_atten_outputs = self.cross_attention(
                target=data,
                source=cross_attention_data,
                attention_logit_biases=cross_attention_logit_biases,
            )
            data = cross_atten_outputs.data
            cross_attention_probs = cross_atten_outputs.probs
        else:
            cross_attention_probs = None
        data = self.feed_forward(data)
        return BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=self_atten_outputs.probs,
            cross_attention_probs=cross_attention_probs,
        )

    def init_state(
        self, *, target_batch_size: int, target_max_len: int
    ) -> BaseTransformerLayer.CacheState:
        """Cached state for autoregressive decoding."""
        return BaseTransformerLayer.CacheState(
            self.self_attention.init_state(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def extend(
        self,
        cached_state: BaseTransformerLayer.CacheState,
        data: torch.Tensor,
        *,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> tuple[BaseTransformerLayer.CacheState, BaseTransformerLayer.Output]:
        """For autoregressive decoding."""
        updated_self_attention_state, self_attention_outputs = self.self_attention.extend(
            cached_state=cached_state.self_attention,
            target=data,
            attention_logit_biases=self_attention_logit_biases,
        )
        data = self_attention_outputs.data
        if cross_attention_data is not None:
            cross_attention_outputs = self.cross_attention(
                target=data,
                source=cross_attention_data,
                attention_logit_biases=cross_attention_logit_biases,
            )
            data = cross_attention_outputs.data
            cross_attention_probs = cross_attention_outputs.probs
        else:
            cross_attention_probs = None
        data = self.feed_forward(data)
        cached_state.self_attention = updated_self_attention_state
        return cached_state, BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=self_attention_outputs.probs,
            cross_attention_probs=cross_attention_probs,
        )


class BottleNeckAdapterTransformerLayer(BaseTransformerLayer):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        layer: TransformerLayer,
        adapter: TransformerFeedForwardLayer,
    ):
        super().__init__()
        self.layer = layer
        self.adapter = adapter

    def forward(
        self,
        data: torch.Tensor,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> BaseTransformerLayer.Output:
        out = self.layer(
            data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        skip_input = out.data
        data = self.adapter(out.data)
        data += skip_input
        return BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=out.self_attention_probs,
            cross_attention_probs=out.cross_attention_probs,
        )

    def init_state(
        self, *, target_batch_size: int, target_max_len: int
    ) -> BaseTransformerLayer.CacheState:
        """Cached state for autoregressive decoding."""
        return BaseTransformerLayer.CacheState(
            self.layer.self_attention.init_state(
                target_batch_size=target_batch_size, target_max_len=target_max_len
            )
        )

    def extend(
        self,
        cached_state: BaseTransformerLayer.CacheState,
        data: torch.Tensor,
        *,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> tuple[BaseTransformerLayer.CacheState, BaseTransformerLayer.Output]:
        """For autoregressive decoding."""
        cached_state, output = self.layer.extend(
            cached_state=cached_state,
            data=data,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        skip_input = output.data
        data = self.adapter(output.data)
        data += skip_input
        return cached_state, BaseTransformerLayer.Output(
            data=data,
            self_attention_probs=output.self_attention_probs,
            cross_attention_probs=output.cross_attention_probs,
        )


def _to_device(x: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Moves x to the specified device if x is not None."""
    if x is None:
        return x
    return x.to(device)


class StackedTransformerLayer(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        num_layers: int,
        layer: BaseTransformerLayer,
    ):
        # N.B. if training from scratch you'll need to re-initialize
        # the weights for this module first!
        super().__init__()
        for i in range(num_layers):
            setattr(self, f"layer{i}", copy.deepcopy(layer))
        self._num_layers = num_layers

    def forward(
        self,
        data: torch.Tensor,
        *,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> BaseTransformerLayer.Output:
        all_layer_outputs = []
        for i in range(self._num_layers):
            layer = getattr(self, f"layer{i}")
            layer_outputs = layer(
                _to_device(data, layer.device),
                self_attention_logit_biases=_to_device(self_attention_logit_biases, layer.device),
                cross_attention_data=_to_device(cross_attention_data, layer.device),
                cross_attention_logit_biases=_to_device(cross_attention_logit_biases, layer.device),
            )
            all_layer_outputs.append(layer_outputs)
            data = layer_outputs.data
        aux_outputs = {}
        for field in BaseTransformerLayer.Output._fields:
            if field == "data":
                continue
            values = [getattr(output, field) for output in all_layer_outputs]
            if None in values:
                assert all(v is None for v in values), f"{field}: {values}"
                aux_outputs[field] = None
            else:
                aux_outputs[field] = torch.stack(values, axis=0)
        return BaseTransformerLayer.Output(data=data, **aux_outputs)

    @dataclass
    class CacheState:
        # Per-layer state.
        transformer_layers: list[BaseTransformerLayer.CacheState]

    def init_state(self, *args: Any, **kwargs: Any) -> CacheState:
        """Cached state for autoregressive decoding."""
        layer_cached_states = []
        for ix in range(self._num_layers):
            layer = getattr(self, f"layer{ix}")
            layer_cached_states.append(layer.init_state(*args, **kwargs))
        return self.CacheState(transformer_layers=layer_cached_states)

    def extend(
        self,
        cached_state: CacheState,
        data: torch.Tensor,
        *,
        self_attention_logit_biases: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> tuple[CacheState, BaseTransformerLayer.Output]:
        """For autoregressive decoding."""
        all_layer_outputs = []
        for i in range(self._num_layers):
            layer = getattr(self, f"layer{i}")
            device = layer.device
            layer_cache_state, layer_outputs = layer.extend(
                cached_state.transformer_layers[i],
                _to_device(data, device),
                self_attention_logit_biases=_to_device(self_attention_logit_biases, device),
                cross_attention_data=_to_device(cross_attention_data, device),
                cross_attention_logit_biases=_to_device(cross_attention_logit_biases, device),
            )
            cached_state.transformer_layers[i] = layer_cache_state
            all_layer_outputs.append(layer_outputs)
            data = layer_outputs.data
        aux_outputs = {}
        for field in BaseTransformerLayer.Output._fields:
            if field == "data":
                continue
            values = [getattr(output, field) for output in all_layer_outputs]
            if None in values:
                assert all(v is None for v in values), f"{field}: {values}"
                aux_outputs[field] = None
            else:
                values = [value.to(values[0].device) for value in values]
                aux_outputs[field] = torch.stack(values, axis=0)
        return cached_state, BaseTransformerLayer.Output(data=data, **aux_outputs)

    def to_pipeline(self, devices: Sequence[torch.device]):
        """Pipelines the model layers over the provided devices."""
        layers_per_device = math.ceil(self._num_layers / len(devices))
        for layer_ix in range(self._num_layers):
            layer = getattr(self, f"layer{layer_ix}")
            device_ix = layer_ix // layers_per_device
            layer.to(devices[device_ix])


# Replication of AXLearn's vision transformer modules:


class ConvertToSequence(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        patch_size: tuple[int, int] = (16, 16),
        stride: Optional[tuple[int, int]] = None,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=patch_size,
            stride=stride if stride is not None else patch_size,
            bias=conv_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).permute(0, 2, 3, 1)
        batch, height, width, output_dim = x.shape
        return x.reshape(batch, height * width, output_dim)


class VisualEmbedding(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        convert_to_sequence: ConvertToSequence,
    ):
        super().__init__()
        self.convert_to_sequence = convert_to_sequence

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.convert_to_sequence(image)
        return x


class Encoder(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        input_dim: int,
        pos_emb: Embedding,
        transformer: StackedTransformerLayer,
        input_dropout: float = 0.1,
        use_input_norm: bool = False,
        norm: str = "layernorm",
    ):
        super().__init__()
        self.input_dropout = Dropout(input_dropout)
        self.pos_emb = pos_emb
        self.transformer = transformer
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.input_norm = _build_normalization_module(norm, input_dim=input_dim)
        self.output_norm = _build_normalization_module(norm, input_dim=input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs + self.pos_emb.weight.unsqueeze(0)[:, : inputs.shape[1], :]
        x = self.input_dropout(x)
        if self.use_input_norm:
            x = self.input_norm(x)
        x = self.transformer(x).data
        x = self.output_norm(x)
        return x


class AveragePooling(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(
        self, tokens: torch.Tensor, paddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward function of AveragePooling.

        Args:
            tokens: A torch.Tensor with shape [batch_size, num_len, dim].
            paddings: A torch.Tensor with shape [batch_size, num_len].
                See ``On paddings`` in the file comments of poolings.py.

        Returns:
            An Average Pooling torch.Tensor with shape [batch_size, 1, dim].
        """
        if paddings is None:
            paddings = torch.zeros((tokens.shape[0], tokens.shape[1])).to(tokens.device)
        input_masks = 1 - paddings
        input_masks = input_masks.unsqueeze(axis=-1)
        embeddings_sum = torch.sum(tokens * input_masks, axis=1, keepdims=True)
        masks_sum = input_masks.sum(axis=1, keepdims=True) + self.eps
        pooled_embeddings = embeddings_sum / masks_sum
        return pooled_embeddings


class FirstNTokenPooling(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(self, num_outputs: int = 1):
        super().__init__()
        self.num_outputs = num_outputs

    def forward(
        self, tokens: torch.Tensor, paddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n = self.num_outputs
        if paddings is not None:
            input_masks = 1 - paddings
            if not torch.all(input_masks[:, :n].to(torch.bool)):
                raise ValueError("Input mask should not mask the first N tokens.")
        return tokens[:, :n, :]


class AttentionPooling(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_outputs: int,
        num_heads: int = 1,  # AXLearn AttentionPooling has num_heads=1 by default
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_outputs = num_outputs

        self.cross_attention = TransformerAttentionLayer(
            target_dim=output_dim,
            source_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = TransformerFeedForwardLayer(
            input_dim=output_dim,
            hidden_dim=4 * output_dim,
            dropout=dropout,
            activation=nn.functional.relu,
        )

        self.query_weight = nn.Parameter(torch.randn(num_outputs, output_dim))

    def forward(
        self, tokens: torch.Tensor, paddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        targets = torch.tile(torch.unsqueeze(self.query_weight, 0), (tokens.shape[0], 1, 1))
        if paddings is None:
            paddings = torch.zeros(
                size=(tokens.shape[0], tokens.shape[1]), dtype=tokens.dtype, device=tokens.device
            )  # [batch_size, seq_len]
        source_masks = 1 - paddings
        target_masks = torch.ones(
            size=(tokens.shape[0], self.num_outputs), dtype=tokens.dtype, device=tokens.device
        )  # [batch_size, num_outputs]
        target_masks = torch.unsqueeze(target_masks, -1)  # [batch_size, num_outputs, 1]
        source_masks = torch.unsqueeze(source_masks, -2)  # [batch_size, 1, seq_len]
        masks = ((source_masks != target_masks) * NEG_INF)[:, None, ...]

        targets = self.cross_attention(
            target=targets, source=tokens, attention_logit_biases=masks
        ).data
        return self.feed_forward(targets)


class VisionTransformer(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        visual_embed: VisualEmbedding,
        encoder_1d: Encoder,
        pooler: FirstNTokenPooling,
        num_cls_tokens: int,
        input_dim: int,
    ):
        super().__init__()
        self.visual_embed = visual_embed
        self.encoder_1d = encoder_1d
        self.pooler = pooler
        self.num_cls_tokens = num_cls_tokens
        if self.num_cls_tokens:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, input_dim))

    def get_encoded_features(self, image: torch.Tensor) -> torch.Tensor:
        x = self.visual_embed(image)
        if self.num_cls_tokens:
            x = torch.concat(
                [self.cls_token.tile((x.shape[0], 1, 1)), x],
                dim=1,
            )
        x = self.encoder_1d(x)
        return x

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.get_encoded_features(image)
        x = self.pooler(x)
        return x


class ViTModel(TorchModule):
    """See AXLearn's vision transformer Model for docs."""

    def __init__(
        self,
        backbone: VisionTransformer,
        classifier: Linear,
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        x = self.backbone(image)
        x = torch.squeeze(x, axis=1)
        logits = self.classifier(x)
        return logits

    def forward(
        self, image: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        logits = self.predict(image)
        loss = None
        if label is not None:
            loss = torch.nn.functional.cross_entropy(logits, label.type(torch.int64))
        return loss, {"logits": logits}


class CausalAttentionLogitBiasLayer(TorchModule):
    """See AXLearn's layer for docs."""

    def forward(
        self,
        *,
        segment_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        segment_mask = segment_ids[:, None, :, None] != segment_ids[:, None, None, :]
        causal_mask = positions[:, None, :, None] < positions[:, None, None, :]
        return torch.logical_or(segment_mask, causal_mask) * NEG_INF


class ALiBiAttentionLogitBiasLayer(CausalAttentionLogitBiasLayer):
    """See AXLearn's layer for docs."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.register_buffer(
            "_slopes", torch.as_tensor(alibi_get_slopes(num_heads)), persistent=False
        )
        self._num_heads = num_heads

    def forward(
        self,
        *,
        segment_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # Lower triangular matrix.
        alibi_mask = positions.unsqueeze(1) - positions.unsqueeze(2)
        # Head dim.
        alibi_mask = alibi_mask.unsqueeze(1)
        # ALiBi slopes.
        alibi_mask = (
            alibi_mask.expand((-1, self._num_heads, -1, -1)) * self._slopes[None, :, None, None]
        )
        # Causal mask.
        return apply_attention_logit_biases(
            super().forward(segment_ids=segment_ids, positions=positions),
            attention_logit_biases=alibi_mask,
        )


class TransformerEmbeddings(TorchModule):
    """See AXLearn's layer for docs."""

    def __init__(
        self,
        token_emb: Embedding,
        *,
        type_emb: Optional[Embedding] = None,
        pos_emb: Optional[Embedding] = None,
        norm: Optional[str] = None,
    ):
        super().__init__()
        self.token_emb = token_emb
        self.type_emb = type_emb
        self.pos_emb = pos_emb
        if norm is not None:
            self.norm = _build_normalization_module(norm, input_dim=self.token_emb.embedding_dim)
        else:
            self.norm = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        token_type_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.token_emb(input_ids)
        if self.type_emb is not None and token_type_ids is not None:
            x = x + self.type_emb(token_type_ids)
        if self.pos_emb is not None:
            if positions is None:
                positions = torch.arange(
                    input_ids.shape[1], dtype=torch.int32, device=input_ids.device
                )[None, :]
            x = x + self.pos_emb(positions)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits with token embedding."""
        return torch.einsum("btd,nd->btn", x, self.token_emb.weight)


def _segment_ids_from_causal_input_ids(input_ids: torch.Tensor, pad_token_id: int = 0):
    """See AXLearn's implementation of the same function."""
    non_pad_indicator = (input_ids != pad_token_id).to(input_ids.dtype)
    non_pad_count = torch.sum(
        torch.cummax(torch.flip(non_pad_indicator, dims=[1]), axis=-1).values, axis=-1
    )
    return (torch.arange(input_ids.shape[1], device=input_ids.device) < non_pad_count[:, None]).to(
        input_ids.dtype
    )


class Decoder(TorchModule):
    """See AXLearn's decoder implementation for docs."""

    def __init__(
        self,
        attention_mask: CausalAttentionLogitBiasLayer,
        emb: TransformerEmbeddings,
        transformer: StackedTransformerLayer,
        output_norm: Optional[str] = "layernorm",
    ):
        super().__init__()
        self.attention_mask = attention_mask
        self.emb = emb
        self.transformer = transformer
        self.output_norm = (
            None
            if output_norm is None
            else _build_normalization_module(
                output_norm, input_dim=self.emb.token_emb.embedding_dim
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if positions is None:
            positions = torch.arange(
                input_ids.shape[-1], dtype=torch.int32, device=input_ids.device
            )[None, :]
        x = self.emb(input_ids, token_type_ids=token_type_ids, positions=positions)
        # [batch_size, num_heads, seq_len, seq_len].
        self_attention_logit_biases = self.attention_mask(
            segment_ids=_segment_ids_from_causal_input_ids(input_ids), positions=positions
        )
        x = self.transformer(
            x,
            self_attention_logit_biases=self_attention_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        ).data
        if self.output_norm is not None:
            x = self.output_norm(x)
        # Embeddings are the LM head.
        logits = self.emb.attend(x)
        return dict(logits=logits, hidden_states=x)

    @dataclass
    class CacheState:
        # Decoding step.
        step: torch.Tensor
        # Input token IDs.
        input_ids: torch.Tensor
        # Transformer state.
        transformer: StackedTransformerLayer.CacheState

    def init_state(self, *, batch_size: int, max_sequence_length: int) -> CacheState:
        """Cached state for autoregressive decoding."""
        device = self.emb.token_emb.weight.device
        return self.CacheState(
            step=torch.as_tensor(0, dtype=torch.int32, device=device),
            input_ids=torch.zeros(
                (batch_size, max_sequence_length), dtype=torch.int32, device=device
            ),
            transformer=self.transformer.init_state(
                target_batch_size=batch_size, target_max_len=max_sequence_length
            ),
        )

    def extend(
        self,
        *,
        cached_state: CacheState,
        input_ids: torch.Tensor,
        cross_attention_data: Optional[torch.Tensor] = None,
        cross_attention_logit_biases: Optional[torch.Tensor] = None,
    ) -> tuple[CacheState, dict[str, torch.Tensor]]:
        """For autoregressive decoding."""
        input_ids = input_ids.to(self.device)
        step = cached_state.step
        end_step = step + input_ids.shape[1]
        positions = torch.arange(step, end_step, dtype=torch.int32, device=input_ids.device)[
            None, :
        ].tile(input_ids.shape[0], 1)
        # [batch, len(input_ids), hidden_dim]
        x = self.emb(input_ids, positions=positions)
        cached_state.input_ids[:, step:end_step] = input_ids
        full_mask_logit_biases = self.attention_mask(
            segment_ids=torch.ones_like(cached_state.input_ids),
            positions=torch.arange(cached_state.input_ids.shape[-1], device=input_ids.device)[
                None, :
            ],
        )
        if full_mask_logit_biases.ndim == 3:
            # Add a dimension for num_heads if missing.
            full_mask_logit_biases = full_mask_logit_biases[:, None, ...]
        # [batch, num_heads, len(input_ids), len(cached_input_ids)]
        mask_logit_biases = full_mask_logit_biases[:, :, step:end_step, :end_step]
        updated_transformer_state, transformer_data = self.transformer.extend(
            cached_state=cached_state.transformer,
            data=x,
            self_attention_logit_biases=mask_logit_biases,
            cross_attention_data=cross_attention_data,
            cross_attention_logit_biases=cross_attention_logit_biases,
        )
        cached_state.step = end_step
        cached_state.transformer = updated_transformer_state
        x = transformer_data.data
        if self.output_norm is not None:
            x = self.output_norm(x.to(self.output_norm.device))
        # Embeddings are the LM head.
        logits = self.emb.attend(x.to(self.emb.device))
        return cached_state, dict(logits=logits, hidden_states=x)

    def to_pipeline(self, devices: Sequence[torch.device]):
        """Pipelines the model layers over the provided devices."""
        self.attention_mask.to(devices[0])
        self.emb.to(devices[0])
        self.output_norm.to(devices[0])
        self.transformer.to_pipeline(devices)


# Model builders.


class ViTModelBuilder:
    """Builds ViTModel.

    See AXLearn's vision transformer named models for more details.
    """

    configs = {
        "Test16": dict(num_layers=1, model_dim=8, num_heads=4),
        "Ti16": dict(num_layers=12, model_dim=192, num_heads=3),
        "S16": dict(num_layers=12, model_dim=384, num_heads=6),
        "B16": dict(num_layers=12, model_dim=768, num_heads=12),
        "B32": dict(num_layers=12, model_dim=768, num_heads=12, patch_size=(32, 32)),
        "L14": dict(num_layers=24, model_dim=1024, num_heads=16, patch_size=(14, 14)),
        "L16": dict(num_layers=24, model_dim=1024, num_heads=16),
        "L32": dict(num_layers=24, model_dim=1024, num_heads=16, patch_size=(32, 32)),
        "H14": dict(
            num_layers=32,
            model_dim=1280,
            num_heads=16,
            patch_size=(14, 14),
            global_feature_extraction="gap",
        ),
        # Table 2 of https://arxiv.org/pdf/2106.04560.pdf.
        "g14-paper": dict(
            num_layers=40,
            model_dim=1408,
            num_heads=16,
            feed_forward_dim=6144,
            patch_size=(14, 14),
            global_feature_extraction="gap",
        ),
        "g14-clip": dict(
            num_layers=40,
            model_dim=1536,
            num_heads=16,
            patch_size=(14, 14),
        ),
        "G14": dict(
            num_layers=48,
            model_dim=1664,
            num_heads=16,
            feed_forward_dim=8192,
            patch_size=(14, 14),
            global_feature_extraction="gap",
            dropout_rate=0.0,
        ),
    }

    @classmethod
    def from_name(cls, config_name: str, **kwargs) -> ViTModel:
        """Build a named ViT Model.

        Args:
            config_name: Should map to something in cls.configs.
            kwargs: Keyword arguments that will override arguments from the
                config name. See ViTModelBuilder.from_args.

        Returns:
            Initialized ViTModel.

        Raises:
            ValueError: If config_name is unsupported.
        """
        if config_name not in cls.configs:
            raise ValueError(f"Config must be one of {cls.configs.keys()}")
        cfg = cls.configs[config_name].copy()
        cfg.update(kwargs)
        return cls.from_args(**cfg)

    @classmethod
    def from_args(
        cls,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        patch_size: tuple[int, int] = (16, 16),
        image_size: tuple[int, int] = (224, 224),
        feed_forward_dim: Optional[int] = None,
        stride: Optional[tuple[int, int]] = None,
        num_classes: int = 1000,
        global_feature_extraction: str = "cls_token",
        dropout_rate: float = 0.0,
        conv_bias: bool = True,
        encoder_1d_use_input_norm: bool = False,
    ) -> ViTModel:
        """Build a named ViT Model.

        Args:
            num_layers: The number of layers in the stack config.
            model_dim: The model dimension.
            num_heads: Number of attention heads.
            patch_size: The size for each patch in the image.
            image_size: The size of the input images.
            feed_forward_dim: The hidden dimension of the feed forward layers.
                If None, default to 4 * model_dim.
            stride: The stride for the convert-to-sequence convolutional model.
                If None, uses the patch size.
            num_classes: Number of classifier head classes.
            global_feature_extraction: One of 'cls_token' or 'gap'.
            conv_bias: Should the conv layer used in the encoder have bias.
            dropout_rate: The dropout rate.
            encoder_1d_use_input_norm: Should the visual encoder pass inputs
                through layernorm before processing

        Returns:
            Initialized ViTModel.
        """
        if stride is None:
            stride = patch_size
        seq_len = int(
            np.prod([(i - p) // s + 1 for i, p, s in zip(image_size, patch_size, stride)])
        )
        num_cls_tokens = 0
        pooler = AveragePooling()
        if global_feature_extraction == "cls_token":
            seq_len += 1
            num_cls_tokens = 1
            pooler = FirstNTokenPooling(num_outputs=1)
        # Build transformer.
        structure = "prenorm"
        dropout = dropout_rate
        transformer_layer = TransformerLayer(
            self_attention=TransformerAttentionLayer(
                target_dim=model_dim,
                source_dim=model_dim,
                num_heads=num_heads,
                structure=structure,
                dropout=dropout,
            ),
            feed_forward=TransformerFeedForwardLayer(
                input_dim=model_dim,
                hidden_dim=feed_forward_dim or 4 * model_dim,
                structure=structure,
                dropout=dropout,
            ),
            cross_attention=None,
        )
        encoder_1d = Encoder(
            input_dim=model_dim,
            pos_emb=Embedding(seq_len, model_dim),
            transformer=StackedTransformerLayer(
                num_layers=num_layers,
                layer=transformer_layer,
            ),
            use_input_norm=encoder_1d_use_input_norm,
            input_dropout=dropout,
        )
        visual_embed = VisualEmbedding(
            convert_to_sequence=ConvertToSequence(
                input_dim=3, output_dim=model_dim, conv_bias=conv_bias
            ),
        )
        backbone = VisionTransformer(
            input_dim=model_dim,
            num_cls_tokens=num_cls_tokens,
            visual_embed=visual_embed,
            encoder_1d=encoder_1d,
            pooler=pooler,
        )
        model = ViTModel(
            backbone=backbone,
            classifier=Linear(model_dim, num_classes),
        )
        return model


# Replication of AXLearn's "Causal LM".


class CausalLM(TorchModule):
    """See AXLearn's causal LM implementation."""

    def __init__(self, decoder: Decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        target_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Produce decoder-only loss and predictions including logits and decoder hidden states in
        auxiliary outputs.

        Args:
            input_ids: an int Tensor of shape [batch_size, seq_len].
                Used as decoder input ids. Values should be in the range [0, vocab_size].
            token_type_ids: an optional int Tensor of shape [batch_size, seq_len].
                Values should be in the range [0, type_vocab_size].
            target_labels: an optional int Tensor of shape [batch_size, seq_len].
                Used as decoder input ids. Values should be in the range [0, vocab_size].

        Returns:
            loss: a float Tensor of shape [batch_size].
            predictions (a dict):
                logits: a float Tensor of shape [batch_size, seq_len, vocab_size].
                hidden_states: a float Tensor of shape [batch_size, seq_len, hidden_dim].
        """
        predictions = self._predict(input_ids, token_type_ids=token_type_ids)
        loss = None
        if target_labels is not None:
            logits = predictions["logits"]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_labels.type(torch.int64).view(-1),
            )
        return loss, predictions

    def extract_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Obtains logits from the language model.

        Args:
            input_ids: an int Tensor of shape [batch_size, seq_len].
                Used as decoder input ids. Values should be in the range [0, vocab_size].

        Returns:
           A float Tensor of shape [batch_size, target_len, hidden_dim] representing logits.
        """
        predictions = self._predict(input_ids)
        return predictions["logits"]

    def _predict(
        self, input_ids: torch.Tensor, *, token_type_ids: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Produce decoder logits and hidden states.

        Args:
            input_ids: an int Tensor of shape [batch_size, seq_len].
                Used as decoder input ids. Values should be in the range [0, vocab_size].
            token_type_ids: an optional int Tensor of shape [batch_size, seq_len].
                Values should be in the range [0, type_vocab_size].
        Returns:
            A dict containing:
                logits: a float Tensor of shape [batch_size, seq_len, vocab_size]
                hidden_states: a float Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        # Decoder hidden states: [batch_size, target_len, hidden_dim].
        decoder_output = self.decoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        return decoder_output

    @dataclass
    class CacheState:
        # Decoder cache state.
        decoder: Decoder.CacheState

    def init_state(self, *, batch_size: int, max_sequence_length: int) -> CacheState:
        """Cached state for autoregressive decoding.

        Args:
            batch_size: The batch size to run autoregressive decoding for.
            max_sequence_length: The maximum sequence length to be considered.

        Returns:
            An object that will hold cache state for autoregressive decoding.
        """
        return self.CacheState(
            decoder=self.decoder.init_state(
                batch_size=batch_size, max_sequence_length=max_sequence_length
            )
        )

    def extend(
        self,
        *,
        cached_state: CacheState,
        input_ids: torch.Tensor,
    ) -> tuple[CacheState, dict[str, torch.Tensor]]:
        """Extend decoding autoregressively.

        Args:
            cached_state: The cached autoregressive decoding state.
            input_ids: The new input IDs.

        Returns:
            A tuple of:
                - The cached state (it will have been updated in place).
                - A dictionary of decoder output..
        """
        decoder_cached_state, decoder_outputs = self.decoder.extend(
            cached_state=cached_state.decoder, input_ids=input_ids
        )
        cached_state.decoder = decoder_cached_state
        return cached_state, decoder_outputs

    def to_pipeline(self, devices: Sequence[torch.device]):
        """Pipelines the model layers over the provided devices."""
        self.decoder.to_pipeline(devices)


# Utilities for decoding with CausalLM.


class LogitsModifier(Protocol):
    """Logits-to-logits function."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Accepts new input token IDs, returning the model-generated continuation."""


class TopP(LogitsModifier):
    """Keep only the top-p fraction of probability mass.

    N.B. Does not handle ties correctly.
    TODO(tom_gunter): Efficient way to handle ties in PyTorch.
    """

    def __init__(self, top_p: float = 0.95):
        self._top_p = top_p

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self._top_p
        # Keep first token above threshold.
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # Set the rest to min value.
        sorted_logits = torch.where(
            sorted_indices_to_remove,
            torch.as_tensor(NEG_INF, dtype=logits.dtype, device=logits.device),
            sorted_logits,
        )
        _, reverse_sorted_indices = torch.sort(sorted_indices, dim=-1, descending=False)
        return torch.gather(sorted_logits, dim=-1, index=reverse_sorted_indices)


class TopK(LogitsModifier):
    """Keep only the top-k ranked logits according to probability mass."""

    def __init__(self, top_k: int = 1):
        self._top_k = top_k

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        assert self._top_k <= logits.shape[-1]
        # Remove logits smaller than the last of the top-k values.
        indices_to_remove = logits < torch.topk(logits, self._top_k, dim=-1)[0][..., -1, None]
        return torch.where(
            indices_to_remove,
            torch.as_tensor(NEG_INF, dtype=logits.dtype, device=logits.device),
            logits,
        )


class ScaleBy(LogitsModifier):
    """Scale the logits by a temperature value."""

    def __init__(self, temperature: float = 1.0):
        assert temperature > 1e-4, "Temperature too small (for numerical stability)."
        self._temperature = temperature

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self._temperature


class Chain(LogitsModifier):
    """Chain multiple logits-modifiers in sequence."""

    def __init__(self, *args):
        self._modifiers = args

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        for modifier in self._modifiers:
            logits = modifier(logits)
        return logits


class SampleDecodingSession:
    """Manages state for a sample decoding session."""

    def __init__(
        self,
        model: CausalLM,
        *,
        batch_size: int,
        max_sequence_len: int,
        logits_modifier: Optional[LogitsModifier] = None,
    ):
        """SampleDecodingSession initializer.

        Args:
            model: A decoding model instance.
            batch_size: The batch size for which we will be decoding.
            max_sequence_len: The maximum sequence length we could ever reach.
            logits_modifier: If not None, modifies the vocab logits before sampling.
        """
        self._cache = dict(
            model=model.init_state(batch_size=batch_size, max_sequence_length=max_sequence_len),
            last_generated_id=None,
        )
        self._model = model
        self._logits_modifier = logits_modifier

    def decode(self, num_tokens: int, *, prompt_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate for num_tokens conditioned on prompt ids.

        Args:
            num_tokens: The number of steps to generate for.
            prompt_ids: An input or continuation prompt (i.e. from a user).
                If None will continue generating where the model left off if this is
                not the first time `decode` is called.

        Returns:
            Continuation token IDs.

        Raises:
            ValueError: if prompt_ids is None and no decoding has yet happened.
                The SampleDecodingSession object must be fed a prompt on first decoding.
        """
        assert num_tokens > 0
        last_generated_id = self._cache["last_generated_id"]
        if last_generated_id is None and prompt_ids is None:
            raise ValueError("Cannot decode without a starting prompt.")
        extend_input_ids: torch.Tensor = None
        if last_generated_id is not None:
            # If this isn't the first decoding for this session we prepend the last generated IDs.
            extend_input_ids = last_generated_id
        if prompt_ids is not None:
            extend_input_ids = (
                prompt_ids
                if extend_input_ids is None
                else torch.concat((extend_input_ids, prompt_ids), dim=-1)
            )
        # Buffer to fill with generated outputs.
        output_token_ids = torch.zeros(
            (extend_input_ids.shape[0], num_tokens),
            dtype=torch.int32,
            device=extend_input_ids.device,
        )
        model_cache = self._cache["model"]
        for i in range(num_tokens):
            model_cache, outputs = self._model.extend(
                cached_state=model_cache, input_ids=extend_input_ids
            )
            # Sample next token from logits.
            # Return when there's no output. This happens when decode length exceeds max length.
            if torch.numel(outputs["logits"]) == 0:
                return output_token_ids
            next_token_logits = outputs["logits"][:, -1, :]
            if self._logits_modifier is not None:
                next_token_logits = self._logits_modifier(next_token_logits)
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            extend_input_ids = torch.multinomial(probs, 1)
            output_token_ids[:, i : i + 1] = extend_input_ids
        self._cache["last_generated_id"] = output_token_ids[:, -1:]
        self._cache["model"] = model_cache
        return output_token_ids


class CausalLmModelBuilder:
    """Builds CausalLM model.

    See AXLearn's Causal LM named models for more details.
    """

    configs = {
        "v1-test": dict(num_layers=4, hidden_dim=8, num_heads=4),
        "v1-85m": dict(num_layers=12, hidden_dim=64 * 12, num_heads=12),
        "v1-302m": dict(num_layers=24, hidden_dim=64 * 16, num_heads=16),
        "v1-1_2b": dict(num_layers=24, hidden_dim=64 * 32, num_heads=32),
        "v1-6_4b": dict(num_layers=32, hidden_dim=128 * 32, num_heads=32),
        "v1-12_6b": dict(num_layers=40, hidden_dim=128 * 40, num_heads=40),
        "v1-29_6b": dict(num_layers=48, hidden_dim=128 * 56, num_heads=56),
        "v1-65_2b": dict(num_layers=64, hidden_dim=128 * 72, num_heads=72),
    }

    @classmethod
    def from_name(cls, config_name: str, *, vocab_size: int = 49_152, **kwargs) -> CausalLM:
        """Build a named Causal LM.

        Args:
            config_name: Should map to something in cls.configs.
            vocab_size: The vocabulary size.
            kwargs: Keyword arguments that will override arguments from the
                config name. See CausalLmModelBuilder.from_args.

        Returns:
            Initialized CausalLM.

        Raises:
            ValueError: If config_name is unsupported.
        """
        if config_name not in cls.configs:
            raise ValueError(f"Config name must be one of {cls.configs.keys()}")
        cfg = cls.configs[config_name].copy()
        cfg.update(kwargs)
        return cls.v1_from_args(vocab_size=vocab_size, **cfg)

    @classmethod
    def v1_from_args(
        cls, vocab_size: int, *, num_layers: int, hidden_dim: int, num_heads: int
    ) -> CausalLM:
        """Build a v1 Causal LM.

        Args:
            vocab_size: The vocabulary size.
            num_layers: The number of transformer layers.
            hidden_dim: The model hidden dimension.
            num_heads: THe number of attention heads.

        Returns:
            Initialized model.
        """
        model = CausalLM(
            decoder=Decoder(
                attention_mask=ALiBiAttentionLogitBiasLayer(num_heads),
                emb=TransformerEmbeddings(Embedding(vocab_size, embedding_dim=hidden_dim)),
                transformer=StackedTransformerLayer(
                    num_layers,
                    layer=TransformerLayer(
                        self_attention=TransformerAttentionLayer(
                            target_dim=hidden_dim,
                            source_dim=hidden_dim,
                            num_heads=num_heads,
                            structure="prenorm",
                            norm="rmsnorm",
                            qkv_linear_cls=FusedQKVLinear,
                            linear_biases=False,
                        ),
                        feed_forward=TransformerFeedForwardLayer(
                            input_dim=hidden_dim,
                            hidden_dim=round(hidden_dim * (21.0 / 8.0)),
                            activation=(
                                _torch_activation_fn("nn.silu"),
                                _torch_activation_fn("linear"),
                            ),
                            structure="prenorm",
                            norm="rmsnorm",
                            linear_biases=False,
                        ),
                    ),
                ),
                output_norm="rmsnorm",
            )
        )
        return model


class AdapterType(Enum):
    BOTTLENECK = "bottleneck"


def _next_power_of_two(n: float) -> int:
    """Rounds up n to the next closest power of two.

    If the integer part of n is a power of two, say m, m returned.

    E.g.
     - 127.9 -> 128
     - 128.9 -> 128
     - 129 -> 256
    """
    if n <= 1:
        return 2
    return 1 << int(math.log2(n - 1)) + 1


class AdapterCausalLmModelBuilder(CausalLmModelBuilder):
    """Builds Causal LM model with adapter layers."""

    configs = {
        # Update the configs with adapter type and properties.
        k: {
            **v,
            **dict(
                adapter_type=AdapterType.BOTTLENECK,
                bottleneck_dim=_next_power_of_two(v["hidden_dim"] * 0.5),
            ),
        }
        for k, v in CausalLmModelBuilder.configs.items()
    }

    @classmethod
    def v1_from_args(
        cls,
        vocab_size: int,
        *,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        adapter_type: AdapterType = AdapterType.BOTTLENECK,
        **kwargs,
    ) -> CausalLM:
        """Build a v1 Causal LM with an adapter.

        Args:
            vocab_size: The vocabulary size.
            num_layers: The number of transformer layers.
            hidden_dim: The model hidden dimension.
            num_heads: The number of attention heads.
            adapter_type: The type of the adapter.
            kwargs: kwargs needed to initialise adapters.

        Returns:
            Initialized model.

        Raises:
            ValueError: When the adapter type is not supported.
        """
        # Instantiate the base LM.
        model = super().v1_from_args(
            vocab_size=vocab_size, num_layers=num_layers, hidden_dim=hidden_dim, num_heads=num_heads
        )

        # Extract the base transformer.
        base_transformer = model.decoder.transformer

        # Create the adapted layer.
        if adapter_type == AdapterType.BOTTLENECK:
            assert "bottleneck_dim" in kwargs
            adapter_layer = BottleNeckAdapterTransformerLayer(
                # Use first layer as they are all the same.
                layer=base_transformer.layer0,
                adapter=TransformerFeedForwardLayer(
                    input_dim=hidden_dim,
                    structure="postnorm",
                    norm="rmsnorm",
                    activation=_torch_activation_fn("nn.relu"),
                    hidden_dim=kwargs["bottleneck_dim"],
                    linear_biases=False,
                ),
            )
        else:
            raise ValueError(f"The adapter type {adapter_type.value} is not supported.")

        # Create the adapted transformer.
        adapted_transformer = StackedTransformerLayer(num_layers=num_layers, layer=adapter_layer)

        # Swap the transformer in the base model with the adpted one.
        model.decoder.transformer = adapted_transformer

        return model


# NOTE: AXLearn CoCaImageStreamEncoder inherits from StreamEncoder but that's just an interface
# that enforces we implement a forward(input_batch) fn.
class CoCaImageStreamEncoder(TorchModule):
    """See AXLearn's module of the same name for docs."""

    def __init__(
        self,
        image_encoder: VisionTransformer,
        contrastive_pooler_output_dim: Optional[int] = None,
        # num heads shared between poolers,
        # no effect if using FirstNTokenPooling or AveragePooling.
        pooler_num_heads: Optional[int] = None,
        caption_pooler_output_dim: Optional[int] = None,
        contrastive_pooler_cls: Union[
            type[AttentionPooling], type[FirstNTokenPooling], type[AveragePooling]
        ] = AttentionPooling,
        caption_pooler_cls: Union[
            type[AttentionPooling], type[FirstNTokenPooling], type[AveragePooling]
        ] = AttentionPooling,
        # NOTE: contrastive_pooler will always have only 1 output.
        caption_pooler_num_outputs: int = 256,
        contrastive_output_norm_type: str = "l2norm",
        contrastive_output_proj_dim: Optional[int] = None,
        contrastive_output_proj_bias: bool = True,
        pooler_mode: str = "cascade",
    ):
        super().__init__()
        if pooler_mode not in ["parallel", "cascade", "bottleneck"]:
            raise ValueError(f"Unknown pooler mode '{pooler_mode}'")

        encoder_output_dim = image_encoder.visual_embed.convert_to_sequence.conv.out_channels
        caption_pooler_output_dim = caption_pooler_output_dim or encoder_output_dim

        self.image_encoder = image_encoder
        self.pooler_mode = pooler_mode
        self.contrastive_output_norm = _build_normalization_module(contrastive_output_norm_type)

        # See the AXLearn CocaImageStreamEncoder docstring for how different pooler_modes work.
        # Some repetition exists in the codeblock to easily understand the flow.

        if pooler_mode == "bottleneck":  # Use contrastive_pooler for both outputs.
            self.contrastive_pooler = self._build_pooler_layer(
                contrastive_pooler_cls,
                input_dim=encoder_output_dim,
                output_dim=contrastive_pooler_output_dim or encoder_output_dim,
                num_outputs=1,
                num_heads=pooler_num_heads,
            )
        elif pooler_mode == "cascade":  # Feed output of caption pooler to the contrastive pooler.
            self.caption_pooler = self._build_pooler_layer(
                caption_pooler_cls,
                input_dim=encoder_output_dim,
                output_dim=caption_pooler_output_dim,
                num_outputs=caption_pooler_num_outputs,
                num_heads=pooler_num_heads,
            )
            self.contrastive_pooler = self._build_pooler_layer(
                contrastive_pooler_cls,
                input_dim=caption_pooler_output_dim,
                output_dim=contrastive_pooler_output_dim or encoder_output_dim,
                num_outputs=1,
                num_heads=pooler_num_heads,
            )
        else:  # Parallel mode: both consume output of ViT.
            self.caption_pooler = self._build_pooler_layer(
                caption_pooler_cls,
                input_dim=encoder_output_dim,
                output_dim=caption_pooler_output_dim,
                num_outputs=caption_pooler_num_outputs,
                num_heads=pooler_num_heads,
            )
            self.contrastive_pooler = self._build_pooler_layer(
                contrastive_pooler_cls,
                input_dim=encoder_output_dim,
                output_dim=contrastive_pooler_output_dim or encoder_output_dim,
                num_outputs=1,
                num_heads=pooler_num_heads,
            )

        if contrastive_output_proj_dim:
            if hasattr(self.contrastive_pooler, "output_dim"):
                input_dim = self.contrastive_pooler.output_dim
            else:
                input_dim = encoder_output_dim
            self.contrastive_output_proj = Linear(
                input_dim, contrastive_output_proj_dim, bias=contrastive_output_proj_bias
            )

    @staticmethod
    def _build_pooler_layer(
        pooler_cls: Union[type[AttentionPooling], type[FirstNTokenPooling], type[AveragePooling]],
        input_dim: int,
        output_dim: int,
        num_outputs: int = 1,
        num_heads: int = 1,
    ) -> Union[AttentionPooling, FirstNTokenPooling, AveragePooling]:
        if pooler_cls == AttentionPooling:
            return AttentionPooling(input_dim, output_dim, num_outputs, num_heads)
        elif pooler_cls == FirstNTokenPooling:
            return FirstNTokenPooling(num_outputs)
        elif pooler_cls == AveragePooling:
            return AveragePooling()
        else:
            raise ValueError(f"Unknown pooler class: {pooler_cls}")

    def forward(self, image: torch.Tensor):
        if len(image.shape) != 5:
            raise ValueError(
                "image should be of form: [batch_size, num_images, channel, height, width]"
            )
        batch_size, num_images = image.shape[0], image.shape[1]
        # VisionTransformer accepts only 4D tensor
        image = image.reshape(batch_size * num_images, *image.shape[2:])
        encoded_features = self.image_encoder.get_encoded_features(image)

        if self.pooler_mode == "bottleneck":
            caption_features = contrastive_features = self.contrastive_pooler(encoded_features)
        elif self.pooler_mode == "parallel":
            caption_features = self.caption_pooler(encoded_features)
            contrastive_features = self.contrastive_pooler(encoded_features)
        elif self.pooler_mode == "cascade":
            caption_features = self.caption_pooler(encoded_features)
            contrastive_features = self.contrastive_pooler(caption_features)
        else:
            raise ValueError(f"unsupported pooler mode: {self.pooler_mode}")
        contrastive_features = contrastive_features.squeeze(dim=1)
        if self.contrastive_output_proj:
            contrastive_features = self.contrastive_output_proj(contrastive_features)
        if self.contrastive_output_norm:
            contrastive_features = self.contrastive_output_norm(contrastive_features)

        contrastive_features = contrastive_features.reshape(
            batch_size, num_images, *contrastive_features.shape[1:]
        )

        caption_features = caption_features.reshape(
            batch_size, num_images, *caption_features.shape[1:]
        )
        return contrastive_features, caption_features


class CoCaImageStreamEncoderBuilder:
    """Builds CoCaImageStreamEncoder."""

    configs = {
        "Test16": dict(
            image_encoder_cfg="Test16",
            additional_vit_args=dict(conv_bias=False, encoder_1d_use_input_norm=True),
            pooler_num_heads=4,
            caption_pooler_num_outputs=4,
            contrastive_output_proj_dim=1,
            contrastive_output_proj_bias=False,
            contrastive_pooler_cls=FirstNTokenPooling,
            caption_pooler_cls=AttentionPooling,
            pooler_mode="parallel",
        ),
    }

    @classmethod
    def from_name(cls, config_name: str, **kwargs) -> CoCaImageStreamEncoder:
        if config_name not in cls.configs:
            raise ValueError(f"unknown config: {config_name}")

        cfg = cls.configs[config_name].copy()
        cfg.update(kwargs)

        return cls.from_args(**cfg)

    @classmethod
    def from_args(
        cls,
        image_encoder_cfg: str,
        **kwargs,
    ) -> CoCaImageStreamEncoder:
        image_encoder = ViTModelBuilder.from_name(
            image_encoder_cfg, **kwargs.get("additional_vit_args", {})
        ).backbone
        del kwargs["additional_vit_args"]
        return CoCaImageStreamEncoder(image_encoder=image_encoder, **kwargs)
