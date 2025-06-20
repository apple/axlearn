# Copyright © 2023 Apple Inc.

"""An example to run offline inference for fuji.

Some optimizations are ported from the not-yet open sourced inference engine so the forward pass
performance is comparable to what we have internally.

Some known issues:
1. KVCache update isn't efficient and potentially requires copying the whole KV cache. This could
be solved by either changing the KV cache update method or using PagedKVCache.
"""

import functools
import zlib
from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from axlearn.common.attention import (
    BaseLayer,
    FusedGroupedQKVLinear,
    GroupedQKVLinear,
    Module,
    TransformerFeedForwardLayer,
    TransformerLayer,
    set_attention_partition_specs,
)
from axlearn.common.config import ConfigBase
from axlearn.common.inference import DataPartitionType, InferenceRunner
from axlearn.common.state_builder import Builder, TensorStoreStateStorageBuilder
from axlearn.common.utils import Tensor
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, mesh_shape_from_axes
from axlearn.experiments.text.gpt.fuji import Version, get_trainer_kwargs
from axlearn.common.module import functional as F

# jax.distributed.initialize(coordinator_address="127.0.0.1:1123", num_processes=1, process_id=0)


def set_inference_partition_spec(cfg: ConfigBase) -> ConfigBase:
    """Set inference-friendly model weight and activation sharding."""
    if isinstance(cfg, TransformerLayer.Config):
        raise ValueError("TransformerLayer cannot be the root of the input model config.")

    batch_axis_names = ("data", "expert", "fsdp")
    fsdp_axis_names = "fsdp"
    tp_axis_names = "model"
    seq_axis_names = "seq"

    def enter_fn(_, layer_cfg, default_kv):
        if isinstance(layer_cfg, TransformerLayer.Config):
            set_attention_partition_specs(
                layer_cfg.self_attention.attention,
                fsdp_axis_names=fsdp_axis_names,
                tp_axis_names=tp_axis_names,
            )
            if layer_cfg.cross_attention is not None:
                set_attention_partition_specs(
                    layer_cfg.cross_attention.attention,
                    fsdp_axis_names=fsdp_axis_names,
                    tp_axis_names=tp_axis_names,
                )
            if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
                cfg = layer_cfg.feed_forward
                # Shard weights.
                cfg.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
                cfg.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
                cfg.linear1.output_partition_spec = (
                    batch_axis_names,
                    seq_axis_names,
                    tp_axis_names,
                )
                # Do not shard output. This is to avoid having reduce-scatter + allgather instead of
                # a single allreduce. The latter has lower latency for inference.
                cfg.linear2.output_partition_spec = (batch_axis_names, seq_axis_names, None)
        return default_kv

    cfg.visit(visit_fn=lambda k, v: None, enter_fn=enter_fn)
    return cfg


class DummyBuilder(TensorStoreStateStorageBuilder):
    """A dummy state builder that returns random weights."""

    def input_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSOR_SPECS

    def __call__(self, state: Builder.State) -> Builder.State:
        cfg = self.config
        seed = zlib.adler32(cfg.dir.encode("utf-8"))

        key = jax.random.PRNGKey(seed)
        out_shardings = jax.tree.map(lambda spec: spec.sharding, state.trainer_state)

        @functools.partial(jax.jit, out_shardings=out_shardings)
        def jit_init():
            return jax.tree.map(
                lambda spec: jax.random.normal(key=key, shape=spec.shape, dtype=spec.dtype)
                if spec.dtype in [jnp.bfloat16, jnp.float32]
                else jnp.zeros(shape=spec.shape, dtype=spec.dtype),
                state.trainer_state,
            )

        return Builder.State(
            step=0,
            trainer_state=jit_init(),
            built_keys=set(),
        )


def replace_layer_config_recursively(
    cfg: ConfigBase,
    *,
    target_cls: Module,
    source_config: BaseLayer.Config,
    exclude_keys: Optional[Sequence[str]] = None,
) -> ConfigBase:
    """Replaces the target_cls's config with the source_config.

    This function is useful when one wants to replace a specific layer in the model, e.g.,
    replacing MultiheadAttention layer with GroupedQueryAttention. Note that the target layer
    should not be the root layer in the given model config.

    Example usage -- Replacing MultiheadAttention by GroupedQueryAttention:
        model_cfg = ... # The original model config.
        replace_layer_config_recursively(
            model_cfg,
            target_cls=MultiheadAttention,
            source_config=GroupedQueryAttention.default_config().set(num_kv_heads=8),
            exclude_keys=["num_kv_heads"],
        )

    Args:
        cfg: A ConfigBase, usually a top-level model config or a trainer config.
        target_cls: A Module, the target layer class to be replaced.
        source_config: A new BaseLayer config to be put into the model.
        exclude_keys: A sequence of strings specifying which fields in the source config should
           not be copied from the target config. By default, only klass is excluded.

    Return:
        A ConfigBase with the modified configs. This function also revises the input config in
        place. So it is okay to not return anything.

    Raises:
        ValueError: If the target layer is the root of the input cfg.
    """
    if isinstance(cfg, target_cls.Config):
        raise ValueError("The target cls cannot be the root of the input model config.")

    exclude_kwargs = set(["klass"])
    if exclude_keys is not None:
        exclude_kwargs.update(exclude_keys)

    def enter_fn(_, child, default_kv):
        if isinstance(child, ConfigBase):
            for key, value in child.items():
                if isinstance(value, target_cls.Config):
                    new_cfg = source_config.set(
                        **{k: v for k, v in value.items() if k not in exclude_kwargs},
                    )
                    setattr(child, key, new_cfg)
        return default_kv

    cfg.visit(visit_fn=lambda k, v: None, enter_fn=enter_fn)
    return cfg


# Stop the generation early for profiling purposes.
class LengthStopingCondition:
    def __init__(self, length: int):
        self._length = length

    def __call__(self, *, index: Tensor, sequences: Tensor, out_of_prompt: Tensor) -> Tensor:
        return jnp.broadcast_to((index >= self._length)[:, None], out_of_prompt.shape)


# StackedTransformer is faster when doing inference.
# Note: you can change the number of layers in fuji.py.
model_cfg = get_trainer_kwargs(
    "3B", vocab_size=32000, version=Version.V3, flash_attention=False, use_stacked=True
)["model_cfg"]
# Groupde QKV linear has better sharding support.
model_cfg = replace_layer_config_recursively(
    model_cfg,
    target_cls=FusedGroupedQKVLinear,
    source_config=GroupedQKVLinear.default_config(),
)
model_cfg = set_inference_partition_spec(model_cfg)
model_cfg.decoder.emb.token_emb.param_partition_spec = ("model", ("expert", "fsdp", "seq"))


inference_runner_cfg = InferenceRunner.default_config().set(
    mesh_shape=mesh_shape_from_axes(model=1),
    mesh_axis_names=MESH_AXIS_NAMES,
    model=model_cfg,
    inference_dtype=jnp.float32,
    input_batch_partition_spec=DataPartitionType.REPLICATED,
    init_state_builder=DummyBuilder.default_config().set(dir="dummy"),
    name="runner",
)
print(inference_runner_cfg)

inference_runner = inference_runner_cfg.instantiate(parent=None)
prng_key = jax.random.PRNGKey(0)
stopping_cond = LengthStopingCondition(3)
time_step = jnp.zeros((32,), dtype=jnp.int32)
input_tokens = jnp.zeros((32, 4096), dtype=jnp.int32)

@jax.jit
def jit_forward(state, time_step, input_tokens):
    (init_states, init_outputs), _ = F(
        module=inference_runner.model.decoder,
        prng_key=prng_key,
        state=state,
        inputs={"time_step": time_step, "input_batch": {"input_ids": input_tokens}},
        method="prefill_states",
        is_training=False,
    )
    return init_states, init_outputs


@jax.jit
def jit_decode(state, cached_states, input_tokens):
    (init_states, init_outputs), _ = F(
        module=inference_runner.model.decoder,
        prng_key=prng_key,
        state=state,
        inputs={
            "cached_states": cached_states,
            "input_ids": input_tokens,
        },
        method="extend_step",
        is_training=False,
    )
    return init_states, init_outputs

with inference_runner._mesh:  
    params = inference_runner._inference_runner_state.model["decoder"]
    forward_lowered = jit_forward.lower(params, time_step, input_tokens)
    # forward_compiled = forward_lowered.compile()
    
    (init_states, _) = jit_forward.eval_shape(
        params, time_step, input_tokens
    )
    
    with open("prefill.stablehlo", "w") as f:
        f.write(forward_lowered.as_text("stablehlo", debug_info=True))

    cached_states = {
        "transformer_state": init_states["transformer_state"],
        "time_step": time_step,
        # extend_step needs cached_inputs to compute self_attention_biases.
        "input_ids": input_tokens,
    }

    decode_lowered = jit_decode.lower(params, cached_states, jnp.zeros((32, 1), dtype=jnp.int32))
    with open("decode.stablehlo", "w") as f:
        f.write(decode_lowered.as_text("stablehlo", debug_info=True))
