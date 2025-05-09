import zlib

import jax
import jax.numpy as jnp

from axlearn.common.attention import (
    TransformerFeedForwardLayer,
    TransformerLayer,
    set_attention_partition_specs,
)
from axlearn.common.config import ConfigBase
from axlearn.common.inference import DataPartitionType, InferenceRunner
from axlearn.common.state_builder import Builder, TensorStoreStateStorageBuilder
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, mesh_shape_from_axes
from axlearn.experiments.text.gpt.fuji import Version, get_trainer_kwargs


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
    """A dummy state builder."""

    def input_state_type(self) -> Builder.StateType:
        return Builder.StateType.TENSOR_SPECS

    def __call__(self, state: Builder.State) -> Builder.State:
        cfg = self.config
        seed = zlib.adler32(cfg.dir.encode("utf-8"))

        key = jax.random.PRNGKey(seed)
        return Builder.State(
            step=0,
            trainer_state=jax.tree.map(
                lambda spec: jax.lax.with_sharding_constraint(
                    jax.random.normal(key=key, shape=spec.shape, dtype=spec.dtype),
                    shardings=spec.sharding,
                )
                if spec.dtype in [jnp.bfloat16, jnp.float32]
                else jnp.zeros(shape=spec.shape, dtype=spec.dtype, device=spec.sharding),
                state.trainer_state,
            ),
            built_keys=set(),
        )


model_cfg = get_trainer_kwargs("test", vocab_size=32000, version=Version.V3, flash_attention=True)[
    "model_cfg"
]
model_cfg = set_inference_partition_spec(model_cfg)
print(model_cfg)
inference_runner_cfg = InferenceRunner.default_config().set(
    mesh_shape=mesh_shape_from_axes(model=2),
    mesh_axis_names=MESH_AXIS_NAMES,
    model=model_cfg,
    inference_dtype=jnp.bfloat16,
    input_batch_partition_spec=DataPartitionType.REPLICATED,
    init_state_builder=DummyBuilder.default_config().set(dir="dummy"),
    name="runner",
)

inference_runner = inference_runner_cfg.instantiate(parent=None)
model = model_cfg.set(name="prefill_model").instantiate(parent=None)
prng_key = jax.random.PRNGKey(0)


def run_once(prng_key, tokens):
    prng_key, subkey = jax.random.split(prng_key, 2)
    r = inference_runner.run(
        input_batches=[{"prefix": tokens}], method="sample_decode", prng_key=subkey
    )
    return list(r)[0], prng_key


input_tokens = jnp.zeros((1, 1024), dtype=jnp.int32)
r, prng_key = run_once(prng_key, input_tokens)
jax.block_until_ready(r)

jax.profiler.start_trace("/tmp/gpt_test/summaries")
r, prng_key = run_once(prng_key, input_tokens)
jax.block_until_ready(r)
jax.profiler.stop_trace()
