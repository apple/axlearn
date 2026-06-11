"""Per-token activation gradient monitoring and clipping for stable training.

This module introduces gradient monitoring/clipping knobs between Transformer blocks (FFN and
attention) to detect and suppress outlier per-token gradients that destabilize training.

* Uses `jax.custom_vjp` to define an identity function in the forward pass that
intercepts activation gradients dL/dy in the backward pass. Since dL/dtheta = (dL/dy)(dy/dtheta),
modifying dL/dy effectively modifies parameter gradients without changing the forward pass.

* A stateful `GradientMonitorAndClipLayer` is added as a child of monitored layers
(e.g. `MonitoredTransformerFeedForwardLayer`). It wraps the layer output with `clip_gradient` and
maintains auxiliary state for per-token gradient statistics (norms, percentiles, clip counts).

* Summary logging: Layer context is unavailable during backward passes, so gradient stats
cannot be written to summaries directly. Instead, stats are initialized as layer parameters
and returned from bwd as pseudo-gradients. A special optimizer replacement rule (via
`CompositeLearner`) replaces the stats param with its "gradient" (the real stats), which is then
logged to summaries in the next forward pass.
"""

import functools
from typing import Callable, Literal, Optional, Sequence, Type, Union

import attr
import jax
import numpy as np
from jax import custom_vjp
from jax import numpy as jnp

from axlearn.common import learner, optimizers
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import InstantiableConfig, config_class, config_for_function
from axlearn.common.param_init import constant_initializer
from axlearn.common.summary import CallbackSummary, ImageSummary
from axlearn.common.utils import Nested, Tensor

DEFAULT_PERCENTILES = (50, 99.9, 99.99, 99.999)
GRADIENT_CLIPPING_PATHS = ("", ".data")  # '' for FFN, '.data' for Attention.


def compute_grad_percentile_no_clip_fn(
    stats_arr: Sequence[float] = DEFAULT_PERCENTILES,
) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
    """Returns a gradient clipping function that only computes stats (no clipping).

    Args:
        stats_arr: Percentiles to compute.

    Returns:
        A function mapping gradient tensor to (original gradient, stats).
    """

    def fn(g: Tensor) -> tuple[Tensor, Tensor]:
        per_token_norm = jnp.linalg.norm(g.astype(jnp.float32), axis=-1)
        stats = jnp.percentile(per_token_norm, jnp.array(stats_arr))
        return g, stats

    return fn


def _top_k_clipping(g: Tensor, k: int = 4, method: Literal["approx", "topk", "sort"] = "approx"):
    per_token_norm = jnp.linalg.norm(g.astype(jnp.float32), axis=-1)
    flat_norm = jnp.ravel(per_token_norm)
    assert 1 <= k < len(flat_norm), "k must be >= 1 and < token count."

    if method == "approx":
        topk, ids = jax.lax.approx_max_k(flat_norm, k, recall_target=0.99)
        flat_mask = jnp.ones_like(flat_norm, dtype=bool).at[ids].set(False)
        is_valid = flat_mask.reshape(per_token_norm.shape)
    elif method == "topk":
        topk, ids = jax.lax.top_k(flat_norm, k)
        threshold = topk[-1]
        is_valid = per_token_norm < threshold
    else:  # method == "sort"
        ids = jnp.argsort(-flat_norm)[:k]
        topk = flat_norm[ids]
        threshold = topk[-1]
        is_valid = per_token_norm < threshold

    clipped = jnp.where(jnp.expand_dims(is_valid, -1), g, 0)
    return clipped, topk, ids


def top_k_clip_fn(
    k: int = 4, method: Literal["approx", "topk", "sort"] = "approx", log_pos: bool = False
) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
    """Returns an activation gradient clipping function that zeros out tokens
    whose gradient norm is among the top-k largest in a batch.

    Args:
        k: number of tokens to remove per batch.
        method: algorithm to find the top-k:
            - approx: use ``jax.lax.approx_max_k``
            - topk: use ``jax.lax.top_k``
            - sort: use sorting to find top k
        log_pos: whether to log clipped token positions in stats

    Returns:
        A function mapping gradient tensor to (clipped gradient, top-k norms).
    """

    def fn(g: Tensor) -> tuple[Tensor, Tensor]:
        clipped, topk, pos = _top_k_clipping(g, k, method)
        if log_pos:
            # Convert token positions (ints) to be representable by bf16
            pos = encode_int32_to_bf16_bytes(pos)
            stats = jnp.concatenate([topk[None, :], pos], axis=0)
        else:
            stats = topk
        return clipped, stats

    return fn


def encode_int32_to_bf16_bytes(x: jnp.ndarray) -> jnp.ndarray:
    """
    Encode an int32 array of shape (k,) into a bfloat16 array of shape (4, k),
    using 4 exact base-256 digits (bytes). This is helpful to convert token positions
    (large ints) to be representable by bfloat16, which is the trainer precision.
    The max consecutive int this function can represent is the same as that of
    int32 (2^31−1).
    """
    x = x.astype(jnp.int32)
    # Reinterpret the same 32 bits as unsigned, without changing the bit pattern.
    ux = jax.lax.bitcast_convert_type(x, jnp.uint32)

    d0 = ((ux >> 0) & 0xFF).astype(jnp.bfloat16)
    d1 = ((ux >> 8) & 0xFF).astype(jnp.bfloat16)
    d2 = ((ux >> 16) & 0xFF).astype(jnp.bfloat16)
    d3 = ((ux >> 24) & 0xFF).astype(jnp.bfloat16)
    return jnp.stack([d0, d1, d2, d3], axis=0)


def gradient_clipping_impl(
    g: Union[Tensor, Nested[Tensor]],
    *,
    clip_fn: Callable[[Tensor], tuple[Tensor, Tensor]],
    prev_stats: Optional[Tensor] = None,
    supported_paths: Sequence[str] = GRADIENT_CLIPPING_PATHS,
) -> tuple[Union[Tensor, Nested[Tensor]], Tensor]:
    """Activation gradient clipping and stats computing function.
    Work for single tensors and PyTrees.
    Only FFN and Attention layers monitoring are supported.

    Args:
        g: gradient tensor or nested gradient tensors.
        clip_fn: function mapping a single tensor to (clipped tensor, stats).
        prev_stats: Optional previous statistics. Can be useful for computing new stats.
        supported_paths: Pytree paths in the output tensor to monitor/clip.
            Paths need to be in the format of "[0]", ".name", "[1]['k1']", "[2].name['k2']" etc.
            The `supported_paths` param allows the client to configure which state to monitor,
            within a layer's outputs, making it adaptable to generic output types (PyTrees).
            For example, FFN output is a plain tensor (path: "");
            Attention output looks like {"data": ..., "probs": ..., "kv_state": ...} and
            only the "data" path is clipped/monitored.
    Returns:
        clipped_g: clipped gradient with the same structure as input.
        total_stats: aggregated statistics across all clipped tensors.
    """
    del prev_stats
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(g)
    clipped_leaves = []
    stats_leaves = []
    supported_paths = set(supported_paths)

    for path, leaf in paths_and_leaves:
        if jax.tree_util.keystr(path) in supported_paths:
            assert isinstance(leaf, Tensor), (
                "Monitoring needs to be applied to plain tensors. "
                "By default, monitoring is only supported for FFN and Attn layers. "
                "Pass in appropriate `supported_paths` for arbitrary layer monitoring. "
            )
            clipped, stats = clip_fn(leaf)
        else:
            clipped, stats = leaf, None  # Do not clip
        clipped_leaves.append(clipped)
        stats_leaves.append(stats)

    clipped_g = jax.tree_util.tree_unflatten(treedef, clipped_leaves)
    stats_leaves = [s for s in stats_leaves if s is not None]
    assert len(stats_leaves) < 2, "Only one tensor per layer is supported for clipping."
    total_stats = stats_leaves[0]
    return clipped_g, total_stats


def _default_top_norms_from_stats(stats: Tensor) -> Tensor:
    return stats[0, :] if stats.ndim > 1 else stats


def _default_token_pos_from_stats(stats: Tensor) -> Tensor:
    return stats[1:, :] if stats.ndim > 1 else jnp.array([])


def default_stats_logging_fn(
    log_pos: bool = False,
    top_norms_fn: Callable[[Tensor], Tensor] = _default_top_norms_from_stats,
    token_pos_fn: Callable[[Tensor], Tensor] = _default_token_pos_from_stats,
) -> Callable[["GradientMonitorAndClipLayer", Tensor], None]:
    """Returns the default `stats_log_fn` for `GradientMonitorAndClipLayer`.

    The returned function is called as `fn(layer, stats)` (the layer is bound
    via `functools.partial` in `GradientMonitorAndClipLayer.__init__`).

    Args:
        log_pos: whether to log clipped token positions. `stats` from clip_fn must carry
            token positions data for this to work.
        top_norms_fn: extracts the top-norms tensor from `stats`.
        token_pos_fn: extracts the (encoded) token-position tensor from `stats`.
            Only consulted when `log_pos=True`.
    """

    def log_clipped_token_positions(layer: "GradientMonitorAndClipLayer", token_pos: Tensor):
        """Logs (encoded) clipped token positions to tensorboard. Due to tensorboard
        constraints, the value is logged as a string instead of a tensor."""

        def _bytes_to_string(pos: np.ndarray) -> str:
            """Produces strings like
            255 127 0 42
            255 255 0 0
            127 0 0 0
            0 0 0 0
            """
            return "\n".join(" ".join(str(int(x)) for x in row) for row in pos)

        layer.add_summary("clipped_token_pos", CallbackSummary(_bytes_to_string, token_pos))

    def log_top_grad_norms(layer: "GradientMonitorAndClipLayer", top_norms: Tensor):
        # Difference between `top_norms` and `top_norms_hist`: `add_summary(name, tensor)`
        # compresses `tensor` into a 30-bin histogram, which discards original values.
        # When token positions are also logged we keep raw per-token values via
        # ImageSummary, with `max_norm`/`clip_threshold` allowing inversion at read time.
        if log_pos:
            max_norm, clip_threshold = top_norms[0], top_norms[-1]  # reverse-sorted
            layer.add_summary("max_norm", max_norm)
            layer.add_summary("clip_threshold", clip_threshold)
            normed_top_norms = jnp.where(
                max_norm > clip_threshold,
                (top_norms - clip_threshold) / (max_norm - clip_threshold),
                0.0,  # use dummy value 0.0 if max_norm == clip_threshold
            )
            layer.add_summary("top_norms", ImageSummary(normed_top_norms[None, None, :, None]))
        else:
            layer.add_summary("top_norms_hist", top_norms)

    def fn(layer: "GradientMonitorAndClipLayer", stats: Tensor) -> None:
        if log_pos:
            token_pos = token_pos_fn(stats)
            if token_pos.size == 0:
                raise ValueError(
                    "`log_pos=True` but `token_pos_fn` produced an empty tensor; consider "
                    "syncing `clip_fn.log_pos`, `stats_log_fn.log_pos`, and `stats_shape`."
                )
            log_clipped_token_positions(layer, token_pos)
        log_top_grad_norms(layer, top_norms_fn(stats))

    return fn


class GradientMonitorAndClipLayer(BaseLayer):
    """
    Class for monitoring and modifying activation gradient ∂L/∂y of any layer y = f(x, θ).
    Param gradient is indirectly modified according to ∂L/∂θ = (∂L/∂y) · (∂y/∂θ).

    Usage:
    class MonitoredMultiheadAttention(MultiheadAttention):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._add_child('gradient_clipping', GradientMonitorAndClipLayer.default_config())

        def forward(self, *args, **kwargs):
            res = super().forward(*args, **kwargs)
            return self.gradient_clipping(res)
    """

    @config_class
    class Config(BaseLayer.Config):
        supported_paths: Sequence[str] = GRADIENT_CLIPPING_PATHS
        stats_shape: tuple = (4,)
        clip_fn: InstantiableConfig = config_for_function(top_k_clip_fn)
        stats_log_fn: InstantiableConfig = config_for_function(default_stats_logging_fn)

    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        # Auto-sync stats_log_fn to decide whether to log positions based on clip_fn settings
        if hasattr(cfg.clip_fn, "log_pos") and hasattr(cfg.stats_log_fn, "log_pos"):
            cfg.stats_log_fn.log_pos = cfg.clip_fn.log_pos
        self._clip_fn = cfg.clip_fn.instantiate()
        # Bind `self` so stats_log_fn can call self.add_summary().
        self._stats_log_fn = functools.partial(cfg.stats_log_fn.instantiate(), self)

    def _create_layer_parameter_specs(self) -> dict[str, ParameterSpec]:
        cfg = self.config
        params = dict(
            stats=ParameterSpec(
                shape=cfg.stats_shape,
                mesh_axes=cfg.param_partition_spec,
                initializer=constant_initializer(0.0),
            )
        )
        return params

    @functools.lru_cache(maxsize=1)
    def make_clip_gradient_fn(self):
        cfg = self.config

        @custom_vjp
        def clip_gradient(x, prev_stats):
            del prev_stats
            return x

        def clip_gradient_fwd(x, prev_stats):
            return x, (prev_stats,)

        def clip_gradient_bwd(res, g):
            (prev_stats,) = res
            g, stats = gradient_clipping_impl(
                g,
                clip_fn=self._clip_fn,
                prev_stats=prev_stats,
                supported_paths=cfg.supported_paths,
            )
            # Return `stats` as if it was the gradient of prev_stats
            # Later on `self.parameters["stats"]` value is replaced with `stats` via optimizer.
            return g, stats

        clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
        return clip_gradient

    def forward(self, x):
        # Records local gradient stats from the PREVIOUS bwd pass.
        previous_stats = self.parameters["stats"]
        self._stats_log_fn(previous_stats)

        # Pass previous_stats because we need to return NEW stats as its gradient.
        x = self.make_clip_gradient_fn()(x, previous_stats)
        return x


def create_monitored_layer_class(base_class: Type[BaseLayer]):
    """Factory that creates a new class with activation gradient monitoring.

    Usage:
        MonitoredAttention = create_monitored_layer_class(MultiheadAttention)
        MonitoredFFN = create_monitored_layer_class(TransformerFeedForwardLayer)
    """

    class MonitoredLayer(base_class):
        """A layer class with its output monitored by GradientMonitorAndClipLayer."""

        @config_class
        class Config(base_class.Config):
            gradient_monitor: GradientMonitorAndClipLayer.Config = (
                GradientMonitorAndClipLayer.default_config()
            )

        def __init__(self, cfg: Config, *, parent=None):
            super().__init__(cfg, parent=parent)
            if cfg.gradient_monitor is not None:
                self._add_child("gradient_monitor", cfg.gradient_monitor)

        def forward(self, *args, **kwargs):
            cfg = self.config
            x = super().forward(*args, **kwargs)
            if cfg.gradient_monitor is not None:
                x = self.gradient_monitor(x)
            return x

    MonitoredLayer.__name__ = f"Monitored{base_class.__name__}"
    return MonitoredLayer


def convert_to_monitored_layer_config(
    cfg: InstantiableConfig,
    gradient_monitor_cfg: Optional[GradientMonitorAndClipLayer.Config] = None,
) -> InstantiableConfig:
    """Convert a config for base_class into a config for MonitoredLayer(base_class).

    This function takes an existing layer config and wraps it with gradient monitoring
    capabilities by converting it to use the corresponding MonitoredLayer class.

    Args:
        base_cfg: Config that will materialize into base_class (e.g., MultiheadAttention.Config).
        gradient_monitor_cfg: Optional custom config for gradient monitor. If None,
            uses GradientMonitorAndClipLayer.default_config().

    Returns:
        Config that will materialize into MonitoredLayer(base_class).

    """
    base_class = cfg.klass
    monitored_class = create_monitored_layer_class(base_class)
    monitored_cfg = monitored_class.default_config()

    for field in attr.fields(type(cfg)):
        if field.name != "klass":
            value = getattr(cfg, field.name)
            setattr(monitored_cfg, field.name, value)

    if gradient_monitor_cfg is None:
        gradient_monitor_cfg = GradientMonitorAndClipLayer.default_config()

    monitored_cfg.set(gradient_monitor=gradient_monitor_cfg)

    return monitored_cfg


def gradient_monitoring_learner_cfg_modifier(
    learner_cfg: learner.Learner.Config,
) -> learner.CompositeLearner.Config:
    """Modifies a regular learner_cfg to work with a model with gradient monitoring applied."""

    # `replace_with_updates` optimizer replaces param values with gradient values.
    replacement_learner_cfg = learner.Learner.default_config().set(
        optimizer=config_for_function(optimizers.replace_with_updates),
    )

    ema_decay = learner_cfg.ema.decay
    learner_cfg.ema.decay = None

    cfg = learner.CompositeLearner.default_config().set(
        learners={
            "inner": learner_cfg,
            "gradient_monitor_stats": replacement_learner_cfg,
        },
        rules=[
            (".*gradient_monitor/stats", "gradient_monitor_stats"),
            (".*", "inner"),
        ],
    )
    cfg.ema.decay = ema_decay
    return cfg
