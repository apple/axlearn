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
from jax import custom_vjp
from jax import numpy as jnp

from axlearn.common import learner, optimizers
from axlearn.common.base_layer import BaseLayer, ParameterSpec
from axlearn.common.config import InstantiableConfig, config_class, config_for_function
from axlearn.common.param_init import constant_initializer
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
    assert 2 <= k < len(flat_norm), "k must be >= 2 and < token count."

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
    k: int = 4,
    method: Literal["approx", "topk", "sort"] = "approx",
) -> Callable[[Tensor], tuple[Tensor, Tensor]]:
    """Returns an activation gradient clipping function that zeros out tokens
    whose gradient norm is among the top-k largest in a batch.

    Args:
        k: number of tokens to remove per batch.
        method: algorithm to find the top-k:
            - approx: use ``jax.lax.approx_max_k``
            - topk: use ``jax.lax.top_k``
            - sort: use sorting to find top k

    Returns:
        A function mapping gradient tensor to (clipped gradient, top-k norms).
    """

    def fn(g: Tensor) -> tuple[Tensor, Tensor]:
        clipped, topk, _ = _top_k_clipping(g, k, method)
        return clipped, topk

    return fn


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

    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        self._clip_fn = cfg.clip_fn.instantiate()

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
        self.add_summary("top_norms", previous_stats)

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
