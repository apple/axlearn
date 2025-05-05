# Copyright © 2023 Apple Inc.

# pylint: disable=too-many-lines
"""Common utilities to set up GPT model trainer configs.

The most important function in this module is `get_get_trainer_config_fn`. Almost all the other
functions are used to build the args for `get_get_trainer_config_fn`, including `model_cfg`,
`learner_cfg`, `train_input_source`, and `evalers`.

See c4_trainer.py for how they are used.
"""
import os
import math
from collections.abc import Sequence
from typing import Literal, Optional, Protocol, Union

import jax.numpy as jnp
import tensorflow as tf
from jax.sharding import PartitionSpec

from axlearn.common import (
    base_model,
    causal_lm,
    input_base,
    input_fake,
    input_lm,
    input_tf_data,
    learner,
    optimizers,
    schedule,
    state_builder,
)
from axlearn.common.attention import (
    AttentionLogitBiasLayer,
    BaseKVCache,
    BaseQKVLinear,
    MultiheadAttention,
    RepeatedTransformerLayer,
    StackedTransformerLayer,
    TransformerLayer,
    build_remat_spec,
    set_double_shard_weights_config,
)
from axlearn.common.checkpointer import every_n_steps_and_last_policy
from axlearn.common.config import (
    ConfigOr,
    FunctionConfigBase,
    InstantiableConfig,
    config_for_function,
    maybe_instantiate,
    maybe_set_config,
)
from axlearn.common.decoder import Decoder
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.evaler import BaseMetricCalculator, ModelSummaryAccumulator, SpmdEvaler
from axlearn.common.evaler import every_n_steps_policy as eval_every_n_steps_policy
from axlearn.common.flash_attention.layer import FlashAttention
from axlearn.common.input_dispatch import InputDispatcher
from axlearn.common.layers import BaseNormalizationLayer, set_bias_recursively, set_norm_recursively
from axlearn.common.mixture_of_experts import TransformerFeedForwardMoE
from axlearn.common.optimizer_base import PartitionedGradientTransformation
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.summary_writer import BaseWriter
from axlearn.common.trainer import MeshShape, SpmdTrainer
from axlearn.common.utils import HybridMeshShape, Nested, get_data_dir
from axlearn.experiments.text.common import DataMixtureComponent, tfds_text_source
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

REPLACE_NEWLINES_WITH = "<n>"
EVAL_EVERY_N_STEPS = 5_000


# We typically use bfloat16 as the step dtype,
# (but usually keep parameters and optimizer state in float32).
STEP_DTYPE = jnp.bfloat16


# The default mesh-axis names for LM training, from least to most communication intensive.
# See mesh_shape_from_axes() docstring for more details.
MESH_AXIS_NAMES = ("pipeline", "data", "expert", "fsdp", "seq", "model")


def scaled_hidden_dim(scale: float, *, round_up_to_multiples_of: int = 256) -> FunctionConfigBase:
    def scale_fn(input_dim: int, *, scale: float, round_up_to_multiples_of: int) -> int:
        return math.ceil(input_dim * scale / round_up_to_multiples_of) * round_up_to_multiples_of

    return config_for_function(scale_fn).set(
        scale=scale,
        round_up_to_multiples_of=round_up_to_multiples_of,
    )


def flash_attention_config() -> FlashAttention.Config:
    """Builds a FlashAttention config with sharding config."""
    return FlashAttention.default_config().set(
        causal=True,
        mha_dim_to_partition_spec={
            "btnh": PartitionSpec(("data", "expert", "fsdp"), None, ("seq", "model"), None),
            "bsnh": PartitionSpec(("data", "expert", "fsdp"), None, ("seq", "model"), None),
            "bnts": PartitionSpec(("data", "expert", "fsdp"), None, None, None),
        },
        output_dim_to_partition_spec={
            "btnh": PartitionSpec(("data", "expert", "fsdp"), "seq", "model", None),
            "bnts": PartitionSpec(("data", "expert", "fsdp"), "model", "seq", None),
        },
    )


def tfds_input(
    *,
    is_training: bool,
    dataset_name: str,
    split: str,
    vocab_cfg: InstantiableConfig,
    max_sequence_length: int,
    train_shuffle_buffer_size: int = 1024 * 16,
    replace_newlines_with: str = "\n",
) -> input_tf_data.BuildDatasetFn:
    """Builds a BuildDatasetFn.

    Args:
        is_training: Whether the input will be used for training
            (and therefore can be packed and shuffled).
        dataset_name: The TFDS dataset name.
        split: The TFDS dataset split, e.g., "train".
        vocab_cfg: A config that instantiates to a seqio.Vocabulary.
        max_sequence_length: The maximum sequence length (after packing).
        train_shuffle_buffer_size: The shuffle buffer size used for training.
        replace_newlines_with: Swaps newlines with this string as a preprocessing step.
            For Sentencepiece BPE tokenizers we use a custom user-defined token for newline.

    Returns:
        A BuildDatasetFn.
    """
    source = config_for_function(tfds_text_source).set(
        dataset_name=dataset_name,
        split=split,
        train_shuffle_buffer_size=train_shuffle_buffer_size,
    )
    processor = config_for_function(
        input_lm.text_to_lm_training_input if is_training else input_lm.text_to_lm_eval_input
    ).set(
        vocab_cfg=vocab_cfg,
        max_len=max_sequence_length,
        replace_newlines_with=replace_newlines_with,
    )

    # For eval we do a strided slice for each document.
    if not is_training:
        processor.set(stride=max_sequence_length // 4)
    return input_tf_data.with_processor(
        source=source,
        processor=processor,
        is_training=is_training,
    )


def mesh_shape_from_axes(
    *,
    pipeline: int = 1,
    data: int = 1,
    expert: int = 1,
    fsdp: int = 1,
    seq: int = 1,
    model: int = 1,
) -> Sequence[int]:
    """Builds a 6D logical mesh from the provided spec.

    Args:
        pipeline: Pipeline-paralellism. Typically means partitioning model layers across this axis.
            E.g. <https://arxiv.org/abs/1811.06965>.
        data: For data-parallelism. Expect model state to be fully replicated over this axis.
            Useful for e.g. multi-slice/granule partitioning with slow networking between granules.
        expert: Designed to be used for partitioning "experts" in mixture-of-expert models.
            E.g. <https://arxiv.org/abs/2006.16668>.
        fsdp: Fully-sharded-data-parallelism a.k.a. async-with-compute model-parallelism.
            E.g. <https://arxiv.org/abs/1910.02054>.
        seq: Used for sequence-parallelism. Typically this means sharding the activation sequence
            dimension, and possibly a subset of the weights.
        model: Tensor parallelism a.k.a. sync-with-compute model-parallelism.
            E.g. <https://arxiv.org/abs/1909.08053>.

    Returns:
        A tuple describing the logical mesh shape (from least to most communication intensive).
    """
    assert MESH_AXIS_NAMES == ("pipeline", "data", "expert", "fsdp", "seq", "model")
    # We set the minimum size for a mesh axis to 1 as anything lower is degenerate, except -1.
    return tuple(max(x, 1) if x != -1 else -1 for x in [pipeline, data, expert, fsdp, seq, model])


def update_model_remat_config(
    *,
    stack_cfg: causal_lm.TransformerStackConfig,
    layer_cfg: TransformerLayer.Config,
):
    """Recomputes and sets the remat_spec based on provided layer_cfg.

    Args:
        stack_cfg: The transformer stack config.
        layer_cfg: The transformer layer config.
        offload_dst: Destination of remat checkptoing offloading.

    """

    remat_spec = build_remat_spec(stack_cfg.clone(layer=layer_cfg))
    layer_cfg.set(remat_spec=remat_spec)


def model_config(
    *,
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    vocab_size: int,
    activation_fn: Union[str, Sequence[str]],
    ffn_dim: Union[int, FunctionConfigBase],
    normalization: BaseNormalizationLayer.Config,
    dropout_rate: float = 0.0,
    stack_cfg: causal_lm.TransformerStackConfig = RepeatedTransformerLayer.default_config(),
    emb_cfg: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config(),
    lm_head_cfg: Optional[TransformerTextEmbeddings.Config] = None,
    layer_cfg: TransformerLayer.Config = TransformerLayer.default_config(),
    attention_cfg: Optional[MultiheadAttention.Config] = None,
    attention_qkv_linear: Optional[BaseQKVLinear.Config] = None,
    attention_kv_cache: Optional[BaseKVCache.Config] = None,
    attention_mask: Optional[AttentionLogitBiasLayer.Config] = None,
    z_loss_scale: float = 0.0,
    ffn_structure: str = "prenorm",
    atten_structure: str = "prenorm",
    atten_logit_cap: Optional[float] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    ffn_layer_types: Optional[Sequence[Literal["dense", "sparse"]]] = None,
    expert_cfg: TransformerFeedForwardMoE = TransformerFeedForwardMoE.default_config(),
) -> causal_lm.Model.Config:
    """Returns an LM model config based on the given hyperparams.

    Args:
        hidden_dim: The transformer layer input/output dim.
        num_heads: The number of attention heads.
        num_layers: The number of transformer Layers.
        vocab_size: The vocabulary size.
        activation_fn: The activation function used for the feed-forward network.
        ffn_dim: The feed-forward dimension or function (e.g., returned by `scaled_hidden_dim`).
        normalization: The normalization layer config used by both attention and feed-forward
            layers.
        dropout_rate: The dropout rate applied throughout the model.
            Defaults to 0.0 (i.e. no dropout).
        stack_cfg: The transformer stack config.
        emb_cfg: The transformer embedding layer config.
        lm_head_cfg: The transformer LmHead layer config.
        layer_cfg: The transformer layer config.
        attention_cfg: The attention config.
        attention_qkv_linear: The attention QKV linear layer.
        attention_kv_cache: The attention KV Cache layer.
        attention_mask: The AttentionLogitBiasLayer config.
        z_loss_scale: The scalar weight for the z-loss to encourages the cross-entropy loss
            normalizer to be well-behaved.
        ffn_structure: The inner structure of the feedforward layer.
            Options: See `SupportedNormStructure`.
        atten_structure: The inner structure of the attention layer.
            Options: See `SupportedNormStructure`.
        atten_logit_cap: Cap the absolute values of logits by tanh.
            Enabled by setting a positive value.
        remat_offload_dst: Destination of remat checkptoing offloading.
        pad_token_id: Int ID of the inputs to be masked for self-attention.
        eos_token_id: Int ID of the end of sequence token id.
        ffn_layer_types: The types of layers in the FFN. Options: [dense, sparse].
        expert_cfg: The expert config for the MoE FFN.
    Returns:
        A causal LM config.
    """
    # Feed-forward.
    # layer_cfg.feed_forward.activation = activation_fn
    # layer_cfg.feed_forward.hidden_dim = ffn_dim
    # layer_cfg.feed_forward.structure = ffn_structure
    # First configure the base layer_cfg.
    # Attention.
    if attention_cfg is not None:
        layer_cfg.self_attention.attention = attention_cfg
    layer_cfg.self_attention.attention.causal = True
    layer_cfg.self_attention.attention.num_heads = num_heads
    if attention_qkv_linear is not None:
        layer_cfg.self_attention.attention.input_linear = attention_qkv_linear
    if attention_kv_cache is not None:
        layer_cfg.self_attention.attention.kv_cache = attention_kv_cache
    layer_cfg.self_attention.structure = atten_structure
    layer_cfg.self_attention.attention.atten_logit_cap = atten_logit_cap
    # if os.getenv('AXLEARN_REMAT_LAYER', 'true') == 'true':
    #     if issubclass(stack_cfg.klass, (RepeatedTransformerLayer, StackedTransformerLayer)):
    #         update_model_remat_config(stack_cfg=stack_cfg, layer_cfg=layer_cfg)
    # Stack.
    # transformer_cfg = stack_cfg.set(num_layers=num_layers, layer=layer_cfg)
    # Shard some FFN and attention weights over multiple axes.
    batch_axis_names = ("data", "expert", "fsdp")
    set_double_shard_weights_config(
        layer_cfg,
        batch_axis_names=batch_axis_names,
        fsdp_axis_names=("expert", "fsdp", "seq"),
        tp_axis_names="model",
        seq_axis_names="seq",
    )
    def config_dense(cfg: TransformerLayer.Config) -> TransformerLayer.Config:
        cfg = layer_cfg.clone()
        cfg.feed_forward.activation = activation_fn
        cfg.feed_forward.hidden_dim = ffn_dim
        cfg.feed_forward.structure = ffn_structure
        return cfg

    def config_sparse(cfg: TransformerLayer.Config) -> TransformerLayer.Config:
        cfg = layer_cfg.clone()
        cfg.feed_forward = expert_cfg
        cfg.feed_forward.activation = activation_fn
        cfg.feed_forward.hidden_dim = ffn_dim
        cfg.feed_forward.structure = ffn_structure
        return cfg

    fn = {
        "dense": config_dense,
        "sparse": config_sparse,
    }
    if ffn_layer_types is None:
        lm_layer_cfg = config_dense(layer_cfg)
    else:
        lm_layer_cfg = [fn[layer_type](layer_cfg) for layer_type in ffn_layer_types]

    # Single layer repeated num_layers times.
    if not isinstance(lm_layer_cfg, Sequence):
        transformer_cfg = stack_cfg.set(num_layers=num_layers, layer=lm_layer_cfg)
    elif len(lm_layer_cfg) == 1:
        # No need to stack together a single layer.
        transformer_cfg = stack_cfg.set(num_layers=num_layers, layer=lm_layer_cfg[0])
    else:
        num_layers_cfgs = len(lm_layer_cfg)
        # Stack together the layers.
        transformer_cfg = stack_cfg.set(
            num_layers=num_layers // num_layers_cfgs,
            layer=StackedTransformerLayer.default_config().set(
                num_layers=num_layers_cfgs, layer=list(lm_layer_cfg)
            ),
        )



    decoder_cfg = Decoder.default_config().set(
        transformer=transformer_cfg,
        attention_mask=attention_mask,
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=emb_cfg,
        lm_head=lm_head_cfg,
        dropout_rate=dropout_rate,
    )
    if pad_token_id is not None:
        decoder_cfg.set(pad_token_id=pad_token_id)
    if eos_token_id is not None:
        decoder_cfg.set(eos_token_id=eos_token_id)
    # Model.
    model_param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan="fan_in", distribution="normal"
            )
        }
    )
    batch_axis_names = ("data", "expert", "fsdp")
    cfg: causal_lm.Model.Config = causal_lm.Model.default_config().set(
        decoder=decoder_cfg,
        param_init=model_param_init,
        batch_axis_names=None,  # We use input dispatch to partition batches.
    )
    if z_loss_scale:
        cfg.metrics = causal_lm.metrics_config(z_loss_scale=z_loss_scale)
    cfg.dtype = jnp.float32
    cfg.decoder.logits_partition_spec = (batch_axis_names, "seq", "model")
    set_bias_recursively(cfg, False)
    set_norm_recursively(cfg, normalization)
    return cfg


def mup_simple_adam_update_transformation(scale_factor: float) -> InstantiableConfig:
    """Builds a transform which scales the adam update for linear layers.

    References:
    - <https://arxiv.org/abs/2309.14322> Section 3.2.4.

    Args:
        scale_factor: The factor by which the update will be scaled for linear layers.

    Returns:
        A config that instantiates to a partitioned gradient transformation.
    """
    return config_for_function(optimizers.scale_update_per_param).set(
        per_param_scale=config_for_function(optimizers.per_param_scale_by_path).set(
            scale_by_path=[
                (".*attention/o_proj/weight", scale_factor),
                (".*attention/i_proj/i_proj/qkv_proj/weight", scale_factor),
                # Dense FFN weights.
                (".*feed_forward/linear1_0/weight", scale_factor),
                (".*feed_forward/linear1_1/weight", scale_factor),
                (".*feed_forward/linear2/weight", scale_factor),
                # MoE FFN weights.
                (".*feed_forward/wi_0_weight", scale_factor),
                (".*feed_forward/wi_1_weight", scale_factor),
                (".*feed_forward/wo_weight", scale_factor),
            ],
            description="scale_by_mup_simple",
            default_scale=1.0,
        )
    )


def adamw_decoupled_learner_config(
    *,
    peak_lr: float,
    max_step: int,
    weight_decay: float,
    lr_warmup_steps: int = 2000,
    alpha: float = 0.1,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None,
) -> learner.Learner.Config:
    """Build learner using the AdamW optimizer and a cosine lr schedule with linear warmup."""
    update_schedule = config_for_function(schedule.cosine_with_linear_warmup).set(
        peak_lr=1.0,
        max_step=max_step,
        warmup_steps=lr_warmup_steps,
        begin_value=0.0,
        # Decay to this fraction of the peak_lr.
        alpha=alpha,
    )
    optimizer_cfg = config_for_function(optimizers.chain).set(
        args=[
            config_for_function(optimizers.clip_by_global_norm).set(max_norm=1),
            config_for_function(optimizers.adamw_decoupled_optimizer).set(
                learning_rate=peak_lr,
                b1=b1,
                b2=b2,
                eps=eps,
                update_schedule=update_schedule,
                weight_decay=weight_decay,
                weight_decay_per_param_scale=None,
                adam_update_transformation=adam_update_transformation,
            ),
        ]
    )
    return learner.Learner.default_config().set(optimizer=optimizer_cfg)


def adastar_learner_config(
    *,
    peak_lr: float,
    max_step: int,
    lr_warmup_steps: int = 2000,
    alpha: float = 0.005,
    weight_decay: float = 3.16e-4,
    b1: float = 0.95,
    b2: float = 0.95,
    adam_update_transformation: Optional[ConfigOr[PartitionedGradientTransformation]] = None,
) -> learner.Learner.Config:
    """Build learner using the AdaStar optimizer and a cosine lr schedule with linear warmup."""
    update_schedule = config_for_function(schedule.cosine_with_linear_warmup).set(
        peak_lr=1.0,
        max_step=max_step,
        warmup_steps=lr_warmup_steps,
        begin_value=0.0,
        # Decay to this fraction of the peak_lr.
        alpha=alpha,
    )
    optimizer_cfg = config_for_function(optimizers.skip_and_clip_by_global_norm).set(
        inner=config_for_function(optimizers.adastar_optimizer).set(
            learning_rate=peak_lr,
            # adafactor does not apply smoothing on gradients (but on raw updates).
            gradient_ema_decay=None,
            gradient_ema_debias=None,
            gradient_square_ema_decay=b2,
            gradient_square_ema_debias=True,
            eps=0,
            # adafactor eps is applied on the square.
            eps_square=1e-30,
            # Clipping is applied on raw updates by per-param norm (not global norm).
            raw_update_clipping_threshold=1.0,
            # Smoothing is applied on raw updates.
            update_ema_decay=b1,
            # ... but without debiasing (!).
            update_ema_debias=False,
            weight_decay=weight_decay,
            update_schedule=update_schedule,
            adam_update_transformation=adam_update_transformation,
        ),
        drop_norm=100,
        max_norm=1,
    )
    return learner.Learner.default_config().set(optimizer=optimizer_cfg)


def tfds_read_config() -> InstantiableConfig:
    return config_for_function(input_tf_data.tfds_read_config).set(
        read_parallelism=1,
        decode_parallelism=8,
    )


def mixture_train_input_source(
    *,
    is_training: bool,
    vocab_cfg: InstantiableConfig,
    preprocessor: Union[InstantiableConfig, list[InstantiableConfig]],
    data_mixture_components: Union[InstantiableConfig, list[DataMixtureComponent]],
    max_sequence_length: int,
    replace_newlines_with: str = REPLACE_NEWLINES_WITH,
    fake_input_source_cfg: Optional[InstantiableConfig] = None,
    autotune_ram_budget_gb: Optional[int] = None,
) -> input_tf_data.BuildDatasetFn:
    """Build mixture training input source for decoder-only LM model.

    Mixture sampling happens after input processing but before batching, meaning that each batch
    example will only contain tokens from a single source.

    Args:
        is_training: A boolean indicating that inputs will be used for training.
        max_sequence_length: Maximum sequence length of an example.
        vocab_cfg: Config to instantiate the seqio vocab.
        preprocessor: A single or a list of lm text preprocessor config(s). When
            used as a list, each preprocessor must correspond to one data source;
            when used as a single config, it will be broadcast for all data sources.
        data_mixture_components: List of DataMixtureComponent(s).
        replace_newlines_with: Value to replace newlines with in the text.
        fake_input_source_cfg: A config that instantiates to a BuildDatasetFn for the input source
            used during unittest.
        autotune_ram_budget_gb: The memory budget (in GiB) the tensorflow datasets optimization
            pipeline will target. Typically configure as 50%-75% of available memory.
            If None, uses tensorflow defaults.

    Returns:
        A BuildDatasetFn that mixes the given list of DataMixtureComponent(s).
    """
    data_mixture_components = maybe_instantiate(data_mixture_components)
    if fake_input_source_cfg is None:
        fake_input_source_cfg = config_for_function(input_fake.fake_text_source).set(
            is_training=True
        )

    sources = []
    weights = []
    for component in data_mixture_components:
        if get_data_dir() == "FAKE":
            sources.append(fake_input_source_cfg)
        else:
            sources.append(
                config_for_function(input_tf_data.tfds_dataset).set(
                    dataset_name=component.name,
                    split=component.split,
                    train_shuffle_buffer_size=64 * component.shuffle_buffer_size,
                    read_config=tfds_read_config(),
                )
            )
        weights.append(component.weight)

    def _set_config_for_preprocessor(p: InstantiableConfig) -> InstantiableConfig:
        return maybe_set_config(
            p,
            vocab_cfg=vocab_cfg,
            max_sequence_length=max_sequence_length,
            replace_newlines_with=replace_newlines_with,
        )

    if isinstance(preprocessor, list):
        assert len(preprocessor) == len(data_mixture_components)
        processors = [_set_config_for_preprocessor(p) for p in preprocessor]
    else:
        processors = [_set_config_for_preprocessor(preprocessor) for _ in sources]

    sources = [
        config_for_function(input_tf_data.with_processor).set(
            source=source,
            processor=processor,
        )
        for source, processor in zip(sources, processors)
    ]

    return input_tf_data.sample_from_datasets(
        sources=sources,
        weights=weights,
        is_training=is_training,
        autotune_ram_budget_gb=autotune_ram_budget_gb,
    )


def evaler_config_dict(
    input_source_configs: dict[str, InstantiableConfig],
    *,
    metric_calculators: Optional[dict[str, BaseMetricCalculator.Config]] = None,
    summary_writer: Optional[BaseWriter.Config] = None,
) -> dict[str, SpmdEvaler.Config]:
    """Makes evaler configs from the given input sources.

    Args:
        input_source_configs: A dictionary with the eval dataset name(s) as key(s) and
            InstantiableConfig(s) that instantiate to BuildDatasetFn(s) as value(s).
        metric_calculators: A dictionary with eval dataset name(s) as key(s)
            and config(s) that instantiate to BaseMetricCalculator(s)
            as value(s) to override the default metric calculator for the specified eval datasets.
            If None, all eval datasets will use the default metric calculator.
        summary_writer: An optional custom summary writer.

    Returns:
        A dictionary of SpmdEvaler configs.

    Raises:
        ValueError: If metric_calculators have any key not in input_source_configs.
    """
    # Check to ensure input_source_configs and metric_calculators are matched.
    if metric_calculators and not set(metric_calculators.keys()).issubset(
        set(input_source_configs.keys())
    ):
        raise ValueError(
            f"Ensure the keys of metric_calculators come from "
            f"input_source_configs!"
            f" {set(metric_calculators.keys())} not in"
            f"{set(input_source_configs.keys())}"
        )

    evalers = {}
    for dataset_name, input_source_config in input_source_configs.items():
        evaler_input = input_tf_data.Input.default_config().set(
            is_training=False,
            source=input_source_config,
            input_dispatcher=InputDispatcher.default_config(),
            processor=config_for_function(input_tf_data.identity),
            batcher=config_for_function(input_tf_data.per_feed_batch).set(
                prefetch_buffer_size=tf.data.AUTOTUNE,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
        )
        metric_calculator = (
            metric_calculators[dataset_name]
            if metric_calculators and dataset_name in metric_calculators
            else ModelSummaryAccumulator.default_config()
        )
        evaler_cfg = evalers[dataset_name] = SpmdEvaler.default_config().set(
            input=evaler_input,
            eval_dtype=STEP_DTYPE,
            metric_calculator=metric_calculator,
        )
        if summary_writer is not None:
            evaler_cfg.set(summary_writer=summary_writer)
    return evalers


def make_config_name(
    arch: str, model_size: str, *, version: Optional[str] = None, suffix: Optional[str] = None
) -> str:
    """Makes config name string as a function of architecture and model-size.

    Useful to keep config names synced with fine-tuning configs.

    Args:
        arch: The architecture of the model.
        model_size: The number of transformer parameters (not including vocab embeddings).
        version: An optional version string.
        suffix: Optional config suffix.

    Returns:
        f"{arch}-{model_size}".
        If version is supplied, a f"-{version}" suffix will be appended.
        If suffix is supplied, it will be appended last (without any delimiter).
    """
    name = f"{arch}-{model_size}"
    if version:
        name += f"-{version}"
    if suffix:
        name += suffix
    return name


def get_trainer_config_fn(
    *,
    model_cfg: base_model.BaseModel.Config,
    learner_cfg: learner.Learner.Config,
    max_step: int,
    train_batch_size: int,
    train_input_source: InstantiableConfig[input_tf_data.BuildDatasetFn],
    evalers: dict[str, SpmdEvaler.Config],
    mesh_shape: Union[MeshShape, HybridMeshShape],
    mesh_axis_names: Sequence[str] = MESH_AXIS_NAMES,
    mesh_rules: Optional[Sequence[tuple[str, Optional[Union[MeshShape, HybridMeshShape]]]]] = None,
    eval_every_n_steps: int = 5000,
    eval_batch_size: Optional[int] = None,
    keep_every_n_steps: int = 50_000,
    save_every_n_steps: Optional[int] = None,
    init_state_builder: Optional[state_builder.Builder.Config] = None,
) -> TrainerConfigFn:
    """Builds a TrainerConfigFn according to the model and input specs.

    Args:
        model_cfg: The model config.
        learner_cfg: The learner config.
        max_step: The maximum number of training steps.
        train_batch_size: The global batch size for training inputs.
        train_input_source: A config (often constructed via `tfds_input` or
            `mixture_train_input_source`) that instantiates to a BuildDatasetFn.
        evalers: A dict whose keys represent evaler names and values are configs
            that each instantiates to a SpmdEvaler.
        mesh_shape: The mesh shape.
        mesh_axis_names: The mesh axis names.
        mesh_rules: Optional rules to use different mesh shapes based on the mesh_selector.
        eval_every_n_steps: How often to run evaluation.
        keep_every_n_steps: How often to keep a checkpoint permanently.
        save_every_n_steps: How often to write a checkpoint.
            If None, defaults to a value based on eval_every_n_steps.
        init_state_builder: Builder to initialize trainer states. If none, default initializer.
            Load a checkpoint using state_builder.TensorStoreStateStorageBuilder, setting `dir` to
            the checkpoint path (such as mixture_general_lm.PRETRAINED_CHECKPOINTS[config_name]).

    Returns:
        A function that returns a trainer config.
    """

    def config_fn() -> InstantiableConfig:
        cfg: SpmdTrainer.Config = SpmdTrainer.default_config()
        cfg.name = "gpt_trainer"
        cfg.model = model_cfg
        cfg.learner = learner_cfg
        cfg.max_step = max_step
        cfg.train_dtype = STEP_DTYPE
        cfg.input = input_tf_data.Input.default_config().set(
            is_training=True,
            source=train_input_source,
            input_dispatcher=InputDispatcher.default_config().set(
                global_logical_batch_size=train_batch_size,
            ),
            processor=config_for_function(input_tf_data.identity),
            batcher=config_for_function(input_tf_data.per_feed_batch).set(
                prefetch_buffer_size=tf.data.AUTOTUNE,
                pad_example_fn=input_tf_data.default_pad_example_fn,
            ),
            input_partitioner=config_for_function(input_base.partition_by_path_rank).set(
                path_rank_to_partition={
                    # Note: the batch axes are different here than in `cfg.batch_axis_names`,
                    # as we partition sequence dim over `seq`.
                    (None, 1): PartitionSpec(("data", "expert", "fsdp")),
                    (None, 2): PartitionSpec(("data", "expert", "fsdp"), "seq"),
                }
            ),
        )
        cfg.evalers = {}
        for name, evaler_cfg in evalers.items():
            evaler_cfg.input.input_dispatcher.global_logical_batch_size = (
                eval_batch_size or train_batch_size
            )
            evaler_cfg.set(
                eval_policy=config_for_function(eval_every_n_steps_policy).set(
                    n=eval_every_n_steps,
                    max_step=max_step,
                )
            )
            cfg.evalers[name] = evaler_cfg
        # Summaries and checkpoints.
        cfg.checkpointer.save_policy = config_for_function(every_n_steps_and_last_policy).set(
            n=save_every_n_steps or min(eval_every_n_steps, 5_000),
            max_step=max_step,
        )
        cfg.checkpointer.keep_every_n_steps = min(max_step, keep_every_n_steps)
        cfg.checkpointer.keep_last_n = 3
        cfg.summary_writer.write_every_n_steps = min(eval_every_n_steps, 100)
        cfg.summary_writer.max_queue = 1000
        if len(mesh_axis_names) != len(mesh_shape):
            raise ValueError(
                f"Number of mesh axis names ({mesh_axis_names}) "
                f"must match number of mesh dims ({mesh_shape})."
            )
        cfg.mesh_axis_names = mesh_axis_names
        cfg.mesh_shape = mesh_shape
        # Set batch sharding spec to exclude the "model" axis (assumed for tensor-parallelism) and
        # "pipeline" axis (for pipeline parallelism).
        # TODO(markblee): Remove this and use `cfg.input.input_partitioner`.
        cfg.batch_axis_names = tuple(
            el for el in mesh_axis_names if el not in ("model", "pipeline")
        )
        cfg.mesh_rules = mesh_rules
        # Maybe load state.
        if init_state_builder:
            cfg.init_state_builder = init_state_builder
        return cfg

    return config_fn


class SourceBuilder(Protocol):
    """A protocol that builds an input source."""

    def __call__(
        self, *, vocab_size: int, max_sequence_length: int
    ) -> Nested[ConfigOr[input_tf_data.BuildDatasetFn]]:
        raise NotImplementedError(type(self))
