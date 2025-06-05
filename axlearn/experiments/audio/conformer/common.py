# Copyright © 2024 Apple Inc.

"""Common Conformer config builders.

[1] https://arxiv.org/abs/2005.08100
"""

from typing import Optional

import seqio

from axlearn.audio.decoder_asr import CTCDecoderModel, TransducerDecoderModel
from axlearn.audio.encoder_asr import ASREncoder, SpeechContextNetwork, SpeechFeatureLayer
from axlearn.audio.input_asr import make_autoregressive_inputs, speech_input, text_input
from axlearn.common import input_tf_data, optimizers, schedule
from axlearn.common.attention import (
    FusedQKVLinear,
    MultiheadAttentionXL,
    PerDimScale,
    ScaleQuery,
    build_remat_spec,
)
from axlearn.common.config import InstantiableConfig, config_for_function
from axlearn.common.conformer import RepeatedConformerLayer
from axlearn.common.layers import VariationalNoise, set_dropout_rate_recursively
from axlearn.common.learner import Learner
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.common.rnn import LSTMCell


def stack_config(*, num_layers: int, num_heads: int) -> RepeatedConformerLayer.Config:
    """Builds a Conformer stack config.

    Args:
        num_layers: Number of Conformer layers.
        num_heads: Number of attention heads.

    Returns:
        A repeated Conformer layer.
    """

    cfg: RepeatedConformerLayer.Config = RepeatedConformerLayer.default_config()
    cfg.num_layers = num_layers
    cfg.layer.self_attention.attention = MultiheadAttentionXL.default_config().set(
        input_linear=FusedQKVLinear.default_config(),
        num_heads=num_heads,
        query_scale=ScaleQuery.default_config().set(
            per_dim_scale=PerDimScale.default_config(),
        ),
        scale_position=MultiheadAttentionXL.ScalePosition.QUERY,
    )
    cfg.layer.self_attention.attention.input_linear.layer.bias = True
    cfg.layer.lconv.conv.window = 32  # Conv kernel size is 32 across model sizes.
    cfg.remat_spec = build_remat_spec(cfg)

    return cfg


def encoder_config(
    *,
    dim: int,
    stack_cfg: RepeatedConformerLayer.Config,
    feature_cfg: SpeechFeatureLayer.Config,
    dropout_rate: Optional[float] = 0.1,
) -> ASREncoder.Config:
    """Builds a Conformer encoder config.

    Args:
        dim: Encoder dimension.
        stack_cfg: Conformer stack config. See also `stack_config`.
        feature_cfg: Feature layer config. See also `librispeech_trainer.feature_config`.
        dropout_rate: Dropout to apply throughout the encoder.

    Returns:
        Asr encoder config.
    """
    context_cfg = SpeechContextNetwork.default_config().set(context=stack_cfg)
    cfg: ASREncoder.Config = ASREncoder.default_config().set(
        dim=dim, feature=feature_cfg, context=context_cfg
    )
    if dropout_rate:
        set_dropout_rate_recursively(cfg.context, dropout_rate=dropout_rate)
    return cfg


def rnn_transducer_config(
    *,
    vocab_size: int,
    lm_dim: int,
    emb_dim: int,
    joint_dim: int,
    eos_id: int,
    vn_std: float = 0.075,
) -> TransducerDecoderModel.Config:
    """Builds an RNN Transducer (RNN-T) decoder config.

    Args:
        vocab_size: Text vocab size.
        lm_dim: Output dim of prediction network.
        emb_dim: Embedding dim.
        joint_dim: Joint network dim.
        eos_id: EOS token ID.
        vn_std: Variational noise applied to the prediction network.

    Returns:
        A RNN-T model config.
    """
    rnn_cfg: LSTMCell.Config = LSTMCell.default_config().set(
        max_cell_value=10.0,
        output_proj=None,
        norm=None,  # No norm for input_proj.
    )
    rnn_cfg.param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan=None, distribution="uniform", scale=0.1
            )
        }
    )
    rnn_cfg.input_proj.bias = True

    cfg: TransducerDecoderModel.Config = TransducerDecoderModel.default_config().set(
        vocab_size=vocab_size, lm_dim=lm_dim, joint_dim=joint_dim, eos_id=eos_id
    )
    cfg.prediction_network.set(
        emb_dim=emb_dim,
        rnn_cell=rnn_cfg,
        # Add variational noise to LSTM gate weights and embedding weights, including biases.
        param_noise=VariationalNoise.default_config().set(vn_std=vn_std),
    )
    cfg.prediction_network.embedding.param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan=None, distribution="uniform", scale=1.0
            )
        }
    )
    return cfg


def ctc_decoder_config(vocab_size: int = 1024) -> CTCDecoderModel.Config:
    """Builds a CTC decoder config.

    Args:
        vocab_size: Text vocab size.

    Returns:
        A CTC decoder model config.
    """
    cfg: CTCDecoderModel.Config = CTCDecoderModel.default_config().set(vocab_size=vocab_size)
    cfg.lm_head.param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan=None, distribution="uniform", scale=0.5
            )
        }
    )
    return cfg


def asr_input(
    *,
    max_source_len: int,
    max_target_len: int,
    vocab_cfg: InstantiableConfig[seqio.Vocabulary],
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> input_tf_data.DatasetToDatasetFn:
    """An ASR input processor.

    Args:
        max_source_len: Max length of "source" inputs (to be fed to an ASR encoder).
        max_target_len: Max length of "target" inputs (to be fed to an ASR decoder).
        vocab_cfg: A config that instantiates to a vocab.
        bos_id: An optional BOS token ID. If None, attempts to infer from vocab.
        eos_id: An optional EOS token ID. If None, attempts to infer from vocab.

    Returns:
        A process for ASR inputs.
        * Each input example is a dict containing "speech" with float values and "text" with text
            values (and possibly other keys);
        * Each output example contains "source/inputs" and "source/paddings" corresponding to speech
            values processed according to `speech_input`; "target/input_ids" and "target_labels"
            corresponding to text values processed according to `text_input`. "source" inputs can be
            provided to an ASR encoder and "target" inputs to an ASR decoder.
    """
    vocab = vocab_cfg.instantiate()
    bos = vocab.bos_id if bos_id is None else bos_id
    eos = vocab.eos_id if eos_id is None else eos_id
    key_map = {
        "source/inputs": "inputs",
        "source/paddings": "paddings",
        "target/input_ids": "input_ids",
        "target_labels": "target_labels",
    }
    return input_tf_data.chain(
        speech_input(max_len=max_source_len, truncate=False),
        text_input(max_len=max_target_len, vocab=vocab_cfg, truncate=False, eos_id=eos),
        make_autoregressive_inputs(vocab=vocab_cfg, bos_id=bos),
        input_tf_data.rekey(key_map=key_map, default_value=None, separator="/"),
    )


def adam_learner_config(
    *,
    lr_scale: float,
    warmup_steps: int = 10_000,
    ema_decay: Optional[float] = 0.9999,
) -> Learner.Config:
    """Constructs an Adam learner config.

    See [1] Section 3.2 for details.

    Reference:
    https://github.com/tensorflow/lingvo/blob/3e19cc5ca20ecdd1fa748a7f555c6b62a1b193bc/lingvo/core/schedule.py#L305

    Args:
        lr_scale: Learning rate scale. See `schedule.adafactor` for details.
        warmup_steps: Number of warmup steps.
        ema_decay: EMA decay rate. See `optimizers.param_ema` for details.

    Returns:
        A learner config.
    """
    l2_regularizer_ignore_paths = [
        ".*norm/.*",
        "(.*/)?bias",
        "(.*/)?scale",
        "(.*/)?per_dim_scale/param",
        # Exclude BatchNorm stats.
        "(.*/)?moving_mean",
        "(.*/)?moving_variance",
    ]
    optimizer = config_for_function(optimizers.adam_optimizer).set(
        learning_rate=config_for_function(schedule.adafactor).set(
            scale=lr_scale, warmup_steps=warmup_steps, step_offset=1
        ),
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        l2_regularizer_weight=1e-6,
        l2_regularizer_per_param_scale=config_for_function(optimizers.per_param_scale_by_path).set(
            description="l2_regularizer_scale",
            scale_by_path=[(path, 0) for path in l2_regularizer_ignore_paths],
        ),
    )
    scale_norm = config_for_function(optimizers.clip_by_global_norm).set(max_norm=1.0)
    cfg: Learner.Config = Learner.default_config().set(
        optimizer=config_for_function(optimizers.chain).set(args=[scale_norm, optimizer])
    )
    if ema_decay is not None:
        cfg.ema.decay = ema_decay
    return cfg
