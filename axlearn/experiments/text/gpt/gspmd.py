# Copyright © 2024 Apple Inc.

"""Utilities to replicate configs from GSPMD.

Note that these configs are mainly intended for performance testing.

https://arxiv.org/abs/2105.04663
"""

from typing import Any, Dict

from axlearn.common.attention import PipelinedTransformerLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import config_for_function
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm
from axlearn.common.learner import Learner
from axlearn.common.optimizers import adafactor_optimizer
from axlearn.common.pipeline import BaseSchedule, GPipeSchedule, StreamSchedule
from axlearn.common.schedule import adafactor, adafactor_decay_rate
from axlearn.common.utils import HybridMeshShape
from axlearn.experiments.text.gpt.common import (
    SourceBuilder,
    evaler_config_dict,
    get_trainer_config_fn,
    make_config_name,
    mesh_shape_from_axes,
)
from axlearn.experiments.text.gpt.common import model_config as common_model_config
from axlearn.experiments.text.gpt.common import scaled_hidden_dim
from axlearn.experiments.trainer_config_utils import TrainerConfigFn

_VOCAB_SIZE = 32 * 1024
_MAX_SEQUENCE_LENGTH = 1024
_MAX_STEP = 100_000


def _model_config(schedule: BaseSchedule.Config) -> BaseModel.Config:
    """Constructs a model config for the given pipeline schedule."""
    emb_cfg: TransformerTextEmbeddings.Config = TransformerTextEmbeddings.default_config()
    # Only partitioning batch over 'pipeline' results in involuntary full remat, but still seems to
    # achieve better step time than partitioning over (('data', 'pipeline'), 'model').
    emb_cfg.token_emb.param_partition_spec = ("pipeline", "model")
    # See table 4 for pipelining configurations.
    stack_cfg: PipelinedTransformerLayer.Config = PipelinedTransformerLayer.default_config().set(
        num_stages=2,
        num_microbatches=16,
    )
    stack_cfg.pipeline.schedule = schedule
    return common_model_config(
        # See page 10 for model dims.
        num_layers=64,
        hidden_dim=128 * 32,
        num_heads=64,
        stack_cfg=stack_cfg,
        vocab_size=_VOCAB_SIZE,
        ffn_dim=scaled_hidden_dim(4),
        activation_fn="nn.relu",
        normalization=RMSNorm.default_config().set(forward_dtype=None),
        emb_cfg=emb_cfg,
    )


def _trainer_kwargs() -> Dict[str, Dict[str, Any]]:
    """Construct trainer kwargs for all configurations."""
    # pylint: disable=use-dict-literal

    learner_cfg = Learner.default_config().set(
        optimizer=config_for_function(adafactor_optimizer).set(
            learning_rate=config_for_function(adafactor).set(scale=1e-3),
            b1=0.9,
            b2=config_for_function(adafactor_decay_rate),
            multiply_by_parameter_scale=False,
            clipping_threshold=5,
            weight_decay=1e-3,
            weight_decay_scale_by_learning_rate_exponent=1,
        ),
    )
    common_kwargs = dict(
        learner_cfg=learner_cfg,
        train_batch_size=512,
        max_step=_MAX_STEP,
        mesh_shape=HybridMeshShape(
            # Note that while model=16 would be more compatible with the physical 4x4 mesh for
            # v5e-16, in practice model=8 achieves better step time even with the naive mesh.
            ici_mesh_shape=mesh_shape_from_axes(data=-1, model=8),
            dcn_mesh_shape=mesh_shape_from_axes(data=-1, pipeline=2),
        ),
    )

    return {
        # The 2x16x8 configuration can be run on 16 slices of v5e-16, similar to the 2x16x8 v3
        # configuration in table 4.
        make_config_name("gspmd", "16B", suffix="-2x16x8"): dict(
            model_cfg=_model_config(schedule=GPipeSchedule.default_config()),
            **common_kwargs,
        ),
        # Same as the 2x16x8 configuration but using "streaming" schedule.
        make_config_name("gspmd", "16B", suffix="-2x16x8-stream"): dict(
            model_cfg=_model_config(schedule=StreamSchedule.default_config()),
            **common_kwargs,
        ),
    }


def trainer_configs(
    train_input_source: SourceBuilder, eval_input_sources: SourceBuilder
) -> Dict[str, TrainerConfigFn]:
    """Returns a mapping from config_name to TrainerConfigFn's.

    Args:
        train_input_source: A callable (vocab_size, max_sequence_length) -> input source config.
        eval_input_soruces: A callable (vocab_size, max_sequence_length) -> eval input sources.
    """
    return {
        # pylint: disable-next=unexpected-keyword-arg,missing-kwoa
        config_name: get_trainer_config_fn(
            train_input_source=train_input_source(
                vocab_size=_VOCAB_SIZE,
                max_sequence_length=_MAX_SEQUENCE_LENGTH,
            ),
            evalers=evaler_config_dict(
                eval_input_sources(
                    vocab_size=_VOCAB_SIZE, max_sequence_length=_MAX_SEQUENCE_LENGTH
                ),
            ),
            **kwargs,
        )
        for config_name, kwargs in _trainer_kwargs().items()
    }
