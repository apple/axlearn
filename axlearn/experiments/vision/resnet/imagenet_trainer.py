# Copyright © 2023 Apple Inc.

"""ResNet on ImageNet trainer configs.

References:
https://arxiv.org/abs/1512.03385
https://arxiv.org/abs/1409.0575

Examples:

    ```
    # Run the training locally on CPU and fake inputs.
    mkdir -p /tmp/resnet_test;
    python3 -m axlearn.common.launch_trainer_main \
        --module=vision.resnet.imagenet_trainer --config=ResNet-Test \
        --trainer_dir=/tmp/resnet_test --data_dir=FAKE --jax_backend=cpu

    # Launch training on a v4-8 TPU, reading and writing from GCS.
    #
    # Notes:
    # * Training summaries and checkpoints will be emitted to --trainer_dir.
    # * The input dataset is expected to already exist in --data_dir.
    # * It's recommended to use the same dir for launch --output_dir and --trainer_dir.
    GS_ROOT=gs://my-bucket; \
    OUTPUT_DIR=${GS_ROOT}/$USER/experiments/resnet50-$(date +%F); \
    axlearn gcp launch --instance_type=tpu-v4-8 --output_dir=$OUTPUT_DIR -- \
        python3 -m axlearn.common.launch_trainer_main \
        --module=vision.resnet.imagenet_trainer --config=ResNet-50 \
        --trainer_dir=$OUTPUT_DIR --data_dir=${GS_ROOT}/tensorflow_datasets --jax_backend=tpu

    # Sample docker launch.
    GS_ROOT=gs://my-bucket; \
    DOCKER_REPO=us-docker.pkg.dev/my-repo/my-image; \
    OUTPUT_DIR=${GS_ROOT}/$USER/experiments/resnet50-$(date +%F); \
    axlearn gcp launch --instance_type=tpu-v5litepod-16 --output_dir=$OUTPUT_DIR \
        --bundler_type=docker \
        --bundler_spec=repo=${DOCKER_REPO} \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu -- \
        python3 -m axlearn.common.launch_trainer_main \
        --module=vision.resnet.imagenet_trainer --config=ResNet-50 \
        --trainer_dir=$OUTPUT_DIR --data_dir=${GS_ROOT}/tensorflow_datasets --jax_backend=tpu
    ```

"""
from typing import Optional

import jax.numpy as jnp

from axlearn.common import checkpointer, config, evaler, input_tf_data, schedule
from axlearn.common.evaler import SpmdEvaler
from axlearn.common.trainer import SpmdTrainer
from axlearn.experiments.trainer_config_utils import TrainerConfigFn
from axlearn.experiments.vision import imagenet
from axlearn.experiments.vision.resnet.common import learner_config, model_config
from axlearn.vision import resnet


def _get_config_fn(
    backbone: resnet.ResNet.Config,
    *,
    train_batch_size: int,
    ema_decay: Optional[float] = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> TrainerConfigFn:
    """Returns a TrainerConfigFn for training ResNet on ImageNet.

    Args:
        backbone: The ResNet backbone config.
        train_batch_size: Global training batch size.
        ema_decay: If set, the EMA of model params will be computed during training.
        compute_dtype: The precision used for both training and evaluation.

    Returns:
        A function that returns a ResNet trainer, consisting of:
        - Training input config to load examples from the train split.
        - A summary writer to save training summaries every 100 steps.
        - Two evaluator configs to run every 10 epochs and save summaries:
            * eval_train: the first 50K samples from the train split
                (or 2 fake batches if get_data_dir() = "FAKE").
            * eval_validation: the validation split.
        - A checkpointer config to save a ckpt every 10 epochs.
    """

    def config_fn():
        steps_per_epoch = imagenet.NUM_TRAIN_EXAMPLES // train_batch_size
        max_step = steps_per_epoch * 90
        eval_every_n_epochs = ckpt_every_n_epochs = 10

        cfg = SpmdTrainer.default_config().set(
            name="resnet_imagenet",
            train_dtype=compute_dtype,
            max_step=max_step,
        )
        # Construct model and training inputs.
        cfg.model = model_config().set(backbone=backbone, num_classes=imagenet.NUM_CLASSES)
        cfg.input = imagenet.input_config(split="train", global_batch_size=train_batch_size)
        # Construct evalers.
        cfg.evalers = {}
        for name, split in [("eval_train", "train[:50000]"), ("eval_validation", "validation")]:
            evaler_cfg = SpmdEvaler.default_config().set(
                input=imagenet.input_config(split=split, global_batch_size=80),
                eval_policy=config.config_for_function(evaler.every_n_steps_policy).set(
                    n=steps_per_epoch * eval_every_n_epochs,
                ),
                eval_dtype=compute_dtype,
            )
            # Explicitly disable shuffling during eval.
            input_tf_data.disable_shuffle_recursively(evaler_cfg)
            cfg.evalers[name] = evaler_cfg
        # Construct learner.
        cfg.learner = learner_config(
            learning_rate=config.config_for_function(schedule.cosine_with_linear_warmup).set(
                # Scale the initial learning rate with train batch size.
                peak_lr=0.1 * train_batch_size / 256,
                max_step=max_step,
                warmup_steps=steps_per_epoch * 5,
            ),
            ema_decay=ema_decay,
        )
        # Configure checkpointing and summaries.
        cfg.checkpointer.save_policy = config.config_for_function(
            checkpointer.every_n_steps_policy
        ).set(n=steps_per_epoch * ckpt_every_n_epochs)
        cfg.checkpointer.keep_every_n_steps = cfg.checkpointer.save_policy.n
        cfg.summary_writer.write_every_n_steps = 100
        return cfg

    return config_fn


def named_trainer_configs() -> dict[str, TrainerConfigFn]:
    """Returns a mapping from trainer config names to TrainerConfigFn's."""
    return {
        "ResNet-Test": _get_config_fn(
            resnet.ResNet.resnet18_config().set(hidden_dim=4, num_blocks_per_stage=[]),
            train_batch_size=2,
        ),
        "ResNet-Testb": _get_config_fn(
            resnet.ResNet.resnet18_config().set(hidden_dim=4, num_blocks_per_stage=[1]),
            train_batch_size=256,
        ),
        "ResNet-18": _get_config_fn(resnet.ResNet.resnet18_config(), train_batch_size=1024),
        "ResNet-34": _get_config_fn(resnet.ResNet.resnet34_config(), train_batch_size=1024),
        "ResNet-50": _get_config_fn(resnet.ResNet.resnet50_config(), train_batch_size=1024),
        "ResNet-50-ema": _get_config_fn(
            resnet.ResNet.resnet50_config(), train_batch_size=1024, ema_decay=0.9999
        ),
        "ResNet-101": _get_config_fn(resnet.ResNet.resnet101_config(), train_batch_size=1024),
        "ResNet-152": _get_config_fn(resnet.ResNet.resnet152_config(), train_batch_size=1024),
    }
