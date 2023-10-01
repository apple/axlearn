# Copyright © 2023 Apple Inc.

"""Common ResNet config builders.

Reference:
https://arxiv.org/abs/1512.03385
"""

import math
from typing import Optional

from axlearn.common import config, learner, optimizers, param_init, schedule
from axlearn.vision import image_classification


def learner_config(
    *, learning_rate: schedule.Schedule, ema_decay: Optional[float] = None
) -> learner.Learner.Config:
    """Constructs optimization configs for ResNet.

    Args:
        learning_rate: The learning rate schedule.
        ema_decay: If set, the EMA of model params will be computed during training.

    Returns:
        The learner config.
    """
    per_param_decay = config.config_for_function(optimizers.per_param_scale_by_path).set(
        description="weight_decay_scale",
        scale_by_path=[
            (".*norm.*", 0),  # Exclude the norm parameters from regularization.
        ],
    )
    optimizer = config.config_for_function(optimizers.sgd_optimizer).set(
        learning_rate=learning_rate,
        decouple_weight_decay=False,  # Set to False to match Torch behavior.
        momentum=0.9,
        weight_decay=1e-4,
        weight_decay_per_param_scale=per_param_decay,
    )
    cfg = learner.Learner.default_config().set(optimizer=optimizer)
    if ema_decay:
        cfg.ema.decay = ema_decay
    return cfg


def model_config() -> image_classification.ImageClassificationModel.Config:
    """Constructs an image classification model config.

    The caller should configure a backbone, as well as the number of classes, e.g.:
        ```
        model_config().set(backbone=ResNet.resnet18_config(), num_classes=100)
        ```

    Returns:
        An image classification model config.
    """
    cfg = image_classification.ImageClassificationModel.default_config()
    cfg.param_init = param_init.DefaultInitializer.default_config().set(
        init_by_param_name={
            param_init.PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
                # Equivalent to kaiming_normal_(mode='fan_out', nonlinearity='relu').
                fan="fan_out",
                distribution="normal",
                scale=math.sqrt(2),
            )
        }
    )
    return cfg
