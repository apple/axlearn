# Copyright © 2023 Apple Inc.

"""Detection metric calculators."""
from typing import Optional, Union

import jax
import jax.random
import numpy as np

from axlearn.common.base_model import BaseModel
from axlearn.common.config import config_class
from axlearn.common.evaler import BaseMetricCalculator
from axlearn.common.metrics import WeightedScalar
from axlearn.common.module import Module
from axlearn.common.utils import NestedPartitionSpec, NestedTensor, Tensor, replicate_to_local_data
from axlearn.vision import coco_evaluator
from axlearn.vision.utils_visualization import visualize_detections


class COCOMetricCalculator(BaseMetricCalculator):
    """Accumulates model summaries over evaluation batches for COCO metric.

    This module creates a wrapper for axlearn.vision.coco_evaluator.COCOEvaluator.
    """

    @config_class
    class Config(BaseMetricCalculator.Config):
        """Configures COCOMetricCalculator."""

        per_category_metrics: bool = False  # Whether to return per category metrics or not.
        include_mask: bool = False  # Whether to include the instance mask eval or not.
        # A JSON file that stores annotations of the eval dataset. If `annotation_file` is None,
        # groundtruth annotations will be loaded from the dataloader.
        annotation_file: Optional[str] = None
        need_rescale_bboxes: bool = True
        # When visualize_detections is set to True, we display one image per validation batch in
        # the tensorboard along with groundtruth and prediction boxes.
        visualize_detections: bool = False

    def __init__(
        self,
        cfg: Config,
        *,
        parent: Optional[Module],
        model: BaseModel,
        model_param_partition_specs: NestedPartitionSpec,
    ):
        super().__init__(
            cfg, parent=parent, model=model, model_param_partition_specs=model_param_partition_specs
        )
        self._jit_predict = self._pjit(self._predict_in_pjit)
        self._coco_metric = coco_evaluator.COCOEvaluator(
            annotation_file=cfg.annotation_file,
            include_mask=cfg.include_mask,
            per_category_metrics=cfg.per_category_metrics,
            need_rescale_bboxes=cfg.need_rescale_bboxes,
        )

    def init_state(self, *, prng_key: Tensor, model_params: NestedTensor) -> NestedTensor:
        self._coco_metric.reset_states()
        return dict(prng_key=prng_key)

    def _update_coco_metric(
        self, *, input_batch: NestedTensor, model_outputs: NestedTensor
    ) -> Optional[Tensor]:
        labels = input_batch["labels"]
        coco_model_outputs = {
            "detection_boxes": model_outputs["detection_boxes"],
            "detection_scores": model_outputs["detection_scores"],
            "detection_classes": model_outputs["detection_classes"],
            "num_detections": model_outputs["num_detections"],
            "source_id": labels["groundtruths"]["source_id"],
            "image_info": input_batch["image_data"]["image_info"],
        }
        # Aggregate model outputs for each eval step.
        self._coco_metric.update_state(
            groundtruths=replicate_to_local_data(labels["groundtruths"]),
            predictions=replicate_to_local_data(coco_model_outputs),
        )

    def forward(
        self,
        input_batch: NestedTensor,
        *,
        model_params: NestedTensor,
        state: NestedTensor,
    ) -> dict[str, NestedTensor]:
        outputs = self._jit_predict(model_params, state["prng_key"], input_batch["image_data"])
        self._update_coco_metric(input_batch=input_batch, model_outputs=outputs["per_example"])

        visualized_detections = None
        if self.config.visualize_detections:
            visualized_detections = visualize_detections(
                images=np.array(replicate_to_local_data(input_batch["image_data"]["image"])),
                predictions=replicate_to_local_data(outputs["per_example"]),
                groundtruths=replicate_to_local_data(input_batch["labels"]["groundtruths"]),
            )

        return dict(
            state=dict(prng_key=outputs["replicated"]["prng_key"]),
            output={"image": visualized_detections},
        )

    def _predict_in_pjit(
        self,
        model_params: NestedTensor,
        prng_key: Tensor,
        input_batch: NestedTensor,
    ) -> dict[str, NestedTensor]:
        predict_key, next_key = jax.random.split(prng_key)
        model_outputs, model_output_collection = self._call_model(
            method="predict",
            model_params=model_params,
            prng_key=predict_key,
            input_batch=input_batch,
        )
        return dict(
            replicated=dict(prng_key=next_key, summaries=model_output_collection.summaries),
            per_example=model_outputs,
        )

    def get_summaries(
        self,
        *,
        model_params: Optional[NestedTensor] = None,
        state: Optional[NestedTensor] = None,
        all_forward_outputs: Optional[list[NestedTensor]] = None,
    ) -> dict[str, Union[WeightedScalar, np.ndarray]]:
        # Compute COCO metrics after aggregating outputs for all eval steps.
        metrics = self._coco_metric.result()

        if self.config.visualize_detections:
            images = np.array([forward_output["image"] for forward_output in all_forward_outputs])
            metrics["images"] = images

        return metrics
