# Copyright © 2023 Apple Inc.

# pylint: disable=no-self-use
"""Tests for eval_detection.py"""

import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax.experimental.pjit import pjit

from axlearn.common import utils
from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import InstantiableConfig, config_class, config_for_function
from axlearn.common.module import REQUIRED, Module, Required
from axlearn.common.test_utils import TestCase
from axlearn.common.utils import NestedTensor, Tensor
from axlearn.vision.eval_detection import COCOMetricCalculator
from axlearn.vision.input_detection import DetectionInput, fake_detection_dataset


class DummyModel(BaseModel):
    """A dummy model directly outputs groundtruth."""

    @config_class
    class Config(BaseModel.Config):
        """Configures DummyModel."""

        need_rescale_bboxes: Required[BaseLayer.Config] = REQUIRED
        # input data config for fake predictions. This model simply outputs groundtruth boxes and
        # classes as the predicted outputs.
        input_cfg: Required[InstantiableConfig] = REQUIRED

    def __init__(self, cfg: Config, *, parent: Module):
        super().__init__(cfg, parent=parent)
        self.fake_results = next(iter(self.config.input_cfg.instantiate(parent=None)))

    def predict(self, input_batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.need_rescale_bboxes:
            image_scale = jnp.tile(input_batch["image_info"][:, 2:3, :], (1, 1, 2))
        else:
            image_scale = 1.0
        outputs = {
            "detection_boxes": self.fake_results["labels"]["groundtruths"]["boxes"] * image_scale,
            "detection_scores": jnp.ones(
                shape=self.fake_results["labels"]["groundtruths"]["classes"].shape
            ),
            "detection_classes": self.fake_results["labels"]["groundtruths"]["classes"],
            "num_detections": self.fake_results["labels"]["groundtruths"]["num_detections"],
        }
        return outputs

    def forward(self, input_batch: NestedTensor) -> tuple[float, NestedTensor]:
        del input_batch
        return 0.0, {}


class COCOMetricCalculatorTest(TestCase, parameterized.TestCase):
    """Tests COCOMetricCalculator."""

    def _input_config(self, is_training: bool, batch_size: int, image_size: int):
        cfg = DetectionInput.default_config().set(name="test", is_training=is_training)
        cfg.source.set(
            dataset_name="coco/2017",
            split="validation",
            train_shuffle_buffer_size=100,
        )
        cfg.processor.set(image_size=(image_size, image_size))
        cfg.batcher.set(global_batch_size=batch_size)
        return cfg

    @parameterized.parameters(False, True)
    def test_coco_metric_under_perfect_detection(self, need_rescale_bboxes):
        with jax.sharding.Mesh(
            jax.experimental.mesh_utils.create_device_mesh((1, 1)), ("data", "model")
        ):
            batch_size = 2
            image_size = 640
            input_cfg = self._input_config(
                is_training=False,
                batch_size=batch_size,
                image_size=image_size,
            )
            input_cfg.source = config_for_function(fake_detection_dataset)
            input_cfg.source.set(total_num_examples=batch_size)
            dataset = input_cfg.instantiate(parent=None)
            input_batch = next(iter(dataset))

            model = (
                DummyModel.default_config()
                .set(name="model", need_rescale_bboxes=need_rescale_bboxes, input_cfg=input_cfg)
                .instantiate(parent=None)
            )
            model_param_partition_specs = jax.tree.map(
                lambda spec: spec.mesh_axes, model.create_parameter_specs_recursively()
            )
            calculator: COCOMetricCalculator = (
                COCOMetricCalculator.default_config()
                .set(name="calculator", need_rescale_bboxes=need_rescale_bboxes)
                .instantiate(
                    parent=None,
                    model=model,
                    model_param_partition_specs=model_param_partition_specs,
                )
            )
            model_params = pjit(
                model.initialize_parameters_recursively,
                in_shardings=(None,),
                out_shardings=model_param_partition_specs,
            )(jax.random.PRNGKey(0))

            state = calculator.init_state(prng_key=jax.random.PRNGKey(0), model_params=model_params)
            for _ in range(5):
                forward_outputs = calculator.forward(
                    utils.host_to_global_device_array(input_batch),
                    model_params=model_params,
                    state=state,
                )
                state = forward_outputs["state"]
            summaries = calculator.get_summaries()

            self.assertEqual(summaries["AP"], 1.0)


if __name__ == "__main__":
    absltest.main()
