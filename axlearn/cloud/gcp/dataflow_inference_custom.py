"""An Apache Beam example pipeline to run batch inference jobs using a model trained with AXLearn.

To debug locally:
$ python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
--module=text.gpt.c4_trainer \ (the same module used for training)
--config=fuji-7B-single  \ (the same config used for training)
--trainer_dir='gs://.../checkpoints/step_xxx' (location of your checkpoints for inference)


To launch the job locally and run locally:
$ DOCKER_REPO=
$ DOCKER_IMAGE=
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DirectRunner \
    -- "'python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

To launch the job locally and run on GCP Dataflow:
$ DOCKER_REPO=
$ DOCKER_IMAGE=
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DataflowRunner \
    -- "'python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

To use GPUs for your job:
$ DOCKER_REPO=
$ DOCKER_IMAGE=
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DataflowRunner \
    --dataflow_spec=dataflow_service_options="worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    -- "'python3 -m axlearn.cloud.gcp.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

To launch the job on GCE VM and run on GCP Dataflow:
$


"""


from absl import app, flags
from absl.flags import argparse_flags
import logging
import argparse
import warnings
import jax
import sys
import tensorflow as tf

from axlearn.common.inference import InferenceRunner, MethodRunner
import axlearn.common.launch_trainer as trainer_utils
from axlearn.common.utils import (
    DataPartitionType,
    NestedPartitionSpec,
    NestedTensor,
    PartitionSpec,
    Tensor,
    TensorSpec,
)
import axlearn.common.input_fake as input_fake

from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence


import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult


warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

class CustomModelHandler(ModelHandler[Dict,PredictionResult,Any]):
    """Defines how to load a model and run inference"""

    def load_model(self) -> MethodRunner:
        """Loads a pre-trained model in the desired type (MethodRunner in this case).
        Reference: https://github.com/apple/axlearn/blob/main/axlearn/common/inference.py#L54

        Returns:
          An instance of MethodRunner.
        """
        # get Trainer Config from flag values
        module_config = trainer_utils.get_trainer_config(flag_values=FLAGS)

        # get InferenceRunner Config from Trainer Config and instantiate InferenceRunner
        inference_runner_cfg = InferenceRunner.config_from_trainer(module_config)
        inference_runner_cfg.init_state_builder.set(dir=FLAGS.trainer_dir)
        inference_runner = InferenceRunner(cfg=inference_runner_cfg,parent=None)

        # create Method Runner only once
        method_runner = inference_runner.create_method_runner(method='predict',prng_key=jax.random.PRNGKey(1))
        return method_runner

    def run_inference(
        self,
        batch: Sequence[NestedTensor],
        model: MethodRunner,
        inference_args: Optional[Dict[str, Any]] = None
    ):
        """Runs inferences on a batch of NestedTensors.
        NestedTensor reference: https://github.com/apple/axlearn/blob/main/axlearn/common/utils.py#L56

        Args:
          batch: A sequence of examples as NestedTensors.
          model: An instance of a MethodRunner.
          inference_args: Any additional arguments for an inference.

        Returns:
          A list of type MethodRunner.Output.
        """
        logging.info("RUNNING INFERENCE")
        output_list = []
        for el in batch:
            output_list.append(model(el))

        return output_list


def get_examples() -> Sequence[NestedTensor]:
    """Returns a list of fake input. You can edit this function to return your desired input.
    Fake input reference: https://github.com/apple/axlearn/blob/main/axlearn/common/input_fake.py#L49

    Returns:
        A list of examples of type FakeLmInput. Must be a Sequence since Beam expects a Sequence of examples.
        A Sequence of NestedTensor, Tensor, or other types should all work.
    """
    cfg = input_fake.FakeLmInput.default_config()
    cfg.is_training = False
    cfg.global_batch_size = 1
    cfg.total_num_batches = 1

    fake_input = input_fake.FakeLmInput(cfg)
    example_list = []
    for _ in range(cfg.total_num_batches):
        example_list.append(fake_input.__next__())

    return example_list

def parse_flags(argv):
    """Parse out arguments in addition to the defined absl flags (can be found in axlearn/common/launch_trainer.py).
    Addition arguments are returned to the 'main' function by 'app.run'.
    """
    parser = argparse_flags.ArgumentParser(description='Parser to parse additional arguments other than defiend ABSL flags.')
    # Assume all remaining unknown arguments are Dataflow Pipeline options
    known_args, pipeline_args = parser.parse_known_args(argv[1:])
    return pipeline_args

def main(args):
    pipeline_input = get_examples()

    # run pipeline
    pipeline_options = PipelineOptions(args)
    pipeline = beam.Pipeline(options=pipeline_options)

    with pipeline as p:
        (p
        | "CreateInput" >> beam.Create(pipeline_input)
        | "RunInference" >> RunInference(CustomModelHandler())
        | "PrintOutput" >> beam.Map(print)
        )

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main, flags_parser=parse_flags)