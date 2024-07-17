# Copyright © 2024 Google LLC

"""An Apache Beam example pipeline to run batch inference jobs using a model trained with AXLearn.

Command line options:
--module: the same module used for training
--config: the same config used for training
--trainer_dir: location of your checkpoints for inference

To debug locally:
$ docker run -it --mount type=bind,src=$HOME/.config/gcloud,dst=/root/.config/gcloud \
    --entrypoint /bin/bash ${DOCKER_REPO}/${DOCKER_IMAGE}:{DOCKER_TAG}
> python3 -m axlearn.cloud.gcp.examples.dataflow_inference_custom \
    --module=text.gpt.c4_trainer \
    --config=fuji-7B-single  \
    --trainer_dir='gs://.../checkpoints/step_xxx'

To use axlearn CLI:
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DataflowRunner \
    -- "'python3 -m axlearn.cloud.gcp.examples.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

To use GPUs for your job:
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DataflowRunner \
    --dataflow_spec=dataflow_service_options=\
    "worker_accelerator=type:nvidia-l4;count:1;install-nvidia-driver" \
    -- "'python3 -m axlearn.cloud.gcp.examples.dataflow_inference_custom \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --trainer_dir='gs://.../checkpoints/step_xxx' \
        '"

"""


import copy
import logging
from typing import Any, Dict, Sequence

# pylint: disable=import-error
import apache_beam as beam  # pytype: disable=import-error
import jax
from absl import app, flags
from absl.flags import argparse_flags
from apache_beam.ml.inference.base import (  # pytype: disable=import-error
    ModelHandler,
    PredictionResult,
    RunInference,
)
from apache_beam.options.pipeline_options import PipelineOptions  # pytype: disable=import-error

import axlearn.common.launch_trainer as trainer_utils
from axlearn.common import input_fake
from axlearn.common.inference import InferenceRunner, MethodRunner
from axlearn.common.utils import NestedTensor

# pylint: enable=import-error


class CustomModelHandler(ModelHandler[Dict, PredictionResult, Any]):
    """Defines how to load a custom checkpoint and run inference.

    The RunInference transform natively supports TF, PyTorch, HF pre-trained models.
    For JAX models, we can define a custom model handler like the example here.

    References:
    https://cloud.google.com/dataflow/docs/notebooks/run_inference_pytorch
    https://cloud.google.com/dataflow/docs/notebooks/run_custom_inference
    """

    # pylint: disable-next=super-init-not-called
    def __init__(self, flag_dict: Dict):
        # Store absl FLAGS in a flag dictionary to avoid pickling issues
        self._flag_dict = flag_dict

    def _flag_values_from_dict(self, flag_values: Dict) -> flags.FlagValues:
        # Avoid mutating global FLAGS.
        fv = copy.deepcopy(flags.FLAGS)
        for k, v in flag_values.items():
            try:
                fv.set_default(k, v)
            # pylint: disable-next=protected-access
            except flags._exceptions.UnrecognizedFlagError:
                # Ignore unrecognized flags from other modules
                pass
        fv.mark_as_parsed()
        return fv

    def load_model(self) -> MethodRunner:
        """Loads a pre-trained model in the desired type (MethodRunner in this case).
        Reference: https://github.com/apple/axlearn/blob/main/axlearn/common/inference.py#L54
        Returns an instance of MethodRunner.
        """
        # construct absl FlagValues from dict
        flag_values = self._flag_values_from_dict(self._flag_dict)

        trainer_config = trainer_utils.get_trainer_config(flag_values=flag_values)

        # get InferenceRunner Config from Trainer Config and instantiate InferenceRunner
        inference_runner_cfg = InferenceRunner.config_from_trainer(trainer_config)
        inference_runner_cfg.init_state_builder.set(dir=flag_values.trainer_dir)
        inference_runner = inference_runner_cfg.instantiate(parent=None)

        return inference_runner.create_method_runner(
            method="predict", prng_key=jax.random.PRNGKey(1)
        )

    def run_inference(
        self, batch: Sequence[NestedTensor], model: MethodRunner
    ) -> Sequence[MethodRunner.Output]:
        """Runs inferences on a batch of NestedTensors.
        NestedTensor: https://github.com/apple/axlearn/blob/main/axlearn/common/utils.py#L56

        Args:
            batch: A sequence of examples as NestedTensors.
            model: An instance of a MethodRunner.
            inference_args: Optional additional keyword arguments for inference.

        Returns:
            A list of method runner outputs.
        """
        logging.info("Running Inference...")
        output_list = []
        for el in batch:
            output_list.append(model(el))

        return output_list


class PostProcessFn(beam.DoFn):
    """Defines the transformations needed for post processing."""

    # pylint: disable-next=unused-argument
    def process(self, element: Any):
        # Add your desired post processing here
        logging.info("Inference finished.")
        yield None


def get_examples() -> Sequence[NestedTensor]:
    """Returns a list of fake input. You can edit this function to return your desired input.
    Fake input: https://github.com/apple/axlearn/blob/main/axlearn/common/input_fake.py#L49

    Returns:
        A list of examples of type FakeLmInput.
        Must be a Sequence since Beam expects a Sequence of examples.
        A Sequence of NestedTensor, Tensor, or other types should all work.
    """
    cfg = input_fake.FakeLmInput.default_config()
    cfg.is_training = False
    cfg.global_batch_size = 1
    cfg.total_num_batches = 3

    fake_input = input_fake.FakeLmInput(cfg)
    example_list = []
    for _ in range(cfg.total_num_batches):
        example_list.append(fake_input.next())

    return example_list


def parse_flags(argv):
    """Parse out Dataflow pipeline options in addition to the defined trainer-related absl flags.
    (can be found in axlearn/common/launch_trainer.py).
    Addition arguments are returned to the 'main' function by 'app.run'.
    """
    parser = argparse_flags.ArgumentParser(
        description="Parser to parse additional arguments other than defined ABSL flags."
    )
    parser.add_argument("--save_main_session", default=True)

    # The default pickler is dill and cannot pickle absl FlagValues. Use cloudpickle instead.
    parser.add_argument("--pickle_library", default="cloudpickle")

    # Assume all remaining unknown arguments are Dataflow Pipeline options
    known_args, pipeline_args = parser.parse_known_args(argv[1:])
    if known_args.save_main_session is True:
        pipeline_args.append("--save_main_session")
    pipeline_args.append(f"--pickle_library={known_args.pickle_library}")
    return pipeline_args


def main(args):
    absl_flags = flags.FLAGS

    # get pipeline input
    pipeline_input = get_examples()

    # run pipeline
    pipeline_options = PipelineOptions(args)

    pipeline = beam.Pipeline(options=pipeline_options)

    with pipeline as p:
        # pylint: disable-next=unused-variable
        result = (
            p
            | "CreateInput" >> beam.Create(pipeline_input)
            | "RunInference" >> RunInference(CustomModelHandler(absl_flags.flag_values_dict()))
            | "PrintOutput" >> beam.ParDo(PostProcessFn())
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main, flags_parser=parse_flags)
