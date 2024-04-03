"""An Apache Beam example pipeline to run batch inference jobs using a model trained with AXLearn.

To debug locally:
$ python3 -m axlearn.cloud.gcp.dataflow_test

To launch the job locally and run locally:
$ axlearn gcp dataflow start \
    --bundler_spec=dockerfile=Dockerfile \
    --bundler_spec=repo=${DOCKER_REPO} \
    --bundler_spec=image=${DOCKER_IMAGE} \
    --bundler_spec=target=dataflow \
    --bundler_spec=allow_dirty=True \
    --dataflow_spec=runner=DirectRunner \
    -- "'rm -r /tmp/output_dir; \
        python3 -m axlearn.cloud.gcp.dataflow_test \
        --module=text.gpt.c4_trainer \
        --config=fuji-7B-single \
        --checkpoint_path= \
        '"

To launch the job locally and run on GCP Dataflow:
$

To launch the job on GCE VM and run on GCP Dataflow:
$


"""


from absl import app, flags
import logging
import argparse
import warnings

import axlearn.cloud.gcp.dataflow_utils as df_utils
import jax
import pickle
from flax.serialization import to_state_dict, from_state_dict
from google.cloud import storage

from axlearn.common.inference import InferenceRunner, MethodRunner
import axlearn.common.launch_trainer as trainer_utils

from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult


warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

class FlaxModelHandler(ModelHandler[Dict,PredictionResult,Any]):
    """Defines how to load a model and run inference"""

    def __init__(
        self,
        model_path: str
    ):
        self._model_path = model_path

    def load_model(self) -> Any:
        # get Trainer Config from flag values
        module_config = trainer_utils.get_trainer_config(flag_values=FLAGS)

        # get InferenceRunner Config from Trainer Config and instantiate InferenceRunner
        inference_runner_cfg = InferenceRunner.config_from_trainer(module_config)
        inference_runner_cfg.init_state_builder.set(dir=FLAGS.trainer_dir)
        inference_runner = InferenceRunner(cfg=inference_runner_cfg,parent=None)
        logging.info(type(inference_runner))
        return inference_runner

    def run_inference(
        self,
        batch: Sequence[Dict],
        model: Any,
        inference_args: Optional[Dict[str, Any]] = None
    ) -> Iterable[PredictionResult]:
        """Runs inferences on a batch of dictionaries.

        Args:
          batch: A sequence of examples as dictionaries.
          model: A Flax TrainState
          inference_args: Any additional arguments for an inference.

        Returns:
          An Iterable of type PredictionResult.
        """
        # TODO: inference.py ln 289
        logging.info("RUNNING INFERENCE")
        quit()
        predictions = []
        for d in batch:
            prediction = df_utils.pred_step(model, d)
            predictions.append(prediction)
        return [PredictionResult(x, y) for x, y in zip(batch, predictions)]

def get_examples():
    """Get pipeline input. Can edit this function to load examples from GCS."""
    examples = ["This is an example input for inference"]
    return examples

def main(argv=None, save_main_session=True):
    # TODO: parse out unknown args as pipeline options. Currently we're only passing in the exact required flags

    pipeline_input = get_examples()
    logging.info(f"EXAMPLES:{pipeline_input}")

    # run pipeline
    # pipeline_options = PipelineOptions(pipeline_args, number_of_worker_harness_threads=1)

    pipeline = beam.Pipeline()

    with pipeline as p:
        (p
        | "Create file pattern" >> beam.Create(pipeline_input)
        | "RunInferenceFlax" >> RunInference(FlaxModelHandler(FLAGS.trainer_dir))
        )
    # TODO: provide examples for huggingface model handler too

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main)