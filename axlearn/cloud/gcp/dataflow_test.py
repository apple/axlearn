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
import seqio

import axlearn.cloud.gcp.dataflow_utils as df_utils
import jax
import pickle
from flax.serialization import to_state_dict, from_state_dict
from google.cloud import storage
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
from axlearn.common.config import config_for_function
import axlearn.common.input_tf_data as input_tf_data
import axlearn.common.input_text as input_text

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

        # create Method Runner only once
        method_runner = inference_runner.create_method_runner(method='predict')
        return method_runner

    def run_inference(
        self,
        batch: Sequence[NestedTensor],
        model: MethodRunner,
        inference_args: Optional[Dict[str, Any]] = None
    ):
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
        output = model(batch)
        logging.info(f'OUTPUT: {output}')

        return output

def get_examples2() -> Sequence[NestedTensor]:
    filenames = ["gs://axlearn-public/tensorflow_datasets/c4/en/3.0.1/c4-validation.tfrecord-00000-of-00008", "gs://axlearn-public/tensorflow_datasets/c4/en/3.0.1/c4-validation.tfrecord-00001-of-00008"]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    examples = []

    tokenized_dataset = seqio.preprocessors.tokenize(raw_dataset, output_features={"text": tf.string})
    print(tokenized_dataset)
    for example in tokenized_dataset:
        print(type(example))
        print(example)
        examples.append(example)
    return examples


def get_examples() -> Sequence[NestedTensor]:
    """
    for raw_record in raw_dataset.take(2):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature
        for key, value in features.items():
            if key == "text":
                print(key,value)
                print(value.bytes_list.value)
        examples.append(example)
    """
    source = config_for_function(input_tf_data.tfds_dataset).set(
        dataset_name="c4",
        split="train",
    )
    processor = config_for_function(input_text.tokenize).set(
        output_features="text"
    )
    ds_fn = input_tf_data.with_processor(
        source=source,
        processor=processor,
        is_training=False,
    )
    for example in ds_fn():
        print(type(example))
        print(example)
        examples.append(example)
    return examples

def main(argv=None, save_main_session=True):
    # TODO: parse out unknown args as pipeline options. Currently we're only passing in the exact required flags

    pipeline_input = get_examples()
    quit()

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