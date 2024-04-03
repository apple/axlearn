"""An Apache Beam example pipeline to run batch inference jobs.


"""


from absl import app, flags, logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence

import warnings
warnings.filterwarnings("ignore")

# Define a custom model handler to load JAX models
class FlaxModelHandler(ModelHandler[Dict,
                                     PredictionResult,
                                     Any]):
    def __init__(
        self,
        model_path: str
    ):
        self._model_path = model_path

    def load_model(self) -> Any:
        """Loads and initializes a TrainState for processing."""
        init_rng = jax.random.key(0)
        learning_rate = 0.01
        momentum = 0.9
        cnn = CNN()
        initial_state = create_train_state(cnn, init_rng, learning_rate, momentum)
        file = FileSystems.open(self._model_path, 'rb')
        return from_state_dict(initial_state, pickle.load(file))

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
        # Loop each text string, and use a tuple to store the inference results.
        predictions = []
        for d in batch:
            prediction = pred_step(model, d)
            predictions.append(prediction)
        return [PredictionResult(x, y) for x, y in zip(batch, predictions)]

def run():
    # TODO: parse pipeline options

    # Define and run your Beam pipeline
    pipeline = beam.Pipeline(options=pipeline_options)

    with pipeline as p:
        p
        | "Create file pattern" >> beam.Create([args.input_tfrecord_pattern])
        | "Load data" >> tfrecordio.ReadAllFromTFRecord()
        | "Reshuffle after loading data" >> beam.Reshuffle()
        | "RunInferenceFlax" >> RunInference(FlaxModelHandler(state_dict_path))

if __name__ == "__main__":
    configure_logging(logging.INFO)
    run()