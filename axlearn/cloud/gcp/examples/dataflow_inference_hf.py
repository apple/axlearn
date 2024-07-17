# Copyright © 2024 Google LLC

"""An Apache Beam example pipeline to run batch inference jobs with a HuggingFace model.

Reference: https://cloud.google.com/dataflow/docs/notebooks/
run_inference_huggingface#runinference_with_a_pretrained_model_from_hugging_face_hub

"""


import logging
import warnings
from typing import Dict, Tuple

# pylint: disable=import-error
import apache_beam as beam  # pytype: disable=import-error
import tensorflow as tf
from absl import app
from apache_beam.ml.inference.base import (  # pytype: disable=import-error
    KeyedModelHandler,
    PredictionResult,
    RunInference,
)
from apache_beam.ml.inference.huggingface_inference import (  # pytype: disable=import-error
    HuggingFaceModelHandlerKeyedTensor,
)
from apache_beam.options.pipeline_options import PipelineOptions  # pytype: disable=import-error
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

# pylint: enable=import-error


warnings.filterwarnings("ignore")


# Define how to preprocess input
def add_mask_to_last_word(text: str) -> Tuple[str, str]:
    """Replace the last word of sentence with <mask> and return
    the original sentence and the masked sentence."""
    text_list = text.split()
    masked = " ".join(text_list[:-2] + ["<mask>" + text_list[-1]])
    return text, masked


def tokenize_sentence(
    text_and_mask: Tuple[str, str], tokenizer
) -> Tuple[str, Dict[str, tf.Tensor]]:
    """Convert string examples to tensors."""
    text, masked_text = text_and_mask
    tokenized_sentence = tokenizer.encode_plus(masked_text, return_tensors="tf")

    return text, {k: tf.squeeze(v) for k, v in dict(tokenized_sentence).items()}


# Define how to postprocess output
class PostProcessor(beam.DoFn):
    """Processes the PredictionResult to get the predicted word.

    The logits are the output of the BERT Model. To get the word with the highest
    probability of being the masked word, take the argmax.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def process(self, element: Tuple[str, PredictionResult]) -> None:
        text, prediction_result = element
        inputs = prediction_result.example
        logits = prediction_result.inference["logits"]
        mask_token_index = tf.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[0]
        predicted_token_id = tf.math.argmax(logits[mask_token_index[0]], axis=-1)
        decoded_word = self.tokenizer.decode(predicted_token_id)
        print(f"Actual Sentence: {text}\nPredicted last word: {decoded_word}")
        print("-" * 80)


def main():
    # Create a model handler
    model_handler = HuggingFaceModelHandlerKeyedTensor(
        model_uri="stevhliu/my_awesome_eli5_mlm_model",
        model_class=TFAutoModelForMaskedLM,
        framework="tf",
        load_model_args={"from_pt": True},
        max_batch_size=1,
    )

    # Define your input
    text = [
        "The capital of France is Paris .",
        "It is raining cats and dogs .",
        "He looked up and saw the sun and stars .",
        "Today is Monday and tomorrow is Tuesday .",
        "There are 5 coconuts on this palm tree .",
    ]

    tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_eli5_mlm_model")
    pipeline_options = PipelineOptions(["--save_main_session"])

    with beam.Pipeline(options=pipeline_options) as beam_pipeline:
        tokenized_examples = (
            beam_pipeline
            | "CreateExamples" >> beam.Create(text)
            | "AddMask" >> beam.Map(add_mask_to_last_word)
            | "TokenizeSentence" >> beam.Map(lambda x: tokenize_sentence(x, tokenizer))
        )

        _ = (
            tokenized_examples
            | "RunInference" >> RunInference(KeyedModelHandler(model_handler))
            | "PostProcess" >> beam.ParDo(PostProcessor(tokenizer))
        )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    app.run(main)
