# Copyright © 2023 Apple Inc.

"""Base multistream model layer and supporting functions.

input_batch: A dictionary with the following structures:
    input_batch:
        *input:
            *input_item: Tensor representing the input_item.
        *layer_name:
            *layer_output: Tensor represents the outputs for this layer.
input_batch should only allow *insertation*.
Here, layer_name = encoder_network_name & fusion_layer_name.

Consider we define a Model with "text_encoder" and "image_encoder" as stream encoders.
The "cross_entropy_loss" is defined as a fusion network.
Under this example, the input_batch can be:
    *"input":
        *"image": Tensor representing the images.
        *"text": Tensor representing the text.
    *"text_encoder":
        *"text_emb": Tensor representing the text embeddings.
    *"image_encoder":
        *"image_emb": Tensor representing the image embeddings.

The losses dictionary can be:
    *"cross_entropy_loss": A scalar Tensor representing the cross-entropy loss.

Each encoder/fusion_network consumes items from input_batch and updates the input_batch
    with the new output.
"""
from collections import defaultdict
from typing import Optional

from axlearn.common.base_layer import BaseLayer
from axlearn.common.base_model import BaseModel
from axlearn.common.config import config_class
from axlearn.common.module import Module
from axlearn.common.utils import NestedTensor, Tensor


class StreamEncoder(BaseLayer):
    """Stream encoder class.

    The stream encoder should encode single input stream into an embedding.
    """

    def forward(self, input_batch: NestedTensor) -> NestedTensor:
        """Forward function for the stream encoder.

        The forward function should contain:
            1. Rename the item in input_batch to fit the input of the specific
                encoder.
            2. Call the forward function of the encoder.
            3. Return encoder_outputs. MultiStreamModel will put
                encoder_outputs in input_batch[encoder_name].

        Args:
            input_batch: See the input batch definition section.

        Outputs:
            A Nested Tensor representing encoder output.
        """
        raise NotImplementedError(type(self))


class FusionNetwork(BaseLayer):
    """Fusion network class.

    The fusion network should take multiple inputs and calculate one loss or
        embeddings.
    """

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        """Forward function for the fusion network.

        The forward function should contain:
            If the encoder is defined:
                1. Rename the item in input_batch to fit the input of the specific
                    encoder.
                2. Call the forward function of the encoder.
            3. Calculate the loss (loss=0, if this is purely an encoder function).
            4. Return (loss, fusion_outputs). MultiStreamModel will put
                fusion_outputs in input_batch[fusion_name].

        Noted that each FusionNetwork should only calculate one loss.

        Args:
            input_batch: See the input batch definition section.

        Outputs:
            A tuple containing:
                A scalar Tensor containing the loss.
                A NestedTensor representing fusion network output.
        """
        raise NotImplementedError(type(self))


class MultiStreamModel(BaseModel):
    """A multi-stream model used for multi-tower & cross-encoder model."""

    @config_class
    class Config(BaseLayer.Config):
        # Keys in stream_encoder and fusion_network should be strictly disjointed.
        stream_encoder: dict[str, StreamEncoder.Config] = {}
        fusion_network: dict[str, FusionNetwork.Config] = {}
        loss_weights: dict[str, float] = defaultdict(lambda: 1)

    def __init__(self, cfg: Config, *, parent: Optional[Module]):
        super().__init__(cfg, parent=parent)
        cfg = self.config

        # Check for duplicated key names.
        deduplicated_keys = list(cfg.stream_encoder.keys())
        deduplicated_keys.extend(list(cfg.fusion_network.keys()))
        assert len(deduplicated_keys) == len(set(deduplicated_keys))

        # Set up the stream encoder.
        self._stream_encoder = {}
        for stream_encoder_name, stream_encoder_cfg in cfg.stream_encoder.items():
            self._stream_encoder[stream_encoder_name] = self._add_child(
                stream_encoder_name, self._overwrite_config(stream_encoder_name, stream_encoder_cfg)
            )
        # Set up the fusion network.
        self._fusion_network = {}
        for fusion_network_name, fusion_network_cfg in cfg.fusion_network.items():
            self._fusion_network[fusion_network_name] = self._add_child(
                fusion_network_name, self._overwrite_config(fusion_network_name, fusion_network_cfg)
            )

    def _overwrite_config(  # pylint: disable=unused-argument,no-self-use
        self, child_name: str, child_config: Config
    ) -> Config:
        """Override the child module config.

        Args:
            child_name: A string for the child name.
            child_config: A child config.

        Returns:
            An overwritten configuration.
        """
        return child_config

    def _aggregate_loss(self, losses: dict[str, Tensor]) -> Tensor:
        """Aggregating the loss from losses dictionary."""
        loss = 0
        for fusion_name in self._fusion_network:
            loss += losses[fusion_name] * self.config.loss_weights[fusion_name]
        return loss

    def _forward_all_stream_encoder(self, input_batch: NestedTensor) -> NestedTensor:
        """Call the forward function of all the stream encoders."""
        for encoder_name in self._stream_encoder:  # pylint: disable=consider-using-dict-items
            assert encoder_name not in input_batch
            input_batch[encoder_name] = self._stream_encoder[encoder_name](input_batch["input"])
        return input_batch

    def forward_single_stream_encoder(
        self, input_batch: NestedTensor, encoder_name: str
    ) -> NestedTensor:
        """Calls the forward function of a specific stream encoders.

        This function is intended to be used with run_inference.py.
        """
        if encoder_name not in self._stream_encoder:
            raise ValueError(f"{encoder_name} has not been found in stream_encoder.")
        return self._stream_encoder[encoder_name](input_batch["input"])

    def predict(self, input_batch: NestedTensor) -> NestedTensor:
        """Predict function for the multi-stream model.

        The predict function should at least contain:
            Calling the encoder forward function.

        Args:
            input_batch: See the input batch definition section.

        Outputs:
            A NestedTensor representing additional features.

        TODO(bwzhang@) [Feature Request] support calling specific encoders/networks.
        """
        input_batch = self._forward_all_stream_encoder(input_batch)
        return input_batch

    def forward(self, input_batch: NestedTensor) -> tuple[Tensor, NestedTensor]:
        """Forward function for the multi-stream model.

        The forward function should contain:
            1. Calling the encoder forward function.
            2. Calling the fusion network forward function to calculate the loss.
            3. Calling the self._aggregate_loss function.

        Args:
            input_batch: See the input batch definition section.

        Outputs:
            A tuple containing:
                A scalar Tensor representing the total loss.
                A NestedTensor representing additional metrics (currently none).
        """
        # Disentangling the forward and the predict is based on the following thoughts:
        # User might override the predict function for some specific tasks, the predict
        #       function could produce redundant outputs that might interfere the _fusion_network.
        input_batch = self._forward_all_stream_encoder(input_batch)
        losses = {}
        for fusion_name in self._fusion_network:  # pylint: disable=consider-using-dict-items
            assert fusion_name not in input_batch
            loss, input_batch[fusion_name] = self._fusion_network[fusion_name](input_batch)
            self.add_summary(f"loss_{fusion_name}", loss)
            losses[fusion_name] = loss
        return self._aggregate_loss(losses), {}
