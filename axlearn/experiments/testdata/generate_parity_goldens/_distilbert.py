# Copyright © 2025 Apple Inc.

"""Golden file generator for distilbert_test."""

import numpy as np
import torch
from transformers.models.distilbert import configuration_distilbert as hf_distilbert_config
from transformers.models.distilbert import modeling_distilbert as hf_distilbert

from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.experiments.testdata.generate_parity_goldens._common import (
    save_golden,
    setup_determinism,
    to_numpy_tree,
)

MODULE_NAME = "axlearn.common.distilbert_test"


def generate():
    """Generate golden data for distilbert_test.TestDistilBertModel.test_distilbert."""
    setup_determinism()

    model_dim = 32
    ff_dim = 64
    num_layers = 1
    num_heads = 8
    batch_size = 2
    max_seq_len = 12

    hf_cfg = hf_distilbert_config.DistilBertConfig(
        attention_dropout=0,
        dropout=0,
        qa_dropout=0,
        seq_classif_dropout=0,
        dim=model_dim,
        hidden_dim=ff_dim,
        n_layers=num_layers,
        n_heads=num_heads,
        max_position_embeddings=max_seq_len,
        pad_token_id=0,
        vocab_size=30522,
        activation="gelu",
    )
    hf_model = hf_distilbert.DistilBertModel(hf_cfg).eval()

    params = to_numpy_tree(parameters_from_torch_layer(hf_model))

    # Generate inputs (matches generate_random_tokenized_text in the test).
    tokenized_text = np.random.randint(low=3, high=30522, size=[batch_size, max_seq_len - 1])
    eos_position = np.random.randint(low=1, high=max_seq_len - 1)
    tokenized_text_hf = tokenized_text.copy()
    tokenized_text_hf[:, eos_position + 1 :] = 0
    tokenized_text[:, eos_position + 1 :] = 0

    with torch.no_grad():
        ref_outputs = hf_model.forward(
            input_ids=torch.as_tensor(tokenized_text_hf),
            attention_mask=torch.as_tensor(tokenized_text_hf != 0).int(),
        )

    save_golden(
        MODULE_NAME,
        "test_distilbert",
        {
            "params": params,
            "inputs": {"input_ids": tokenized_text},
            "outputs": {"text_embeddings": ref_outputs[0][:, 0:1].detach().numpy()},
        },
    )
