# Copyright © 2025 Apple Inc.

"""Golden file generator for neural_retrieval_test."""

import numpy as np
import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder
from transformers.models.dpr import modeling_dpr as hf_dpr

from axlearn.common.torch_utils import parameters_from_torch_layer
from axlearn.experiments.testdata.generate_parity_goldens._common import (
    save_golden,
    setup_determinism,
    to_numpy_tree,
)

MODULE_NAME = "axlearn.common.neural_retrieval_test"


def _make_dpr_config(model_dim=32, ff_dim=128, num_layers=3, num_heads=8, max_seq_len=12):
    return hf_dpr.DPRConfig(
        attention_probs_dropout_prob=0,
        hidden_dropout_prob=0,
        hidden_size=model_dim,
        intermediate_size=ff_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        max_position_embeddings=max_seq_len,
        pad_token_id=0,
        vocab_size=30522,
        hidden_act="gelu",
    )


def _generate_inputs(batch_size=2, max_seq_len=12):
    tokenized_text = np.random.randint(low=3, high=30522, size=[batch_size, max_seq_len - 1])
    eos_position = np.random.randint(low=1, high=max_seq_len - 1)
    tokenized_text_hf = tokenized_text.copy()
    tokenized_text_hf[:, eos_position + 1 :] = 0
    tokenized_text[:, eos_position + 1 :] = 0
    return tokenized_text_hf, np.expand_dims(tokenized_text, 1)


def generate():
    """Generate golden data for neural_retrieval_test (DPR question + context encoders)."""
    setup_determinism()

    hf_cfg = _make_dpr_config()

    # TestDPRQuestionEncoder.test_dpr
    hf_question_encoder = DPRQuestionEncoder(hf_cfg).eval()
    params = to_numpy_tree(parameters_from_torch_layer(hf_question_encoder))
    tokenized_text_hf, tokenized_text = _generate_inputs()

    with torch.no_grad():
        ref_outputs = hf_question_encoder.forward(torch.as_tensor(tokenized_text_hf))

    save_golden(
        MODULE_NAME,
        "test_dpr_question_encoder",
        {
            "params": params,
            "inputs": {"positive_input_ids": tokenized_text},
            "outputs": {"positive_embeddings": ref_outputs[0].unsqueeze(1).numpy()},
        },
    )

    # TestDPRContextEncoder.test_dpr — reset seed for independent inputs.
    np.random.seed(0)
    torch.manual_seed(0)

    hf_context_encoder = DPRContextEncoder(hf_cfg).eval()
    params = to_numpy_tree(parameters_from_torch_layer(hf_context_encoder))
    tokenized_text_hf, tokenized_text = _generate_inputs()

    with torch.no_grad():
        ref_outputs = hf_context_encoder.forward(torch.as_tensor(tokenized_text_hf))

    save_golden(
        MODULE_NAME,
        "test_dpr_context_encoder",
        {
            "params": params,
            "inputs": {"positive_input_ids": tokenized_text},
            "outputs": {"positive_embeddings": ref_outputs[0].unsqueeze(1).numpy()},
        },
    )
