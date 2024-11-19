# Copyright © 2024 Apple Inc.

"""Tests fuji weight loading from llama."""

import os

import jax
import numpy as np
import pytest
import torch
from absl.testing import absltest, parameterized
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from axlearn.common import utils
from axlearn.common.causal_lm import Model
from axlearn.common.module import InvocationContext
from axlearn.common.module import functional as F
from axlearn.common.module import new_output_collection, set_current_context
from axlearn.common.param_converter import parameters_from_llama_3
from axlearn.common.test_utils import TestCase
from axlearn.common.update_transformation import ForwardOutputs
from axlearn.common.utils import NestedTensor
from axlearn.experiments.text.gpt import c4_trainer

dir_path = os.path.dirname(os.path.realpath(__file__))
# Use cpu for the test.
jax.config.update("jax_platform_name", "cpu")

# Parameters are based on https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json
config_dict_1b = {
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "pretraining_tp": 1,
    "tie_word_embeddings": True,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "torch_dtype": "bfloat16",
    "architectures": ["LlamaForCausalLM"],
}
# Parameters are based on https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json
config_dict_3b = {"hidden_size": 3072, "num_attention_heads": 24, "num_hidden_layers": 28}
# Parameters are based on https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
config_dict_8b = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "tie_word_embeddings": False,
}
# Parameters are based on https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json
config_dict_70b = {
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "tie_word_embeddings": False,
}


def compute_fuji_grad(prng_key, fuji: Model, state: NestedTensor, input_batch: NestedTensor):
    """Compute gradient of fuji model with a pseudo loss."""
    param_noise_key, forward_key = jax.random.split(prng_key, 2)

    def _forward(model_params: NestedTensor, *, inputs: NestedTensor) -> ForwardOutputs:
        model_params = fuji.apply_parameter_noise_recursively(
            inputs["param_noise_key"], model_params
        )
        model_output_collection = new_output_collection()
        context = InvocationContext(
            "root",
            parent=None,
            module=fuji,
            state=model_params,
            prng_key=inputs["forward_key"],
            output_collection=model_output_collection,
            is_training=True,
        )
        with set_current_context(context):
            _, aux = fuji(input_batch=inputs["input_batch"], return_aux=True)
            loss = aux["logits"].mean()
        return loss

    _, grads = jax.value_and_grad(_forward, has_aux=False)(
        state,
        inputs=dict(
            input_batch=input_batch,
            forward_key=forward_key,
            param_noise_key=param_noise_key,
        ),
    )
    return grads


def compute_llama_grad(llama, torch_ids, state):
    """Compute the gradient of llama using a pseudo loss function.

    and return the grad in the same nested dict as fuji."""
    llama_logits = llama(torch_ids).logits
    loss = llama_logits.mean()
    loss.backward()
    for p in llama.parameters():
        p.data = p.grad.data
    return parameters_from_llama_3(llama, state)


class FujiConvertStateTest(TestCase):
    @parameterized.parameters(
        dict(fuji_model_name="fuji-1B-v3-tiktoken"),
        dict(fuji_model_name="fuji-3B-v3-tiktoken"),
        dict(fuji_model_name="fuji-8B-v3-tiktoken"),
        dict(fuji_model_name="fuji-70B-v3-tiktoken"),
    )
    @pytest.mark.high_cpu
    def test_weight_loading(self, fuji_model_name):
        trainer_config_map = c4_trainer.named_trainer_configs()
        trainer_config_fn = trainer_config_map[fuji_model_name]
        trainer_config = trainer_config_fn()
        model_config = trainer_config.model
        model_config.set(name="fuji-test-model")
        fuji: Model = model_config.instantiate(parent=None)
        prng_key = jax.random.PRNGKey(0)
        state = fuji.initialize_parameters_recursively(prng_key=prng_key)
        config_dict = config_dict_1b
        if fuji_model_name == "fuji-3B-v3-tiktoken":
            config_dict.update(config_dict_3b)
        elif fuji_model_name == "fuji-8B-v3-tiktoken":
            config_dict.update(config_dict_8b)
        elif fuji_model_name == "fuji-70B-v3-tiktoken":
            config_dict.update(config_dict_70b)
        config = LlamaConfig(**config_dict)
        llama = LlamaForCausalLM._from_config(config)  # pylint: disable=W0212
        llama = llama.eval()
        ids = jax.random.randint(jax.random.PRNGKey(123), shape=(2, 2), minval=0, maxval=128256)
        torch_ids = torch.from_numpy(np.asarray(ids))
        state = parameters_from_llama_3(llama, state)
        input_batch = {"input_ids": ids}
        (_, aux), _ = F(
            fuji,
            is_training=False,
            prng_key=jax.random.PRNGKey(123),
            state=state,
            inputs={"input_batch": input_batch, "return_aux": True},
        )

        with torch.no_grad():
            output = llama(torch_ids)

        fuji_logits = np.asarray(aux["logits"])
        llama_logits = output.logits.numpy()

        # The difference is caused by the SDPA attention layer. The deeper the larger the error.
        if fuji_model_name == "fuji-1B-v3-tiktoken":
            atol = 2e-3
        elif fuji_model_name == "fuji-3B-v3-tiktoken":
            atol = 2e-2
        elif fuji_model_name == "fuji-8B-v3-tiktoken":
            atol = 2e-1
        elif fuji_model_name == "fuji-70B-v3-tiktoken":
            atol = 2.0
        else:
            atol = 2e-3
        assert np.allclose(fuji_logits, llama_logits, atol=atol), (
            f"{fuji_logits[0,0,:10]} != {llama_logits[0,0,:10]}, "
            f"{np.abs(fuji_logits - llama_logits).max()}"
        )

        prng_key, grad_prng_key = jax.random.split(prng_key, 2)
        fuji_grad = compute_fuji_grad(grad_prng_key, fuji, state, input_batch)
        llama_grad = compute_llama_grad(llama, torch_ids, state)
        self.assertNestedAllClose(fuji_grad, llama_grad, atol * 1e-2)


if __name__ == "__main__":
    with utils.numeric_checks(True):
        absltest.main()
