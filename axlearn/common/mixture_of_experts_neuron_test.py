# Copyright © 2024 Apple Inc.
#
# Some of the code in this file is adapted from:
#
# tensorflow/lingvo:
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Integration Test for mixture_of_experts.py"""
from functools import partial

import jax
from absl.testing import absltest, parameterized
from axlearn.common.mixture_of_experts import TopKGatingGather, TopKGating, TopKGatingGatherBlockwise
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils_neuron import TestCaseConfig
from axlearn.common.utils_neuron import get_training_configs
import os

# is_unit = os.getenv('IS_UNIT', 'false').lower() == 'true'
# is_blockwise = os.getenv('IS_BLOCKWISE', 'false').lower() == 'true'
# test_configs = get_training_configs(is_unit, is_blockwise)

# pylint: disable=no-self-use,protected-access
class TestImplOnCpu(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'cpu')
    
    def _fwd_call(self, layer, state, inputs):
        return F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state=state,
                inputs=inputs,
        )

    def helper_fwd(self, cfg):
        cfg.instantiate()
        cfg.print_summary()
        @partial(jax.jit, static_argnums=0)
        def test_fwd_call(test_layer, test_state, test_inputs):
            test_output, _ = self._fwd_call(test_layer, test_state, test_inputs)
            return test_output

        @partial(jax.jit, static_argnums=0)
        def golden_fwd_call(golden_layer, golden_state, golden_inputs):
            golden_output, _ =  self._fwd_call(golden_layer, golden_state, golden_inputs)
            return golden_output
        
        with cfg.test.mesh:
            test_output = test_fwd_call(cfg.test.layer, cfg.test.state, cfg.test.inputs)
        with cfg.golden.mesh:
            golden_output = golden_fwd_call(cfg.golden.layer, cfg.golden.state, cfg.golden.inputs)

        if cfg.conv_output != None:
            test_output = cfg.conv_output(test_output)
        
        # Transfer results to CPU before comparison
        self.assertNestedAllClose(jax.device_get(test_output), jax.device_get(golden_output),
                                  atol=cfg.test.atol, rtol=cfg.test.rtol)

    @parameterized.named_parameters(get_training_configs(test=TopKGatingGatherBlockwise, golden=TopKGatingGather, test_device="cpu", golden_device="cpu"))
    def test_fwd_blockwisegather_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_fwd(cfg)

    # @parameterized.named_parameters(get_training_configs(test=TopKGatingGatherBlockwise, golden=TopKGatingGather, test_device="cpu", golden_device="cpu"))
    # def test_fwd_bwd_correctness(self, cfg: TestCaseConfig):
    #     cfg.instantiate()
    #     cfg.print_summary()
    #     @partial(jax.jit, static_argnums=0)
    #     def test_bwd_call(test_layer, test_state, test_inputs):
    #         def loss_fn(state):
    #             output, aux = self._fwd_call(test_layer, state, test_inputs)
    #             return cfg.loss_fn(output), output  # Return both loss and output

    #         (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(test_state)
    #         return loss, grads, output

    #     @partial(jax.jit, static_argnums=0)
    #     def golden_bwd_call(golden_layer, golden_state, golden_inputs):
    #         def loss_fn(state):
    #             output, aux = self._fwd_call(golden_layer, state, golden_inputs)
    #             return cfg.loss_fn(output), output  # Return both loss and output

    #         (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(golden_state)
    #         return loss, grads, output

    #     with cfg.test.mesh:
    #         test_loss, test_grads, test_output = test_bwd_call(cfg.test.layer, cfg.test.state, cfg.test.inputs)
    #     with cfg.golden.mesh:
    #         golden_loss, golden_grads, golden_output = golden_bwd_call(cfg.golden.layer, cfg.golden.state, cfg.golden.inputs)

    #     #Transfer results to CPU before comparison
    #     test_loss = jax.tree_map(jax.device_get, test_loss)
    #     golden_loss = jax.tree_map(jax.device_get, golden_loss)
    #     test_grads = jax.tree_map(jax.device_get, test_grads)
    #     golden_grads = jax.tree_map(jax.device_get, golden_grads)
    #     test_output = jax.tree_map(jax.device_get, test_output)
    #     golden_output = jax.tree_map(jax.device_get, golden_output)
        
    #     # Compare losses
    #     self.assertNestedAllClose(test_loss, golden_loss, atol=cfg.test.atol, rtol=cfg.test.rtol)
        
    #     # Compare gradients
    #     self.assertNestedAllClose(test_grads, golden_grads, atol=cfg.test.atol, rtol=cfg.test.rtol)
        
    #     # Compare outputs
    #     self.assertNestedAllClose(test_output, golden_output, atol=cfg.test.atol, rtol=cfg.test.rtol)

class TestImplOnTrn(TestImplOnCpu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'neuron')
    
    @parameterized.named_parameters(get_training_configs(test=TopKGatingGatherBlockwise, golden=TopKGatingGather, test_device="neuron", golden_device="cpu"))
    def test_fwd_blockwisegather_vs_einsum(self, cfg):
        self.helper_fwd(cfg)
    
    def _fwd_call(self, layer, state, inputs):
        return F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state=state,
                inputs=inputs,
        )

if __name__ == "__main__":
    absltest.main()
