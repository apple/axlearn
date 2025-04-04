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
"""Unit Test for mixture_of_experts.py"""
from functools import partial

import jax
from absl.testing import absltest, parameterized

from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils_neuron import TestConfig
from axlearn.common.utils_neuron import get_training_configs

test_configs = get_training_configs(is_unit=True)

# pylint: disable=no-self-use,protected-access
class TestImplCorrectnessUnit(TestCase):

    def _fwd_call(self, layer, state, inputs):
        return F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state=state,
                inputs=inputs,
        )

    @parameterized.named_parameters(test_configs)
    def test_fwd_correctness(self, cfg: TestConfig):

        cfg.instantiate()

        @partial(jax.jit, static_argnums=0) 
        def test_fwd_call(test_layer, test_state, test_inputs):
            test_output, _ = self._fwd_call(test_layer, test_state, test_inputs)
            return test_output

        @partial(jax.jit, static_argnums=0)
        def golden_fwd_call(golden_layer, golden_state, golden_inputs):
            golden_output, _ =  self._fwd_call(golden_layer, golden_state, golden_inputs)
            return golden_output

        with cfg.mesh_test:
            test_output = test_fwd_call(cfg.test_layer, cfg.test_state, cfg.test_inputs)
        with cfg.mesh_golden:
            golden_output = golden_fwd_call(cfg.golden_layer, cfg.golden_state, cfg.golden_inputs)

        if cfg.conv_output != None:
            test_output = cfg.conv_output(test_output)
        
        # Transfer results to CPU before comparison
        self.assertNestedAllClose(jax.device_get(test_output), jax.device_get(golden_output),
                                  atol=cfg.test.tol["atol"], rtol=cfg.test.tol["rtol"])

    @parameterized.named_parameters(test_configs)
    def test_bwd_correctness(self, cfg: TestConfig):

        cfg.instantiate()

        @partial(jax.jit, static_argnums=0) 
        def test_bwd_call(test_layer, test_state, test_inputs):
            def loss_fn(state):
                test_output, _ = self._fwd_call(test_layer, state, test_inputs)
                return cfg.loss_fn(test_output)
            
            loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(test_state)
            return  loss, grads

        @partial(jax.jit, static_argnums=0)
        def golden_bwd_call(golden_layer, golden_state, golden_inputs):
            def loss_fn(state):
                golden_output, _ = self._fwd_call(golden_layer, state, golden_inputs)
                return cfg.loss_fn(golden_output)
            
            loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(golden_state)
            return loss, grads

        with cfg.mesh_test:
            test_loss, test_grads = test_bwd_call(cfg.test_layer, cfg.test_state, cfg.test_inputs)
        with cfg.mesh_golden:
             golden_loss, golden_grads = golden_bwd_call(cfg.golden_layer, cfg.golden_state, cfg.golden_inputs)

        # Transfer results to CPU before comparison
        test_loss = jax.tree_map(jax.device_get, test_loss)
        golden_loss = jax.tree_map(jax.device_get, golden_loss)
        test_grads = jax.tree_map(jax.device_get, test_grads)
        golden_grads = jax.tree_map(jax.device_get, golden_grads)
        
        self.assertNestedAllClose(test_loss, golden_loss,
                                  atol=cfg.test.tol["atol"], rtol=cfg.test.tol["rtol"])
        self.assertNestedAllClose(test_grads, golden_grads,
                                  atol=cfg.test.tol["atol"], rtol=cfg.test.tol["rtol"])

if __name__ == "__main__":
    absltest.main()
