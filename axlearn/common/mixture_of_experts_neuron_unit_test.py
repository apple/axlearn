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

from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils_neuron import TestConfig
from axlearn.common.utils_neuron import get_training_configs

#jax.config.update('jax_platform_name', 'cpu')  # Do we need this ?

MODULE_UNIT_TEST_ATOL=1e-6
MODULE_UNIT_TEST_RTOL=1e-3

test_configs = get_training_configs(is_unit=True)
print(test_configs)

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

        @partial(jax.jit, out_shardings=cfg.out_shard_test) # cannot specify both backend and sharding together
        def test_fwd_call():
            test_output, _ = self._fwd_call(cfg.test_layer, cfg.test_state, cfg.test_inputs)
            return test_output

        @partial(jax.jit, out_shardings=cfg.out_shard_golden)
        def golden_fwd_call():
            golden_output, _ =  self._fwd_call(cfg.golden_layer, cfg.golden_state, cfg.golden_inputs)
            return golden_output

        with cfg.mesh_test:
            test_output = test_fwd_call()
        with cfg.mesh_golden:
            golden_output = golden_fwd_call()

        if cfg.conv_output != None:
            test_output = cfg.conv_output(test_output)
        
        # Transfer results to CPU before comparison
        self.assertNestedAllClose(jax.device_get(test_output), jax.device_get(golden_output))

    @parameterized.named_parameters(test_configs)
    def test_bwd_correctness(self, cfg: TestConfig):

        @partial(jax.jit, out_shardings=cfg.out_shard_test)
        def test_bwd_call():
            def loss_fn(state):
                test_output, _ = self._fwd_call(cfg.test_layer, state, cfg.test_inputs)
                return cfg.loss_fn(test_output)
            
            loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(cfg.test_state)
            return  loss, grads

        @partial(jax.jit, out_shardings=cfg.out_shard_golden)
        def golden_bwd_call():
            def loss_fn(state):
                golden_output, _ = self._fwd_call(cfg.golden_layer, state, cfg.golden_inputs)
                return cfg.loss_fn(golden_output)
            
            loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(cfg.golden_state)
            return loss, grads

        with cfg.mesh_test:
            test_loss, test_grads = test_bwd_call()
        with cfg.mesh_golden:
             golden_loss, golden_grads = golden_bwd_call()

        # Transfer results to CPU before comparison
        test_loss = jax.tree_map(jax.device_get, test_loss)
        golden_loss = jax.tree_map(jax.device_get, golden_loss)
        test_grads = jax.tree_map(jax.device_get, test_grads)
        golden_grads = jax.tree_map(jax.device_get, golden_grads)
        
        self.assertNestedAllClose(test_loss, golden_loss)
        self.assertNestedAllClose(test_grads, golden_grads)

if __name__ == "__main__":
    absltest.main()
