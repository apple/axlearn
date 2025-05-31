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
import unittest
import os
import jax
import math
import numpy as np
import jax.numpy as jnp
from jax_neuronx.experimental import debug_callback

from absl.testing import absltest, parameterized
from axlearn.common.mixture_of_experts import TopKGatingGather, TopKGating, TopKGatingGatherBlockwise, TopKGatingGatherBlockwiseV2
from axlearn.common.module import functional as F
from axlearn.common.test_utils import TestCase
from axlearn.common.utils_neuron import TestCaseConfig, create_test_config, get_training_configs


TEST_SUITE = os.environ.get("TEST_SUITE", 'presubmit').lower()

class LayerTestCase(TestCase):
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
        # @debug_callback

        @jax.jit
        def test_fwd_call(test_state, test_inputs):
            test_output, _ = self._fwd_call(cfg.test.layer, test_state, test_inputs)
            return test_output

        @jax.jit
        def golden_fwd_call(golden_state, golden_inputs):
            golden_output, _ =  self._fwd_call(cfg.golden.layer, golden_state, golden_inputs)
            return golden_output
        
        with cfg.test.mesh:
            test_output = test_fwd_call(cfg.test.state, cfg.test.inputs)

        if cfg.golden:
            with cfg.golden.mesh:
                golden_output = golden_fwd_call(cfg.golden.state, cfg.golden.inputs)

            if cfg.conv_output != None:
                test_output = cfg.conv_output(test_output)
            # print('test_output', jax.device_get(test_output))
            # print('golden_output', jax.device_get(golden_output))
            # Transfer results to CPU before comparison
            self.assertNestedAllClose(jax.device_get(test_output), jax.device_get(golden_output),
                                    atol=cfg.test.atol, rtol=cfg.test.rtol)


    def helper_bwd(self, cfg: TestCaseConfig):
        cfg.instantiate()
        cfg.print_summary()
        @partial(jax.jit, static_argnums=0)
        def test_bwd_call(test_layer, test_state, test_inputs):
            def loss_fn(state):
                output, aux = self._fwd_call(test_layer, state, test_inputs)
                return cfg.loss_fn(output), output  # Return both loss and output

            (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(test_state)
            return loss, grads, output

        @partial(jax.jit, static_argnums=0)
        def golden_bwd_call(golden_layer, golden_state, golden_inputs):
            def loss_fn(state):
                output, aux = self._fwd_call(golden_layer, state, golden_inputs)
                return cfg.loss_fn(output), output  # Return both loss and output

            (loss, output), grads = jax.value_and_grad(loss_fn, has_aux=True)(golden_state)
            return loss, grads, output

        jax.config.update('jax_platform_name', cfg.test.device)
        with cfg.test.mesh:
            test_loss, test_grads, test_output = test_bwd_call(cfg.test.layer, cfg.test.state, cfg.test.inputs)
        
        test_output = jax.tree_map(jax.device_get, test_output)
        test_loss = jax.tree_map(jax.device_get, test_loss)
        test_grads = jax.tree_map(jax.device_get, test_grads)
        
        if cfg.golden:
            jax.config.update('jax_platform_name', cfg.golden.device)
            with cfg.golden.mesh:
                golden_loss, golden_grads, golden_output = golden_bwd_call(cfg.golden.layer, cfg.golden.state, cfg.golden.inputs)

                #Transfer results to CPU before comparison
                if cfg.golden.device == "neuron":
                    golden_loss = jax.tree_map(jax.device_get, golden_loss)
                    golden_grads = jax.tree_map(jax.device_get, golden_grads)
                    golden_output = jax.tree_map(jax.device_get, golden_output)

            # print('test_loss', test_loss, 'golden_loss', golden_loss)
            # Compare losses
            self.assertNestedAllClose(test_loss, golden_loss, atol=cfg.test.atol, rtol=cfg.test.rtol)
            # print('test_grads', test_grads, 'golden_grads', golden_grads)
            # Compare gradients
            self.assertNestedAllClose(test_grads, golden_grads, atol=cfg.test.atol, rtol=cfg.test.rtol)
            # print('test_output', test_output, 'golden_output', golden_output)
            # Compare outputs
            self.assertNestedAllClose(test_output, golden_output, atol=cfg.test.atol, rtol=cfg.test.rtol)

class GatingTestCase(TestCase):
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
        assert cfg.golden is not None, "Golden config must be provided for comparison."

        @partial(jax.jit, static_argnums=0)
        def test_fwd_call(test_layer, test_state, test_inputs):
            test_output = self._fwd_call(test_layer, test_state, test_inputs)
            return test_output

        @partial(jax.jit, static_argnums=0)
        def golden_fwd_call(golden_layer, golden_state, golden_inputs):
            golden_output =  self._fwd_call(golden_layer, golden_state, golden_inputs)
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

    def validate_block_to_expert(self, block_to_expert, cfg, num_blocks, num_blocks_per_expert):
        # Validating block_to_expert tensor
        # (O, G, N)
        O, G, N = block_to_expert.shape
        assert N == num_blocks
        for o in range(O):
            for g in range(G):
                num_blocks_for_expert = {}
                for n in range(N):
                    expert_id = block_to_expert[o, g, n]
                    if expert_id not in num_blocks_for_expert:
                        num_blocks_for_expert[expert_id] = 0
                    num_blocks_for_expert[expert_id] += 1
                assert len(num_blocks_for_expert) == cfg.test.cfg.num_experts, f"Expected {cfg.test.cfg.num_experts} experts, but got {len(num_blocks_for_expert)}"
                for expert_id, num_blocks in num_blocks_for_expert.items():
                    assert num_blocks == num_blocks_per_expert, f"Expert {expert_id} has {num_blocks} blocks, expected {num_blocks_per_expert}"

    def validate_token_position_to_id(self, O, G, N, block_size, S, block_to_expert, expert_affinities_masked, token_position_to_id):
        # Validating token_position_to_id (O, G, N*B)
        token_position_to_id = token_position_to_id.reshape(O, G, N, block_size)
        in_range = np.where(((token_position_to_id >=0) & (token_position_to_id<=S)), True, False)
        all_in_range = jnp.all(in_range)
        for o in range(O):
            for g in range(G):
                for n in range(N):
                    bid = n
                    expert_id = block_to_expert[o, g, n]
                    for b in range(block_size):
                        # current block's id:
                        token_id_in_seq = token_position_to_id[o, g, n, b]
                        if token_id_in_seq == S:
                            # padding token
                            continue
                        elif token_id_in_seq < 0 or token_id_in_seq > S:
                            print(token_position_to_id[o, g, n, b], token_id_in_seq)
                        else:
                            assert expert_affinities_masked[o, g, token_id_in_seq, expert_id] > 0
                        # must be in range [0, S]
        assert all_in_range, f"token_position_to_id out of range: {token_position_to_id}, {in_range}"

    def validate_expert_affinties(self, expert_affinities_masked, cfg):
        # assert that max of top k in each row of expert_affinities_masked
        # O, G, S, E
        assert np.all(np.count_nonzero(expert_affinities_masked, axis=3) <= cfg.test.cfg.top_k)

    def helper_blockwise_gating(self, cfg):
        cfg.instantiate()
        cfg.print_summary()
        assert cfg.golden is None, "This test doesn't use golden "
        @partial(jax.jit, static_argnums=0)
        def test_fwd_call(test_layer, test_state, test_inputs):
            return self._fwd_call(test_layer, test_state, test_inputs)
        with cfg.test.mesh:
            test_output = test_fwd_call(cfg.test.layer, cfg.test.state, cfg.test.inputs)

        test_output = jax.device_get(test_output)
        outputs = test_output[0]
        token_position_to_id, expert_affinities_masked = outputs.combine_tensor
        _,_, S, E = expert_affinities_masked.shape
        expert_capacity = int(S * cfg.test.cfg.train_capacity_factor / E)
        if isinstance(cfg.test.cfg, TopKGatingGatherBlockwise.Config):
            block_size = cfg.test.cfg.block_size
        else:
            block_size = expert_capacity
        num_blocks = math.ceil(expert_capacity / block_size) * E
        num_blocks_per_expert = num_blocks / E

        block_to_expert = outputs.dispatch_tensor
        O, G, N = block_to_expert.shape
        self.validate_block_to_expert(block_to_expert, cfg, num_blocks, num_blocks_per_expert)
        self.validate_token_position_to_id(O, G, N, block_size, S, block_to_expert, expert_affinities_masked, token_position_to_id)
        self.validate_expert_affinties(expert_affinities_masked, cfg)

    def helper_blockwise_gating_v2(self, cfg):
        cfg.instantiate()
        cfg.print_summary()
        @partial(jax.jit, static_argnums=0)
        def test_fwd_call(test_layer, test_state, test_inputs):
            return self._fwd_call(test_layer, test_state, test_inputs)
        with cfg.test.mesh:
            test_output = test_fwd_call(cfg.test.layer, cfg.test.state, cfg.test.inputs)
        with cfg.golden.mesh:
            golden_output = test_fwd_call(cfg.golden.layer, cfg.golden.state, cfg.golden.inputs)
        test_output = jax.device_get(test_output)
        golden_output = jax.device_get(golden_output)
        outputs = test_output[0]
        golden_outputs = golden_output[0]

        token_position_to_id, expert_affinities_masked = outputs.combine_tensor
        g_token_position_to_id, g_expert_affinities_masked = golden_outputs.combine_tensor
        block_to_expert = outputs.dispatch_tensor

        self.assertNestedAllClose(
            expert_affinities_masked,
            g_expert_affinities_masked,
            atol=cfg.test.atol, rtol=cfg.test.rtol)
        self.assertNestedAllClose(
            token_position_to_id,
            g_token_position_to_id,
            atol=cfg.test.atol, rtol=cfg.test.rtol)

        _,_, S, E = expert_affinities_masked.shape
        expert_capacity = int(S * cfg.test.cfg.train_capacity_factor / E)

        if isinstance(cfg.test.cfg, TopKGatingGatherBlockwise.Config):
            block_size = cfg.test.cfg.block_size
        else:
            block_size = expert_capacity
        num_blocks = math.ceil(expert_capacity / block_size) * E
        num_blocks_per_expert = num_blocks / E

        O, G, N = block_to_expert.shape
        self.validate_block_to_expert(block_to_expert, cfg, num_blocks, num_blocks_per_expert)
        self.validate_token_position_to_id(O, G, N, block_size, S, block_to_expert, expert_affinities_masked, token_position_to_id)
        self.validate_expert_affinties(expert_affinities_masked, cfg)

class TestGatingOnCpu(GatingTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'cpu')
    
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, layer='gating', test=TopKGatingGather, golden=TopKGating, test_device="cpu", golden_device="cpu"))
    def test_fwd_gather_vs_einsum(self, cfg):
        self.helper_fwd(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, layer='gating', test=TopKGatingGatherBlockwise, golden=None, test_device="cpu"))
    def test_fwd_blockwisegather(self, cfg):
        self.helper_blockwise_gating(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, layer='gating', test=TopKGatingGatherBlockwiseV2, golden=TopKGatingGatherBlockwise, test_device="cpu", golden_device="cpu"))
    def test_fwd_blockwisev2(self, cfg):
        self.helper_blockwise_gating_v2(cfg)

# pylint: disable=no-self-use,protected-access
class TestLayerOnCpu(LayerTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'cpu')

    @unittest.skip("test fwd skipped as fwd is part of fwd+bwd test")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwise, golden=TopKGatingGather, test_device="cpu", golden_device="cpu"))
    def test_fwd_blockwisegather_vs_gather(self, cfg: TestCaseConfig):
        self.helper_fwd(cfg)

    @unittest.skip("test fwd skipped as fwd is part of fwd+bwd test")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwiseV2, golden=TopKGating, test_device="cpu", golden_device="cpu"))
    def test_fwd_blockwisev2_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_fwd(cfg)

    @unittest.skip("skip gather")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGather, golden=TopKGating, test_device="cpu", golden_device="cpu"))
    def test_fwdbwd_gather_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_bwd(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwise, golden=TopKGating, test_device="cpu", golden_device="cpu"))
    def test_fwdbwd_blockwisegather_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_bwd(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwiseV2, golden=TopKGating, test_device="cpu", golden_device="cpu"))
    def test_fwdbwd_blockwisev2_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_bwd(cfg)

class TestLayerOnTrn(LayerTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'neuron')
    
    @unittest.skip("test fwd skipped as fwd is part of fwd+bwd test")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwise, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwd_blockwisegather_vs_einsum(self, cfg):
        self.helper_fwd(cfg)
    
    @unittest.skip("test fwd skipped as fwd is part of fwd+bwd test")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwiseV2, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwd_blockwisegatherv2_vs_einsum(self, cfg):
        self.helper_fwd(cfg)
    
    @unittest.skip("skip gather")
    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwdbwd_gather_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_bwd(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwise, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwdbwd_blockwisegather_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_bwd(cfg)

    @parameterized.named_parameters(get_training_configs(test_suite=TEST_SUITE, test=TopKGatingGatherBlockwiseV2, golden=TopKGating, test_device="neuron", golden_device="cpu"))
    def test_fwdbwd_blockwisev2_vs_einsum(self, cfg: TestCaseConfig):
        self.helper_bwd(cfg)

class TestDev150bUnit(LayerTestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        jax.config.update('jax_platform_name', 'cpu')
        self.test_device = 'cpu'
        self.golden_device = 'cpu'
        self.golden = TopKGating

    def create_cfg(self, test, golden=None, layer="moe"):
        golden = self.golden if golden is None else golden
        return create_test_config(
            layer=layer,
            test=test,
            golden=golden,
            test_device=self.test_device,
            golden_device=self.golden_device,
            input_dim=8192,
            hidden_dim=16384,
            n_experts=8,
            n_groups=1,
            top_k=2,
            capacity_factor=2,
            mesh_spec={"fsdp": -1, "model": 16},
            batch=8,
            seq=8192,
            dtype=jnp.bfloat16,
        )[1]

    def test_fwd_blockwise_vs_einsum(self):
        self.helper_fwd(self.create_cfg(test=TopKGatingGatherBlockwise))
    
    def test_fwd_blockwisev2_vs_einsum(self):
        self.helper_fwd(self.create_cfg(test=TopKGatingGatherBlockwiseV2))

    @unittest.skip("skip gather")
    def test_fwd_gather_vs_einsum(self):
        self.helper_fwd(self.create_cfg(test=TopKGatingGather))
    
    def test_fwdbwd_blockwise_vs_einsum(self):
        self.helper_bwd(self.create_cfg(test=TopKGatingGatherBlockwise))

    def test_fwdbwd_blockwisev2_vs_einsum(self):
        self.helper_bwd(self.create_cfg(test=TopKGatingGatherBlockwiseV2))
    
    @unittest.skip("skip gather")
    def test_fwdbwd_gather_vs_einsum(self):
        self.helper_bwd(self.create_cfg(test=TopKGatingGather))

class TestDev150bInteg(TestDev150bUnit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_device = 'neuron'
        self.golden_device = 'cpu'
        self.golden = TopKGating
    
    def test_fwd_blockwise_vs_einsum(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGatherBlockwise))

    def test_fwd_blockwisev2_vs_einsum(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGatherBlockwiseV2))
    
    @unittest.skip("skip gather")
    def test_fwd_gather_vs_einsum(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGather))
    
    def test_fwdbwd_blockwise_vs_einsum(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_bwd(self.create_cfg(test=TopKGatingGatherBlockwise))

    def test_fwdbwd_blockwisev2_vs_einsum(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_bwd(self.create_cfg(test=TopKGatingGatherBlockwiseV2))
    
    @unittest.skip("skip gather")
    def test_fwdbwd_gather_vs_einsum(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_bwd(self.create_cfg(test=TopKGatingGather))    

class TestDev150bGating(GatingTestCase):
    def create_cfg(self, test, golden, test_device, golden_device="cpu", layer="gating"):
        return create_test_config(
            layer=layer,
            test=test,
            golden=golden,
            golden_device=golden_device,
            test_device=test_device,
            input_dim=8192,
            hidden_dim=16384,
            n_experts=8,
            n_groups=1,
            top_k=2,
            capacity_factor=2,
            mesh_spec={"fsdp": -1, "model": 16},
            batch=8,
            seq=8192,
            dtype=jnp.bfloat16,
        )[1]
    
    def test_unit_fwd_blockwise(self):
        jax.config.update('jax_platform_name', 'cpu')
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwise, golden=None, test_device="cpu", layer="gating"))

    def test_integ_fwd_blockwise(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwise, golden=None, test_device="neuron", layer="gating"))
    
    def test_unit_fwd_blockwisev2(self):
        jax.config.update('jax_platform_name', 'cpu')
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwiseV2, golden=None, test_device="cpu", layer="gating"))

    def test_integ_fwd_blockwisev2(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_blockwise_gating(self.create_cfg(test=TopKGatingGatherBlockwiseV2, golden=None, test_device="neuron", layer="gating"))

    @unittest.skip("skip gather")
    def test_unit_fwd_gather(self):
        jax.config.update('jax_platform_name', 'cpu')
        self.helper_fwd(self.create_cfg(test=TopKGatingGather, golden=TopKGating, test_device="cpu", golden_device="cpu", layer="gating"))

    @unittest.skip("skip gather")
    def test_integ_fwd_gather(self):
        jax.config.update('jax_platform_name', 'neuron')
        self.helper_fwd(self.create_cfg(test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu", layer="gating"))


if __name__ == "__main__":
    absltest.main()
