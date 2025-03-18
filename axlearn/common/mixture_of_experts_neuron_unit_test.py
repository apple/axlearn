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
import pytest
from absl.testing import absltest, parameterized


from axlearn.common.test_utils import TestCase
from axlearn.common.utils_neuron import TestConfig, ModuleConfig, ModuleTester, TestConfigBuilder, TransformerFeedForwardMoE
from axlearn.common.utils_neuron import get_training_configs

test_configs = get_training_configs(is_unit=True)

# pylint: disable=no-self-use,protected-access
class TestImplCorrectnessUnit(TestCase, ModuleTester):
    def test_fwd_correctness_blockwise(self):
        builder = TestConfigBuilder()
        builder.reset()
        builder.with_dimensions(batch_size=2, seq_len=16, input_dim=16)
        builder.with_expert_settings(
            hidden_dim=1024, outer_batch=1, 
            num_groups=1, num_experts=6, 
            expert_capacity=None, 
            train_capacity_factor=1, 
            block_size=4,
            use_blockwise_kernel=True)
        gating_config = builder.build_gating_layer_config(test_device="cpu")
        gating_config.conv_output = None
        self._test_fwd_internal(gating_config, assert_outputs=False)
    
    @parameterized.named_parameters(test_configs)
    @pytest.mark.skip(reason="skip")
    def test_fwd_correctness(self, cfg: TestConfig):
        self._test_fwd_internal(cfg)
    
    @parameterized.named_parameters(test_configs)
    @pytest.mark.skip(reason="skip")
    def test_bwd_correctness(self, cfg: TestConfig):
        self._test_bwd_internal(cfg)

if __name__ == "__main__":
    absltest.main()
