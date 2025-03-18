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
from axlearn.common.utils_neuron import ModuleTester, TestConfig, TestConfigBuilder
from axlearn.common.utils_neuron import get_training_configs

test_configs = get_training_configs()

# pylint: disable=no-self-use,protected-access
class TestImplCorrectnessInteg(TestCase, ModuleTester):
    def test_fwd_correctness_blockwise(self):
        builder = TestConfigBuilder()
        builder.reset()
        builder.with_dimensions(batch_size=16, seq_len=32, input_dim=4096, )
        builder.with_expert_settings(
            hidden_dim=6144, outer_batch=1, 
            num_groups=1, num_experts=4, 
            expert_capacity=None, 
            train_capacity_factor=1.2,
            block_size=128,
            use_blockwise_kernel=True)
        builder.with_mesh_settings({"fsdp":-1, "model":4})
        # gating_config = builder.build_gating_layer_config(test_device="neuron")
        # gating_config.conv_output = None
        # self._test_fwd_internal(gating_config, assert_outputs=False)
        moe_config = builder.build_moe_layer_config(test_device="neuron")
        self._test_fwd_internal(moe_config, assert_outputs=True)

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
