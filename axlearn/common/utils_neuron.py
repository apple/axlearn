# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Utils for tests for mixture_of_experts.py"""
import os
from functools import partial, cache
from itertools import product
import math

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, Mesh

from axlearn.common.mixture_of_experts import (
    TopKGating,
    TransformerFeedForwardMoE,
    TopKGatingGather,
    TopKGatingGatherBlockwise,
    get_outer_batch_from_mesh
)

from axlearn.common.layers import (
    Dropout,
    StochasticDepth,
    RMSNorm,
)
#
from axlearn.common.utils import PartitionSpec, infer_mesh_shape, cast_floats
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, mesh_shape_from_axes
from axlearn.common.param_init import PARAM_REGEXP_WEIGHT, DefaultInitializer, WeightInitializer
from axlearn.experiments.text.gpt.envy import MOE_OUTER_BATCH_AXIS_NAMES, MOE_DIM_TO_MESH_AXIS_MAP

# FP32 test tolerances
TEST_TOLS_FP32 = {
    "atol": 5e-4,
    "rtol": 1e-2,
}
# BF16 test tolerances
TEST_TOLS_BF16 = {
    "atol": 5e-2,
    "rtol": 1e-2,
}

def get_mesh_dims_from_spec(mesh_spec):
    mesh = mesh_shape_from_axes(**mesh_spec)
    mesh = infer_mesh_shape(mesh)
    return mesh

def _topkgather_to_topk(output, top_k, cf):
    tok_perm_idx, expert_index, exp_aff_mask = output.combine_tensor

    O, G, S, _ = tok_perm_idx.shape
    E = exp_aff_mask.shape[-1]

    expert_cap = jnp.int32(S*cf/E)

    exp_aff = jnp.take_along_axis(exp_aff_mask, expert_index, axis=-1)

    base = jnp.zeros((O, G, S, E * expert_cap), dtype=exp_aff_mask.dtype)

    idx_O, idx_G, idx_S = jnp.meshgrid(
        jnp.arange(O), 
        jnp.arange(G), 
        jnp.arange(S), 
        indexing='ij'
    )

    output_tensor = base.at[idx_O[..., None], idx_G[..., None], idx_S[..., None], tok_perm_idx].add(exp_aff)
    output_tensor = output_tensor.reshape(O, G, S, E, expert_cap)

    dispatch_tensor = output_tensor.astype(bool)

    return TopKGatingGather.Output(
        combine_tensor=output_tensor,
        dispatch_tensor=dispatch_tensor,
        load_balance_loss=output.load_balance_loss,
        router_z_loss=output.router_z_loss
    )

class ModuleConfig():
    def __init__(self, cfg, invoker_cfg):
        self.cfg = cfg
        self.invoker_cfg = invoker_cfg
        self.dtype = invoker_cfg['dtype']
        self.input_shape = invoker_cfg['input_shape']
        self.layer = None
        self.mesh = None
        self.mesh_spec = invoker_cfg['mesh_spec']
        self.mesh_dims = None
        self.num_devices = None
        self.device = invoker_cfg['device']
        self.out_shard = None
        self.inputs = {}
        self.state = None
        self.atol = TEST_TOLS_BF16['atol'] if self.dtype in ["bfloat16", jnp.bfloat16] else TEST_TOLS_FP32['atol']
        self.rtol = TEST_TOLS_BF16['rtol'] if self.dtype in ["bfloat16", jnp.bfloat16] else TEST_TOLS_FP32['rtol']

    @property
    def layer_type(self):
        return "MoE" if isinstance(self.cfg, TransformerFeedForwardMoE.Config) else "Gating"
    
    @property
    def gating_type(self):
        if self.layer_type == "MoE":
            return self.cfg.gating.__class__
        else:
            return self.cfg.__class__

    def print_summary(self):
        print(f"> Class: {self.cfg.__class__}")
        if self.layer_type == "MoE":
            print(f"> GatingClass: {self.cfg.gating.__class__}")
        print(f"> Device: {self.device}")
        print(f"> Dtype: {self.dtype}")
        print(f"> MeshSpec: {self.mesh_spec}")

class TestCaseConfig():
    def __init__(
            self, 
            test_cfg, 
            golden_cfg, 
            test_invoker_cfg, 
            golden_invoker_cfg,
            loss_fn = None, 
            conv_output = None,
            prefix = None
        ):
        self.test = ModuleConfig(test_cfg, test_invoker_cfg)
        self.golden = ModuleConfig(golden_cfg, golden_invoker_cfg)
        self.loss_fn = loss_fn
        self.conv_output = conv_output
        self.prefix = prefix

    def print_summary(self):
        print('Test')
        self.test.print_summary()
        print('Golden')
        self.golden.print_summary()
    
    def instantiate(self):
        self.test.mesh_dims = get_mesh_dims_from_spec(self.test.invoker_cfg["mesh_spec"])
        self.test.num_devices = math.prod(self.test.mesh_dims)
        self.golden.mesh_dims = get_mesh_dims_from_spec(self.golden.invoker_cfg["mesh_spec"])
        self.golden.num_devices = math.prod(self.golden.mesh_dims)
        self.maybe_set_outer_batch()
        self.init_layer(self.golden)
        self.init_layer(self.test, state_to_copy=jax.device_get(self.golden.state) if self.test.cfg == self.golden.cfg else None)
        self.random_inputs_with_mesh()

    def maybe_set_outer_batch(self):
        if isinstance(self.test.cfg, TransformerFeedForwardMoE.Config):
            self.test.cfg.outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.test.mesh_dims)
        if isinstance(self.golden.cfg, TransformerFeedForwardMoE.Config):
            self.golden.cfg.outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.golden.mesh_dims)

    def init_layer(self, module_config, state_to_copy=None):
        devices = jax.devices(module_config.device)[:module_config.num_devices]
        module_config.mesh = Mesh(mesh_utils.create_device_mesh(module_config.mesh_dims, devices=devices), MESH_AXIS_NAMES) 
        with module_config.mesh:
            with jax.default_device(devices[0]):
                module_config.layer = module_config.cfg.instantiate(parent=None) 
                param_specs = module_config.layer.create_parameter_specs_recursively() 
                param_partition_specs = jax.tree.map(lambda spec: spec.sharding, param_specs) 
                
                if state_to_copy:
                    module_config.state = {}
                    for key, value in state_to_copy.items():
                        # print(f"Transferring and sharding {key} to test devices...")
                        # First put on a single device
                        module_config.state[key] = jax.device_put(value, param_partition_specs[key])
                else:
                    def _init_state(prng_key):
                        params = module_config.layer.initialize_parameters_recursively(prng_key)
                        return params
                    init_fn = jax.jit(_init_state, in_shardings=(None,), out_shardings=param_partition_specs)
                    module_config.state = init_fn(jax.random.PRNGKey(123))
                module_config.state = cast_floats(module_config.state, to_dtype=module_config.dtype)
                # TODO: Currently bf16 seeing expert index mismatch with f32. Setting routing to f32.
                module_config.state['gate_weight'] = module_config.state['gate_weight'].astype(jnp.float32)
    
    def random_inputs_with_mesh(self): 
        input_key = 'inputs' if self.test.layer_type == "MoE" else 'logits'
        pspec = PartitionSpec(('data','fsdp'), 'model', None) if self.test.layer_type == "MoE" else PartitionSpec() # seq-parallel inputs

        in_shard_test = NamedSharding(mesh=self.test.mesh, spec=pspec) 
        in_shard_golden = NamedSharding(mesh=self.golden.mesh, spec=pspec)

        assert self.test.input_shape == self.golden.input_shape
        # create tensors on host to avoid OOM
        with jax.default_device(jax.devices("cpu")[0]):
            inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=self.test.input_shape, dtype=self.test.dtype)

        inputs = jax.device_get(inputs)   # device_put seg-faults without this
        self.test.inputs[input_key] = jax.device_put(inputs, in_shard_test)
        self.golden.inputs[input_key] = jax.device_put(inputs, in_shard_golden)

class GridSpaceBuilder:
    # def build_test_configs_integ(self):

    #     seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])
    #     # test_gating_class = TopKGatingGatherBlockwise if self.params["use_blockwise_kernel"] else TopKGatingGather

    #     test_configs = [] 
    #     test_configs.append(
    #         TestCaseConfig(
    #             setup=[
    #                 self.build_moe_topkgather_setup(),
    #                 self.build_moe_topkgather_setup()
    #             ],
    #             test=ModuleConfig(TransformerFeedForwardMoE, "neuron", "MoE", self.params['dtype']),
    #             golden=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", self.params['dtype']),
    #             input_shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"]),
    #             loss_fn=lambda x: jnp.mean(x)*1e2,
    #             mesh_spec=self.params["mesh_spec"],
    #             prefix="_moe")
    #         )
    #     if not self.params["mesh_spec"]: # gating tests only for single-core config
    #         test_configs.append(
    #         TestCaseConfig(
    #             setup=[
    #                 self.build_gating_setup(),
    #                 self.build_gating_setup()
    #             ],
    #             test=ModuleConfig(TopKGatingGather, "neuron", dtype=self.params['dtype']),
    #             golden=ModuleConfig(TopKGatingGather, "cpu", dtype=self.params['dtype']),
    #             input_shape=(self.params["outer_batch"], self.params["num_groups"], seq_len, self.params["num_experts"]),
    #             loss_fn=lambda x: x.load_balance_loss,
    #             mesh_spec=self.params["mesh_spec"],
    #             prefix="_gating")
    #         )
    #     return test_configs

    # def build_test_configs_unit(self):

    #     seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])

    #     test_configs = [] 
    #     test_configs.append(
    #         TestCaseConfig(
    #             setup=[
    #                 self.build_moe_topkgather_setup(),
    #                 self.build_moe_top2_setup()
    #             ],
    #             test=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", self.params['dtype']),
    #             golden=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", self.params['dtype']),
    #             input_shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"]),
    #             loss_fn=lambda x: jnp.mean(x)*1e2,
    #             mesh_spec=self.params["mesh_spec"],
    #             prefix="_moe"
    #             )
    #         )
    #     if not self.params["mesh_spec"]: # gating tests only for single-core config
    #         # gating test
    #         # TODO: use blockwise here
    #         test_configs.append(
    #             TestCaseConfig(
    #                 setup=[
    #                     self.build_gating_setup(),
    #                     self.build_gating_setup()
    #                 ],
    #                 test=ModuleConfig(TopKGatingGather, "cpu", dtype=self.params['dtype']),
    #                 golden=ModuleConfig(TopKGating, "cpu", dtype=self.params['dtype']),
    #                 input_shape=(1, self.params["num_groups"], seq_len, self.params["num_experts"]),
    #                 conv_output=partial(_topkgather_to_topk, top_k=self.params["top_k"], cf=self.params["train_capacity_factor"]),
    #                 loss_fn=lambda x: x.load_balance_loss,
    #                 mesh_spec=self.params["mesh_spec"],
    #                 prefix="_gating"
    #             )
    #         )
    #     return test_configs
    
    def build_grid_space(self):
        # Grid space for testing: Presubmit

        grid_space = []
        # "fsdp":-1, "model":4
        Mistral12B_base = (16, 64, 8, 32, 8, 2, 1, 1, 2, {}, "bfloat16")
        grid_space.append(Mistral12B_base)

        '''
        kwargs={
            'test': test,
            'golden': golden,
            'test_device': test_device,
            'golden_device': golden_device,
            'dtype': jnp.bfloat16,
            'batch': 16,
        }
        # 12B Configs
        grid_space = []
        12b_kwargs = {
            'input_dim': 2048,
            'hidden_dim': 7168,
            'mesh_spec': {"fsdp":-1, "model":4},
        }
        grid_space.extend([
            create_test_config(**kwargs, **12b_kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, seq=4096),
            create_test_config(**kwargs, **12b_kwargs, n_experts=8, top_k=1, n_groups=2, capacity_factor=2, seq=4096),
            create_test_config(**kwargs, **12b_kwargs, n_experts=8, top_k=4, n_groups=2, capacity_factor=2, seq=4096),
            create_test_config(**kwargs, **12b_kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, seq=8192),
            create_test_config(**kwargs, **12b_kwargs, n_experts=1, top_k=1, n_groups=2, capacity_factor=2, seq=4096),
            create_test_config(**kwargs, **12b_kwargs, n_experts=1, top_k=1, n_groups=1, capacity_factor=2, seq=4096),
        ])
        # 50B Config
        grid_space.append(
            create_test_config(
                **kwargs, input_dim=4096, hidden_dim=14336, mesh_spec={"fsdp":-1, "model":4}, 
                n_experts=8, top_k=2, n_groups=2, capacity_factor=2, seq=4096
            )
        )
        # 150B Config
        # 16x10
        grid_space.append(
            create_test_config(
                **kwargs, input_dim=6144, hidden_dim=15360, mesh_spec={"fsdp":-1, "model":16},
                n_experts=16, top_k=4, n_groups=2, capacity_factor=2, seq=4096
            )
        )
        # 8x20
        grid_space.append(
            create_test_config(
                **kwargs, input_dim=6144, hidden_dim=16384, mesh_spec={"fsdp":-1, "model":16},
                n_experts=8, top_k=2, n_groups=2, capacity_factor=2, seq=4096
            )
        )
        '''
        return grid_space

    def build_grid_space_12B(self):
        # Grid space for testing

        grid_space = []

        # Custom Configs
        # b s i h e top_k g ob cf mesh dtype

        # 12B Configs
        Mistral12B_base = (16, 4096, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_base)
        Mistral12B_top1 = (16, 4096, 2048, 7168, 8, 1, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_top1)
        Mistral12B_top4 = (16, 4096, 2048, 7168, 8, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_top4)
        Mistral12B_seq256 = (16, 256, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_seq256)
        Mistral12B_seq2k = (16, 2048, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_seq2k)
        Mistral12B_seq8k = (16, 8192, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_seq8k)
        Mistral12B_seq16k = (16, 16384, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_seq16k)
        Mistral12B_seq32k = (16, 32768, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_seq32k)
        # Mistral12B_tp8 = (8, 4096, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":8}, "bfloat16")
        # grid_space.append(Mistral12B_tp8)
        Mistral12B_tp16 = (4, 4096, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":16}, "bfloat16")
        grid_space.append(Mistral12B_tp16)
        # Mistral12B_tp32 = (2, 4096, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":32}, "bfloat16")
        # grid_space.append(Mistral12B_tp32)
        Mistal8B_tp64 = (1, 4096, 2048, 7168, 8, 2, 2, 1, 2, {"fsdp":-1, "model":64}, "bfloat16")
        grid_space.append(Mistal8B_tp64)
        Mistral12B_expert1 = (16, 4096, 2048, 7168, 1, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_expert1)
        Mistral12B_expert7 = (16, 4096, 2048, 7168, 7, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_expert7)
        Mistral12B_group1 = (16, 4096, 2048, 7168, 8, 2, 1, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral12B_group1)
        Mistal8B_group4 = (16, 4096, 2048, 7168, 8, 2, 4, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistal8B_group4)

        return grid_space

    def build_grid_space_50B(self):
        # Grid space for testing

        grid_space = []

        # Custom Configs
        # b s i h e top_k g ob cf mesh dtype

        # 50B Configs
        Mistral50B_base = (16, 4096, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_base)
        Mistral50B_top1 = (16, 4096, 4096, 14336, 8, 1, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_top1)
        Mistral50B_top4 = (16, 4096, 4096, 14336, 8, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_top4)
        Mistral50B_seq256 = (16, 256, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_seq256)
        Mistral50B_seq2k = (16, 2048, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_seq2k)
        Mistral50B_seq8k = (16, 8192, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_seq8k)
        Mistral50B_seq16k = (16, 16384, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_seq16k)
        Mistral50B_seq32k = (16, 32768, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_seq32k)
        # Mistral50B_tp8 = (8, 4096, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":8}, "bfloat16")
        # grid_space.append(Mistral50B_tp8)
        Mistral50B_tp16 = (4, 4096, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":16}, "bfloat16")
        grid_space.append(Mistral50B_tp16)
        # Mistral50B_tp32 = (2, 4096, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":32}, "bfloat16")
        # grid_space.append(Mistral50B_tp32)
        Mistal50B_tp64 = (1, 4096, 4096, 14336, 8, 2, 2, 1, 2, {"fsdp":-1, "model":64}, "bfloat16")
        grid_space.append(Mistal50B_tp64)


        return grid_space

    def build_grid_space_150B(self):

        grid_space = []
        # Custom Configs
        # b s i h e top_k g ob cf mesh dtype

        Mistral150B_base = (16, 4096, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral150B_base)
        Mistral150B_top1 = (16, 4096, 6144, 15360, 16, 1, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral150B_top1)
        Mistral150B_top2 = (16, 4096, 6144, 15360, 16, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16") # 2 experts
        grid_space.append(Mistral150B_top2)
        Mistral150B_seq256 = (16, 256, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral150B_seq256)
        Mistral150B_seq2k = (16, 2048, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral150B_seq2k)
        # Mistral150B_seq8k = (16, 8192, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16") 
        # grid_space.append(Mistral150B_seq8k)
        # Mistral150B_seq16k = (16, 16384, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        # grid_space.append(Mistral150B_seq16k)
        # Mistral150B_seq32k = (16, 32768, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        # grid_space.append(Mistral150B_seq32k)
        Mistral150B_tp16 = (4, 4096, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":16}, "bfloat16")
        grid_space.append(Mistral150B_tp16)
        Mistral150B_tp64 = (1, 4096, 6144, 15360, 16, 4, 2, 1, 2, {"fsdp":-1, "model":64}, "bfloat16")
        grid_space.append(Mistral150B_tp64)
        Mistral8x20B_base = (16, 4096, 6144, 16384, 8, 2, 2, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral8x20B_base)

        return grid_space


def get_gating_config(gating_cls, top_k, train_capacity_factor, expert_capacity, block_size=None):
    cfg = gating_cls.default_config()
    cfg.top_k = top_k
    cfg.train_capacity_factor = train_capacity_factor
    cfg.expert_capacity = expert_capacity
    if block_size is not None:
        cfg.block_size = block_size
    return cfg

@cache
def get_training_configs(test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"):

    return [
        # works
        create_test_config(
            test, golden, test_device, golden_device, 
            input_dim=3, hidden_dim=6, 
            n_experts=4, top_k=1, n_groups=1, capacity_factor=2, 
            mesh_spec={}, 
            batch=1, seq=8, dtype=jnp.float32, 
            block_size=4
        ),
        # works
        create_test_config(
            test, golden, test_device, golden_device, 
            input_dim=3, hidden_dim=6, 
            n_experts=4, top_k=2, n_groups=1, capacity_factor=2,
            mesh_spec={}, 
            batch=1, seq=8, dtype=jnp.float32, 
            block_size=4
        ),
        # fails
        create_test_config(
            test, golden, test_device, golden_device, 
            input_dim=3, hidden_dim=6, n_experts=4, 
            top_k=1, n_groups=1, capacity_factor=2, 
            mesh_spec={}, 
            batch=4, seq=8, dtype=jnp.float32, 
            block_size=4
        ),
        # fails
        create_test_config(
            test, golden, test_device, golden_device, 
            input_dim=3, hidden_dim=6, n_experts=4, 
            top_k=2, n_groups=1, capacity_factor=2, 
            mesh_spec={}, 
            batch=4, seq=8, dtype=jnp.float32, 
            block_size=4
        ),
        # can be seen as tolerance issue maybe
        create_test_config(
            test, golden, test_device, golden_device, 
            input_dim=3, hidden_dim=6, n_experts=4, 
            top_k=1, n_groups=1, capacity_factor=2, 
            mesh_spec={}, 
            batch=4, seq=8, dtype=jnp.bfloat16, 
            block_size=4
        ),
    ][2]
    
    builder = GridSpaceBuilder()
    test_suite = os.environ.get("TEST_SUITE", 'presubmit').lower()
    if test_suite == 'presubmit':
        grid_space = builder.build_grid_space()
    elif test_suite == '12b':
        grid_space = builder.build_grid_space_12B()
    elif test_suite == '50b':
        grid_space = builder.build_grid_space_50B()
    elif test_suite == '150b':
        grid_space = builder.build_grid_space_150B()
    else:
        raise ValueError(f"Unknown test suite: {test_suite}")
    test_configs = []
    for (batch, seq, input_dim,  hidden_dim, n_experts, top_k, n_groups,
         out_batch, capacity_factor, mesh_spec, dtype) in grid_space:
        test_configs.append(create_test_config(
            test=test,
            golden=golden,
            test_device=test_device,
            golden_device=golden_device,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_experts=n_experts,
            n_groups=n_groups,
            top_k=top_k,
            capacity_factor=capacity_factor, 
            mesh_spec=mesh_spec,
            batch=batch,
            seq=seq,
            dtype=dtype,
        ))
    return test_configs

def create_test_config(test, golden, test_device, golden_device, input_dim, hidden_dim, n_experts, top_k, n_groups, capacity_factor, mesh_spec, batch, seq, dtype, block_size=4):
    model_param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan="fan_in", distribution="normal"
            )
        }
    )

    test_cfg = TransformerFeedForwardMoE.default_config().set(
        name="test",
        param_init=model_param_init
    )
    test_cfg.input_dim = input_dim
    test_cfg.hidden_dim = hidden_dim
    test_cfg.dim_to_mesh_axis_map = MOE_DIM_TO_MESH_AXIS_MAP
    test_cfg.activation = ("nn.silu","linear")
    test_cfg.num_experts = n_experts
    test_cfg.num_groups = n_groups
    test_cfg.gating = get_gating_config(test, top_k, capacity_factor, expert_capacity=None, block_size=4)
    test_invoker_cfg = {
        "batch_size": batch,
        "seq_len": seq,
        "input_dim": input_dim,
        "dtype": jnp.bfloat16 if dtype in ["bfloat16", jnp.bfloat16] else jnp.float32,
        "device": test_device,
        "mesh_spec": mesh_spec,
        "input_shape": (batch, seq, input_dim),
    }

    golden_invoker_cfg = dict(test_invoker_cfg)
    golden_invoker_cfg['device'] = golden_device
    golden_cfg = test_cfg.clone(name="golden")
    golden_cfg.gating = get_gating_config(golden, top_k, capacity_factor, expert_capacity=None)
    
    config = TestCaseConfig(
        test_cfg, 
        golden_cfg, 
        test_invoker_cfg, 
        golden_invoker_cfg,
        loss_fn=lambda x: jnp.mean(x)*1e2,
        conv_output=None,
        prefix="_moe"
    )

    name = f"MoE_b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_topk{top_k}_g{n_groups}_ec{capacity_factor}_mesh{mesh_spec}_dtype_{dtype}"
    return (name + config.prefix, config)