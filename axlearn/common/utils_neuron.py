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
    "atol": 8e-3,
    "rtol": 1e-3,
}

def get_mesh_dims_from_spec(mesh_spec):
    mesh = mesh_shape_from_axes(**mesh_spec)
    mesh = infer_mesh_shape(mesh)
    return mesh

def build_name(cfg, invoker_cfg):
    if invoker_cfg['mesh_spec']:
        mesh_str = f"fsdp{invoker_cfg['mesh_spec']['fsdp']}tp{invoker_cfg['mesh_spec']['model']}"
    else:
        mesh_str = ''

    if invoker_cfg['dtype'] == jnp.bfloat16:
        dtype_str = "bf16"
    elif invoker_cfg['dtype'] == jnp.float32:
        dtype_str = "fp32"
    else:
        dtype_str = f"{invoker_cfg['dtype']}"
    if hasattr(cfg, 'gating'):
        # MoE layer
        if hasattr(cfg.gating, 'block_size'):
            block_size_str = f'_blocksize{cfg.gating.block_size}'
        else:
            block_size_str = ''
        return f"MoE_b{invoker_cfg['batch_size']}_s{invoker_cfg['seq_len']}_i{cfg.input_dim}_h{cfg.hidden_dim}_e{cfg.num_experts}_topk{cfg.gating.top_k}_g{cfg.num_groups}_ec{cfg.gating.train_capacity_factor}{block_size_str}_mesh{mesh_str}_{dtype_str}"
    else:
        # Gating layer
        E = invoker_cfg['input_shape'][-1]
        G = invoker_cfg['input_shape'][1]
        if hasattr(cfg, 'block_size'):
            block_size_str = f'_blocksize{cfg.block_size}'
        else:
            block_size_str = ''
        return f"Gating_b{invoker_cfg['batch_size']}_s{invoker_cfg['seq_len']}_e{E}_topk{cfg.top_k}_g{G}_ec{cfg.train_capacity_factor}{block_size_str}_mesh{mesh_str}_{dtype_str}"

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

    return TopKGating.Output(
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
            return self.cfg.gating.__class__.__name__
        else:
            return self.cfg.__class__.__name__

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
        self.golden = ModuleConfig(golden_cfg, golden_invoker_cfg) if golden_cfg else None
        self.loss_fn = loss_fn
        self.conv_output = conv_output
        self.prefix = prefix

    def print_summary(self):
        print('\n-------------------\nTest:', build_name(self.test.cfg, self.test.invoker_cfg))
        print('> Test device:', self.test.device, 'Layer:', self.test.layer_type, 'Gating:', self.test.gating_type)
        if self.golden:
            print('> Golden device:', self.golden.device, 'Layer:', self.golden.layer_type, 'Gating:', self.golden.gating_type)
    
    def instantiate(self):
        self.test.mesh_dims = get_mesh_dims_from_spec(self.test.invoker_cfg["mesh_spec"])
        self.test.num_devices = math.prod(self.test.mesh_dims)
        if self.golden:
            self.golden.mesh_dims = get_mesh_dims_from_spec(self.golden.invoker_cfg["mesh_spec"])
            self.golden.num_devices = math.prod(self.golden.mesh_dims)
        self.maybe_set_outer_batch()
        if self.golden:
            self.init_layer(self.golden)
        self.init_layer(self.test, state_to_copy=self.golden.state if self.golden else None)
        self.random_inputs_with_mesh()

    def maybe_set_outer_batch(self):
        test_outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.test.mesh_dims)
        if isinstance(self.test.cfg, TransformerFeedForwardMoE.Config):
            self.test.cfg.outer_batch = test_outer_batch
        self.test.outer_batch = test_outer_batch
        
        if self.golden:
            golden_outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.golden.mesh_dims)
            if isinstance(self.golden.cfg, TransformerFeedForwardMoE.Config):
                self.golden.cfg.outer_batch = golden_outer_batch
            self.golden.outer_batch = golden_outer_batch

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
                if 'gate_weight' in module_config.state:
                    module_config.state['gate_weight'] = module_config.state['gate_weight'].astype(jnp.float32)
    
    def random_inputs_with_mesh(self): 
        
        # replace O and S from input shape with outer batch and seq
        if self.test.layer_type == "MoE":
            input_key = 'inputs'
            pspec = PartitionSpec(('data','fsdp'), 'model', None)
        else:
            input_key = 'logits'
            pspec = PartitionSpec(('data','fsdp'), "expert", None, None)
            _, G, _, E = self.test.input_shape
            O = self.test.outer_batch
            S = (self.test.invoker_cfg["batch_size"] * self.test.invoker_cfg["seq_len"])//(O * G)
            self.test.input_shape = (O, G, S, E)
            if self.golden:
                _, G, _, E = self.golden.input_shape
                O = self.golden.outer_batch
                S = (self.golden.invoker_cfg["batch_size"] * self.golden.invoker_cfg["seq_len"])//(O * G)
                self.golden.input_shape = (O, G, S, E)
        
        in_shard_test = NamedSharding(mesh=self.test.mesh, spec=pspec)
        # create tensors on host to avoid OOM
        with jax.default_device(jax.devices("cpu")[0]):
            inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=self.test.input_shape, dtype=self.test.dtype)
        inputs = jax.device_get(inputs)   # device_put seg-faults without this
        self.test.inputs[input_key] = jax.device_put(inputs, in_shard_test)

        if self.golden:
            assert self.test.input_shape == self.golden.input_shape
            in_shard_golden = NamedSharding(mesh=self.golden.mesh, spec=pspec)
            self.golden.inputs[input_key] = jax.device_put(inputs, in_shard_golden)

class GridSpaceBuilder:
    def __init__(self, layer='moe', test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"):
        self.layer = layer
        self.test = test
        self.golden = golden
        self.test_device = test_device
        self.golden_device = golden_device

    def create_test_config(self, **kwargs):
        return create_test_config(
            test=self.test, golden=self.golden, test_device=self.test_device, golden_device=self.golden_device,
            layer=self.layer,
            **kwargs
        )
    
    def build_toy_grid_space(self):
        return [
            self.create_test_config(
                input_dim=3, hidden_dim=6,
                n_experts=4, top_k=1, n_groups=1, capacity_factor=2, 
                mesh_spec={}, 
                batch=1, seq=8, dtype=jnp.float32, 
                block_size=4,
            ),
            self.create_test_config(
                input_dim=3, hidden_dim=6, 
                n_experts=4, top_k=2, n_groups=1, capacity_factor=2,
                mesh_spec={}, 
                batch=1, seq=8, dtype=jnp.float32, 
                block_size=4
            ),
            self.create_test_config(
                input_dim=256, hidden_dim=512,
                n_experts=16, top_k=2, n_groups=1, capacity_factor=2,
                mesh_spec={}, 
                batch=1, seq=2048, dtype=jnp.float32, 
                block_size=256
            ),
            self.create_test_config(
                input_dim=3, hidden_dim=6, n_experts=4, 
                top_k=1, n_groups=1, capacity_factor=2, 
                mesh_spec={}, 
                batch=4, seq=8, dtype=jnp.float32, 
                block_size=4
            ),
            self.create_test_config(
                input_dim=3, hidden_dim=6, n_experts=4, 
                top_k=2, n_groups=1, capacity_factor=2, 
                mesh_spec={}, 
                batch=4, seq=8, dtype=jnp.float32, 
                block_size=4
            ),
            self.create_test_config(
                input_dim=3, hidden_dim=6, n_experts=4, 
                top_k=1, n_groups=1, capacity_factor=2, 
                mesh_spec={}, 
                batch=4, seq=8, dtype=jnp.bfloat16, 
                block_size=4
            ),
    ][0]
    
    def build_presubmit_grid_space(self):
        grid_space = []
        kwargs={
            'dtype': jnp.bfloat16,
            'batch': 16,
        }
        # 12B Configs
        kwargs_12b = {
            'input_dim': 2048,
            'hidden_dim': 7168,
            'mesh_spec': {"fsdp":-1, "model":4},
        }

        grid_space.extend([
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=4096),
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=8, top_k=1, n_groups=1, capacity_factor=2, seq=4096),
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=8, top_k=4, n_groups=1, capacity_factor=2, seq=4096),
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=8192),
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=1, top_k=1, n_groups=1, capacity_factor=2, seq=4096),
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=8, top_k=1, n_groups=2, capacity_factor=2, seq=4096),
            self.create_test_config(**kwargs, **kwargs_12b, n_experts=8, top_k=4, n_groups=2, capacity_factor=2, seq=4096),
        ])
        
        if self.layer == "moe":
            # 50B Config
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=4096, hidden_dim=14336, mesh_spec={"fsdp":-1, "model":4}, 
                    n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=4096
                )
            )
            # 150B Config
            # 16x10
            # OOB on neuron, pass on cpu
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=6144, hidden_dim=15360, mesh_spec={"fsdp":-1, "model":16},
                    n_experts=16, top_k=4, n_groups=1, capacity_factor=2, seq=4096
                )
            )
            # 8x20
            grid_space.append(
                self.create_test_config(
                    **kwargs, input_dim=8192, hidden_dim=16384, mesh_spec={"fsdp":-1, "model":16},
                    n_experts=8, top_k=2, n_groups=1, capacity_factor=2, seq=8192
                )
            )
        return grid_space

    def build_grid_space_12B(self):
        # Grid space for testing
        grid_space = []
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 2048,
            'hidden_dim': 7168,
        }
        grid_space.extend([
            # base
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # topk changes
            self.create_test_config(**kwargs, n_experts=8, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=4, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # seqlen changes
                # failed assertionError
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=256, mesh_spec={"fsdp":-1, "model":4}),
                # failed assertionError
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=2048, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=8192, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=16*1024, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=32*1024, mesh_spec={"fsdp":-1, "model":4}),
            # tp8
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=8, seq=4096, mesh_spec={"fsdp":-1, "model":8}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=2, seq=4096, mesh_spec={"fsdp":-1, "model":32}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=1, seq=4096, mesh_spec={"fsdp":-1, "model":64}),

            # num experts
                # failed broadcasting error
            self.create_test_config(**kwargs, n_experts=1, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
                # failed assertionError
            self.create_test_config(**kwargs, n_experts=7, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # num groups
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
                # failed assertionError
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
        ])
        return grid_space

    def build_grid_space_50B(self):
        # Grid space for testing
        grid_space = []
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 4096,
            'hidden_dim': 14336,
        }

        grid_space.extend([
            # base
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # topk changes
            self.create_test_config(**kwargs, n_experts=8, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=4, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # seqlen changes
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=256, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=2048, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=8192, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=16*1024, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=32*1024, mesh_spec={"fsdp":-1, "model":4}),

            # tp8
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=8, seq=4096, mesh_spec={"fsdp":-1, "model":8}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=2, seq=4096, mesh_spec={"fsdp":-1, "model":32}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=2, capacity_factor=2, batch=1, seq=4096, mesh_spec={"fsdp":-1, "model":64}),

            # num experts
            self.create_test_config(**kwargs, n_experts=1, top_k=1, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=7, top_k=2, n_groups=2, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # num groups
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
        ])
        return grid_space

    def build_grid_space_150B(self):
        # Grid space for testing
        grid_space = []
        kwargs={
            'dtype': jnp.bfloat16,
            'input_dim': 6144,
            'hidden_dim': 15360,
        }

        grid_space.extend([
            # base
            self.create_test_config(**kwargs, n_experts=16, top_k=4, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # topk changes
            self.create_test_config(**kwargs, n_experts=16, top_k=1, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(**kwargs, n_experts=16, top_k=8, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
            # capf change
            self.create_test_config(**kwargs, n_experts=16, top_k=8, n_groups=1, capacity_factor=4, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),

            # seqlen changes
            # using 8x20b
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=256, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=2048, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=8192, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=16*1024, mesh_spec={"fsdp":-1, "model":4}),
            self.create_test_config(dtype=jnp.bfloat16, input_dim=8192, hidden_dim=16384, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=32*1024, mesh_spec={"fsdp":-1, "model":4}),

            # tp8
            # self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=2, capacity_factor=2, batch=8, seq=4096, mesh_spec={"fsdp":-1, "model":8}),
            self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=2, capacity_factor=2, batch=4, seq=4096, mesh_spec={"fsdp":-1, "model":16}),
            # self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=2, capacity_factor=2, batch=2, seq=4096, mesh_spec={"fsdp":-1, "model":32}),
            self.create_test_config(**kwargs, n_experts=16, top_k=2, n_groups=2, capacity_factor=2, batch=1, seq=4096, mesh_spec={"fsdp":-1, "model":64}),
            
            # num groups
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=4, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),

            # num experts
            self.create_test_config(**kwargs, n_experts=8, top_k=2, n_groups=1, capacity_factor=2, batch=16, seq=4096, mesh_spec={"fsdp":-1, "model":4}),
        ])
        return grid_space


def get_gating_config(gating_cls, num_experts, top_k, train_capacity_factor, expert_capacity, block_size=None, name=None):

    cfg = gating_cls.default_config()
    if name:
        cfg.set(name=name)
    cfg.top_k = top_k
    cfg.train_capacity_factor = train_capacity_factor
    cfg.expert_capacity = expert_capacity
    cfg.num_experts = num_experts
    if block_size is not None and isinstance(cfg, TopKGatingGatherBlockwise.Config):
        cfg.block_size = block_size
    return cfg

def create_test_config(test, golden, test_device, golden_device, input_dim, hidden_dim, n_experts, top_k, n_groups, capacity_factor, mesh_spec, batch, seq, dtype, block_size=512, layer='moe'):
    """
    Ensure any new param added here also shows up in the name to prevent multiple tests from having same name.
    You will see an exception calling that out if it happens.
    """

    model_param_init = DefaultInitializer.default_config().set(
        init_by_param_name={
            PARAM_REGEXP_WEIGHT: WeightInitializer.default_config().set(
                fan="fan_in", distribution="normal"
            )
        }
    )

    if layer == "moe":
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
        # enabling nonorm gives us better check of the kernel logits, what's missing here is just add of residual
        
        test_cfg.structure = "nonorm"
        test_cfg.gating = get_gating_config(test, n_experts, top_k, capacity_factor, expert_capacity=None, block_size=block_size)

        if golden:
            golden_cfg = test_cfg.clone(name="golden")
            golden_cfg.gating = get_gating_config(golden, n_experts, top_k, capacity_factor, expert_capacity=None)
        else:
            golden_cfg = None
        conv_output = None
    else:
        test_cfg = get_gating_config(test, n_experts, top_k, capacity_factor, expert_capacity=None, name="test", block_size=block_size)

        if golden:
            golden_cfg = get_gating_config(golden, n_experts, top_k, capacity_factor, expert_capacity=None, name="golden", block_size=block_size)
        else:
            golden_cfg = None
        if test == TopKGatingGather and golden == TopKGating:
            conv_output = partial(_topkgather_to_topk, top_k=top_k, cf=capacity_factor)
        else:
            conv_output = None
    
    test_invoker_cfg = {
        "batch_size": batch,
        "seq_len": seq,
        "input_dim": input_dim,
        "dtype": jnp.bfloat16 if dtype in ["bfloat16", jnp.bfloat16] else jnp.float32,
        "device": test_device,
        "mesh_spec": mesh_spec,
        "input_shape": (batch, seq, input_dim) if layer == "moe" else ('O', n_groups, 'S', n_experts),
    }
    if golden:
        golden_invoker_cfg = dict(test_invoker_cfg)
        golden_invoker_cfg['device'] = golden_device
    else:
        golden_invoker_cfg = {}
    
    config = TestCaseConfig(
        test_cfg, 
        golden_cfg, 
        test_invoker_cfg, 
        golden_invoker_cfg,
        loss_fn=lambda x: jnp.mean(x)*1e2,
        conv_output=conv_output,
        prefix="_moe"
    )

    return (build_name(test_cfg, test_invoker_cfg) + config.prefix, config)

@cache
def get_training_configs(test_suite="presubmit", layer='moe', test=TopKGatingGather, golden=TopKGating, test_device="neuron", golden_device="cpu"):
    builder = GridSpaceBuilder(layer=layer, test=test, golden=golden, test_device=test_device, golden_device=golden_device)
    if test_suite == "toy":
        return builder.build_toy_grid_space()
    elif test_suite == 'presubmit':
        return builder.build_presubmit_grid_space()
    elif test_suite == 'small_models':
        return builder.build_grid_space_12B() + builder.build_grid_space_50B()
    elif test_suite == '12b':
        return builder.build_grid_space_12B()
    elif test_suite == '50b':
        return builder.build_grid_space_50B()
    elif test_suite == '150b':
        return builder.build_grid_space_150B()
    else:
        raise ValueError(f"Unknown test suite: {test_suite}")

    # leaving it here for any custom local testing
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

