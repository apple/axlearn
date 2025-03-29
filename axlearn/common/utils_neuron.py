
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
#
# google/praxis:
# Copyright 2022 The Pax Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Utils for tests for mixture_of_experts.py"""
from functools import partial
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
    get_outer_batch_from_mesh
)

from axlearn.common.layers import (
    Dropout,
    StochasticDepth,
    RMSNorm,
)
jax.config.update('jax_platform_name', 'cpu')
from axlearn.common.utils import PartitionSpec, infer_mesh_shape, cast_floats
from axlearn.experiments.text.gpt.common import MESH_AXIS_NAMES, mesh_shape_from_axes

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

MOE_OUTER_BATCH_AXIS_NAMES = ("data", "fsdp")

MOE_DIM_TO_MESH_AXIS_MAP = {
    "me": PartitionSpec(None, None),
    "emh": PartitionSpec("expert", "fsdp", "model"),
    "ehm": PartitionSpec("expert", "model", "fsdp"),
    "ogsm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, "model"),
    "ogsM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
    "ogse": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
    "ogec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None),
    # Dispatch and combine tensors.
    "ogsec": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, None, "expert", None),
    "oegcm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, "model"),
    "oegcM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, None),
    "ogecm": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, "expert", None, "model"),
    "ogecM": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, None, "expert", None, None),
    "oegch": PartitionSpec(MOE_OUTER_BATCH_AXIS_NAMES, "expert", None, None, "model"),
}

class ModuleConfig():
    def __init__(self, module = None, device = "cpu", layer = None, dtype = jnp.float32):
        assert module is not None
        self.module = module.default_config().set(name="test", dtype=dtype)
        self.device = device
        self.layer = layer # None for top_k, else "MoE"
        self.dtype = dtype
        self.tol = TEST_TOLS_FP32 if dtype == jnp.float32 else TEST_TOLS_BF16

class TestConfig():
    def __init__(self, setup, test: ModuleConfig, golden: ModuleConfig = None, 
                 input_shape: tuple = None, loss_fn = None, conv_output = None, 
                 mesh_spec: dict = None, prefix = None):
        self.setup = setup
        self.test = test
        self.golden = golden if golden is not None else test
        self.input_shape = input_shape
        self.loss_fn = loss_fn
        self.conv_output = conv_output
        self.num_devices = None 
        self.mesh_dims = self.get_mesh_from_spec(mesh_spec)
        self.prefix = prefix

        self.mesh_test = None 
        self.mesh_golden = None
        self.test_inputs = dict() 
        self.golden_inputs = dict()  
        self.out_shard_test = None
        self.out_shard_golden = None        

        for spec, val in setup[0].items():
            setattr(self.test.module, spec, val)

        for spec, val in setup[1].items():
            setattr(self.golden.module, spec, val)

        if test.layer == "MoE":
            self.set_outer_batch()

    def instantiate(self):
                        
        self.instantiate_modules_with_mesh() 
        self.random_inputs_with_mesh()

    def get_mesh_from_spec(self, mesh_spec):  

        mesh = mesh_shape_from_axes(**mesh_spec)
        mesh = infer_mesh_shape(mesh, num_devices=self.num_devices) 
        self.num_devices = math.prod(mesh)
        print("Inferred mesh: ", mesh)

        return mesh 

    def set_outer_batch(self):

        outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.mesh_dims)
        setattr(self.test.module, "outer_batch", outer_batch)
        setattr(self.golden.module, "outer_batch", outer_batch)

    def instantiate_modules_with_mesh(self): 

        print("Instantiating modules with mesh")
        device_type = self.test.device
        devices = jax.devices(device_type)[:self.num_devices]
        print("Test devices: ", devices)
        self.mesh_test = Mesh(mesh_utils.create_device_mesh(self.mesh_dims, devices=devices), MESH_AXIS_NAMES) 
        with self.mesh_test: 
            self.test_layer  = self.test.module.instantiate(parent=None) 
            test_param_specs = self.test_layer.create_parameter_specs_recursively()
            test_param_partition_specs = jax.tree.map(lambda spec: spec.sharding, test_param_specs)
            
            def _init_state(prng_key): 
                params = self.test_layer.initialize_parameters_recursively(prng_key) 
                return params 
            init_fn = jax.jit(_init_state, in_shardings=(None,), out_shardings = test_param_partition_specs) 
            
            self.test_state = init_fn(jax.random.PRNGKey(123)) 
            self.test_state = cast_floats(self.test_state, to_dtype=self.test.dtype)

        device_type = self.golden.device
        devices = jax.devices(device_type)[:self.num_devices]
        print("Golden devices: ", devices)
        self.mesh_golden = Mesh(mesh_utils.create_device_mesh(self.mesh_dims, devices=devices), MESH_AXIS_NAMES) 
        with self.mesh_golden: 
            self.golden_layer  = self.golden.module.instantiate(parent=None) 
            golden_param_specs = self.golden_layer.create_parameter_specs_recursively() 
            golden_param_partition_specs = jax.tree.map(lambda spec: spec.sharding, golden_param_specs) 
            
            def _init_state(prng_key):
                params = self.golden_layer.initialize_parameters_recursively(prng_key)
                return params
            init_fn = jax.jit(_init_state, in_shardings=(None,), out_shardings=golden_param_partition_specs)
            
            self.golden_state = init_fn(jax.random.PRNGKey(123))
            self.golden_state = cast_floats(self.golden_state, to_dtype=self.golden.dtype)

    def random_inputs_with_mesh(self): 

        input_key = 'inputs' if self.test.layer == "MoE" else 'logits'
        pspec = PartitionSpec(('data','fsdp'), 'model', None) if self.test.layer == "MoE" else PartitionSpec() # seq-parallel inputs

        in_shard_test = NamedSharding(mesh=self.mesh_test, spec = pspec) 
        in_shard_golden = NamedSharding(mesh=self.mesh_golden, spec = pspec)
        
        print(self.input_shape) 
        with jax.default_device(jax.devices("cpu")[0]):    # create tensors on host to avoid OOM  
            inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=self.input_shape, dtype=self.test.dtype) 
        
        inputs = jax.device_get(inputs)   # device_put seg-faults without this 
        self.test_inputs[input_key] = jax.device_put(inputs, in_shard_test)
        self.golden_inputs[input_key] = jax.device_put(inputs, in_shard_golden)

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

class TestConfigBuilder:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.params = {
            "batch_size": 1,
            "seq_len": 32,
            "input_dim": 4,
            "hidden_dim": 4,
            "num_experts": 4,
            "top_k" : 2,
            "num_groups": 1,
            "outer_batch": 1,
            "train_capacity_factor": 2,
            "mesh_spec": {}   # dict with keys from MESH_AXIS_NAMES, empty for single core test
        }
        return self
    
    def with_dimensions(self, batch_size, seq_len, input_dim, dtype):
        # Only two data types currently supported
        _dtype = jnp.float32
        if dtype == 'bfloat16':
            _dtype = jnp.bfloat16

        self.params.update({
            "batch_size": batch_size,
            "seq_len": seq_len,
            "input_dim": input_dim,
            "dtype": _dtype
        })
        return self
    
    def with_expert_settings(self, hidden_dim, outer_batch, num_groups, num_experts, top_k=2, train_capacity_factor=None):
        self.params.update({
            "hidden_dim": hidden_dim,
            "outer_batch" : outer_batch,
            "num_groups": num_groups,
            "num_experts": num_experts,
            "top_k": top_k,
            "train_capacity_factor": train_capacity_factor
        })
        return self
    
    def with_mesh_settings(self, mesh_spec):
        self.params.update({
            "mesh_spec": mesh_spec
        })
        return self 
    
    def build_moe_topkgather_setup(self):
        return {
            "input_dim": self.params["input_dim"],
            "hidden_dim": self.params["hidden_dim"],
            "num_experts": self.params["num_experts"],
            "num_groups": self.params["num_groups"],
            "outer_batch": self.params["outer_batch"],
            "dim_to_mesh_axis_map": MOE_DIM_TO_MESH_AXIS_MAP,
            "activation": ("nn.silu","linear"),
            "gating": TopKGatingGather.default_config().set(
                name="gating",
                top_k=self.params["top_k"],
                train_capacity_factor=self.params["train_capacity_factor"]
            ),
            # "norm" : RMSNorm.default_config().set(eps=1e-5, forward_dtype=None),
            "dropout" : Dropout.default_config().set(rate=None),
            "stochastic_depth" : StochasticDepth.default_config().set(rate=None)
        }
    
    def build_moe_top2_setup(self):
        return {
            "input_dim": self.params["input_dim"],
            "hidden_dim": self.params["hidden_dim"],
            "num_experts": self.params["num_experts"],
            "num_groups": self.params["num_groups"],
            "outer_batch": self.params["outer_batch"],
            "dim_to_mesh_axis_map": MOE_DIM_TO_MESH_AXIS_MAP,
            "activation": ("nn.silu","linear"),
            "gating": TopKGating.default_config().set(
                name="gating",
                top_k=self.params["top_k"],
                train_capacity_factor=self.params["train_capacity_factor"]
            )
        }
    
    def build_gating_setup(self):
        return {
            "num_experts": self.params["num_experts"],
            "top_k": self.params["top_k"],
            "train_capacity_factor": self.params["train_capacity_factor"]
        }
    
    def build_test_configs_integ(self):

        seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])

        test_configs = [] 
        test_configs.append(
            TestConfig(
                setup=[
                    self.build_moe_topkgather_setup(),
                    self.build_moe_topkgather_setup()
                ],
                test=ModuleConfig(TransformerFeedForwardMoE, "neuron", "MoE", self.params['dtype']),
                golden=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", self.params['dtype']),
                input_shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"]),
                loss_fn=lambda x: x.mean(),
                mesh_spec=self.params["mesh_spec"],
                prefix="_moe")
            )
        if not self.params["mesh_spec"]: # gating tests only for single-core config
            test_configs.append(
            TestConfig(
                setup=[
                    self.build_gating_setup(),
                    self.build_gating_setup()
                ],
                test=ModuleConfig(TopKGatingGather, "neuron", dtype=self.params['dtype']),
                golden=ModuleConfig(TopKGatingGather, "cpu", dtype=self.params['dtype']),
                input_shape=(self.params["outer_batch"], self.params["num_groups"], seq_len, self.params["num_experts"]),
                loss_fn=lambda x: x.load_balance_loss,
                mesh_spec=self.params["mesh_spec"],
                prefix="_gating")
            )
        return test_configs

    def build_test_configs_unit(self):

        seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])

        test_configs = [] 
        test_configs.append(
            TestConfig(
                setup=[
                    self.build_moe_topkgather_setup(),
                    self.build_moe_top2_setup()
                ],
                test=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", self.params['dtype']),
                golden=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", self.params['dtype']),
                input_shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"]),
                loss_fn=lambda x: x.mean(),
                mesh_spec=self.params["mesh_spec"],
                prefix="_moe"
                )
            )
        if not self.params["mesh_spec"]: # gating tests only for single-core config
            test_configs.append(
                TestConfig(
                    setup=[
                        self.build_gating_setup(),
                        self.build_gating_setup()
                    ],
                    test=ModuleConfig(TopKGatingGather, "cpu", dtype=self.params['dtype']),
                    golden=ModuleConfig(TopKGating, "cpu", dtype=self.params['dtype']),
                    input_shape=(self.params["outer_batch"], self.params["num_groups"], seq_len, self.params["num_experts"]),
                    conv_output=partial(_topkgather_to_topk, top_k=self.params["top_k"], cf=self.params["train_capacity_factor"]),
                    loss_fn=lambda x: x.load_balance_loss,
                    mesh_spec=self.params["mesh_spec"],
                    prefix="_gating"
                )
            )
        return test_configs
    
    def build_grid_space(self):
        # Grid space for testing

        grid_space = []

        # Custom Configs
        # b s i h e top_k g ob cf mesh dtype

        ## Presubmit Tests
        
        # Toy Config
        Mistral8x7B_toy_multi = (128,  256, 1024,  3584, 8, 2, 1, 1, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral8x7B_toy_multi)
        Mistral8x7B_toy_single = (128,  256, 64,  896, 8, 2, 1, 1, 2, {}, "bfloat16")
        grid_space.append(Mistral8x7B_toy_single)

        # 8B Configs
        Mistral8B_base = (16, 4096, 2048, 7168, 8, 2, 1, 4, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral8B_base)
        Mistral8B_top1_cap1 = (16, 4096, 2048, 7168, 8, 1, 1, 4, 1, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral8B_top1_cap1)
        Mistral8B_8k = (16, 8192, 2048, 7168, 8, 2, 1, 4, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral8B_8k)

        # 50B Config
        Mistral50B_base = (16, 2048, 4096, 14336, 8, 2, 1, 4, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral50B_base)

        # 150B Config
        Mistral150B_base = (16, 4096, 6144, 15360, 16, 4, 1, 4, 2, {"fsdp":-1, "model":4}, "bfloat16")
        grid_space.append(Mistral150B_base)

        return grid_space

    
def get_training_configs(is_unit: bool = False):
    builder = TestConfigBuilder()

    test_configs = []

    for (batch, seq, input_dim,  hidden_dim, n_experts, top_k, n_groups, out_batch, capacity_factor, mesh_spec, dtype) in builder.build_grid_space():
        
        if batch % out_batch != 0:
            continue
        
        if mesh_spec and batch < 16: # need large batch to parallelize
            batch = batch*16

        config = builder.reset()
        config = config.with_dimensions(batch, seq, input_dim, dtype)
        config = config.with_expert_settings(
            hidden_dim,
            out_batch,
            n_groups,
            n_experts,
            top_k,
            train_capacity_factor=capacity_factor
        )
        config = config.with_mesh_settings(mesh_spec)
        if is_unit:
            config = config.build_test_configs_unit()
        else:
            config = config.build_test_configs_integ()

        name = f"MoE_b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_g{n_groups}_ob{out_batch}_ec{capacity_factor}_mesh{mesh_spec}_dtype_{dtype}"
        test_configs.extend([(name + cfg.prefix, cfg) for cfg in config])

    return test_configs