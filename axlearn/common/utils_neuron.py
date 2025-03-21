
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

from axlearn.common.module import functional as F
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
    def __init__(self, module = None, device = "cpu", layer = None, config=None):
        assert module is not None
        self.module = module.default_config().set(name="test")
        if config:
            for k in config:
                setattr(self.module, k, config[k])
        self.device = device
        self.layer = layer # None for topk, else "MoE"

class TestConfig():
    def __init__(self, test: ModuleConfig, golden: ModuleConfig = None, 
                 input_shape: tuple = None, loss_fn = None, conv_output = None, 
                 mesh_spec: dict = None, prefix = None):
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
        if test.layer == "MoE":
            self.set_outer_batch()

    def init(self):
        self.instantiate_modules_with_mesh() 
        self.random_inputs_with_mesh()
        #specify empty outsharding for jax.jit because we transfer tensors to host and compare
        out_pspec = PartitionSpec()  
        self.out_shard_test = NamedSharding(mesh=self.mesh_test, spec = out_pspec) 
        self.out_shard_golden = NamedSharding(mesh=self.mesh_golden, spec = out_pspec)

    def get_mesh_from_spec(self, mesh_spec):  
        mesh = mesh_shape_from_axes(**mesh_spec)
        mesh = infer_mesh_shape(mesh, num_devices=self.num_devices) 
        self.num_devices = math.prod(mesh)
        return mesh 

    def set_outer_batch(self):
        outer_batch = get_outer_batch_from_mesh(MESH_AXIS_NAMES, MOE_OUTER_BATCH_AXIS_NAMES, self.mesh_dims)
        setattr(self.test.module, "outer_batch", outer_batch)
        setattr(self.golden.module, "outer_batch", outer_batch)

    def instantiate_modules_with_mesh(self):
        device_type = self.test.device
        devices = jax.devices(device_type)[:self.num_devices]
        self.mesh_test = Mesh(mesh_utils.create_device_mesh(self.mesh_dims, devices=devices), MESH_AXIS_NAMES) 
        with self.mesh_test: 
            self.test_layer  = self.test.module.instantiate(parent=None) 
            self.test_state  = self.test_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123)) 

        device_type = self.golden.device
        devices = jax.devices(device_type)[:self.num_devices]
        self.mesh_golden = Mesh(mesh_utils.create_device_mesh(self.mesh_dims, devices=devices), MESH_AXIS_NAMES) 
        with self.mesh_golden: 
            self.golden_layer  = self.golden.module.instantiate(parent=None) 
            self.golden_state  = self.golden_layer.initialize_parameters_recursively(prng_key=jax.random.PRNGKey(123)) 

    def random_inputs_with_mesh(self):
        input_key = 'inputs' if self.test.layer == "MoE" else 'logits'
        pspec = PartitionSpec(('data','fsdp'), None, 'model') if self.test.layer == "MoE" else PartitionSpec() 

        in_shard_test = NamedSharding(mesh=self.mesh_test, spec = pspec) 
        in_shard_golden = NamedSharding(mesh=self.mesh_golden, spec = pspec)

        with jax.default_device(jax.devices("cpu")[0]):    # create tensors on host to avoid OOM  
            inputs = jax.random.uniform(jax.random.PRNGKey(1), shape=self.input_shape) 
        
        inputs = jax.device_get(inputs)   # device_put seg-faults without this 
        self.test_inputs[input_key] = jax.device_put(inputs, in_shard_test)
        self.golden_inputs[input_key] = jax.device_put(inputs, in_shard_golden)

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
            "expert_capacity": 1000,
            "train_capacity_factor": None,
            "use_blockwise_kernel": False,
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
    
    def with_expert_settings(self, hidden_dim, outer_batch, num_groups, num_experts, expert_capacity, train_capacity_factor=None, use_blockwise_kernel=False, block_size=None):
        self.params.update({
            "hidden_dim": hidden_dim,
            "outer_batch" : outer_batch,
            "num_groups": num_groups,
            "num_experts": num_experts,
            "expert_capacity": expert_capacity,
            "train_capacity_factor": train_capacity_factor,
            "use_blockwise_kernel": use_blockwise_kernel,
            "block_size": block_size,
        })
        return self
    
    def with_mesh_settings(self, mesh_spec):
        self.params.update({
            "mesh_spec": mesh_spec
        })
        return self 

    def build_moe_topkgather_setup(self):
        print(self.params["use_blockwise_kernel"])
        if self.params["use_blockwise_kernel"] is False:
            gating_config = TopKGatingGather.default_config().set(
                    name="gating",
                    expert_capacity=self.params["expert_capacity"],
                    train_capacity_factor=self.params["train_capacity_factor"],
            )
        else:
            gating_config = TopKGatingGatherBlockwise.default_config().set(
                    name="gating",
                    expert_capacity=self.params["expert_capacity"],
                    train_capacity_factor=self.params["train_capacity_factor"],
                    block_size=self.params["block_size"],
            )
        
        return {
            "input_dim": self.params["input_dim"],
            "hidden_dim": self.params["hidden_dim"],
            "num_experts": self.params["num_experts"],
            "num_groups": self.params["num_groups"],
            "outer_batch": self.params["outer_batch"],
            "dim_to_mesh_axis_map": MOE_DIM_TO_MESH_AXIS_MAP,
            "gating": gating_config,
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
            "gating": TopKGating.default_config().set(
                name="gating",
                top_k=self.params["top_k"],
                train_capacity_factor=self.params["train_capacity_factor"]
            )
        }
    
    def build_gating_setup(self, gating_class=Top2Gating):
        
        x = {
            "expert_capacity": self.params["expert_capacity"],
            "num_experts": self.params["num_experts"],
            "train_capacity_factor": self.params["train_capacity_factor"],
        }
        if gating_class == TopKGatingGatherBlockwise:
            x["block_size"] = self.params["block_size"]
        return x
            
    def build_moe_layer_config(self, test_device="neuron"):
        return TestConfig(
            test=ModuleConfig(TransformerFeedForwardMoE, test_device, "MoE", config=self.build_moe_topkgather_setup()),
            golden=ModuleConfig(TransformerFeedForwardMoE, "cpu", "MoE", config=self.build_moe_top2_setup()),
            input_shape=(self.params["batch_size"], self.params["seq_len"], self.params["input_dim"]),
            loss_fn=lambda x: x.mean(),
            mesh_spec=self.params["mesh_spec"],
            prefix="_moe"
        )

    def build_gating_layer_config(self, test_device="neuron", conv_output=True):
        seq_len = (self.params["batch_size"]*self.params["seq_len"])//(self.params["outer_batch"] * self.params["num_groups"])
        test_gating_class = TopKGatingGatherBlockwise if self.params["use_blockwise_kernel"] else TopKGatingGather
        conv_output=partial(_topkgather_to_topk, expert_cap=self.params["expert_capacity"]) if conv_output else None
        return TestConfig(
            test=ModuleConfig(test_gating_class, test_device, layer=None, config=self.build_gating_setup(gating_class=test_gating_class)),
            golden=ModuleConfig(Top2Gating, "cpu", layer=None, config=self.build_gating_setup(gating_class=Top2Gating)),
            input_shape=(self.params["outer_batch"], self.params["num_groups"], seq_len, self.params["num_experts"]),
            conv_output=conv_output,
            loss_fn=lambda x: x.load_balance_loss,
            mesh_spec=self.params["mesh_spec"],
            prefix="_gating"
        )
    
    def build_test_configs_unit(self):
        test_configs = [] 
        test_configs.append(self.build_moe_layer_config(test_device="cpu"))
        if not self.params["mesh_spec"]: # gating tests only for single-core config
            test_configs.append(self.build_gating_layer_config(test_device="cpu"))
        return test_configs
        
    def build_test_configs_integ(self):
        test_configs = [] 
        test_configs.append(self.build_moe_layer_config())
        if not self.params["mesh_spec"]: # gating tests only for single-core config
            test_configs.append(self.build_gating_layer_config())
            # does conv_output need to be removed for neuron
        return test_configs

    def build_grid_space(self):
        # Grid space for testing
        # batchs =            [1, 4]
        # seqs =              [16, 128]
        # input_dims =        [64]
        # hidden_dims =       [128]
        # num_experts =       [2, 8]
        # num_groups =        [1, 4]
        # outer_batches =     [1, 2]
        # expert_capacities = [2, 1000]
        # mesh_specs       =  [{}, {"fsdp":-1, "model":4}]  #empty spec for single-core

        grid_space = [] 
        # grid_space = list(product(batchs, seqs, input_dims, hidden_dims, num_experts, 
        #                           num_groups, outer_batches, expert_capacities, mesh_specs))

        # Custom Configs
        # b s i h e g ob ec        
        # grid_space.extend([(2, 100, 64, 128, 2, 1, 1, 5)])
        # 
        Mistral8x7B_toy = (2,  256, 1024,  3584, 8, 1, 1, 32, {"fsdp":-1, "model":4})
        grid_space.append(Mistral8x7B_toy) 

        return grid_space

    def build_configs(self, batch, seq, input_dim,  hidden_dim, n_experts, n_groups, out_batch, capacity, mesh_spec, use_blockwise_kernel=False, is_unit=False):
        
        if mesh_spec and batch < 16: # need large batch to parallelize
            batch = batch*16 
        
        capacity_factor = 2 if not capacity else None 

        self.reset()
        self.with_dimensions(batch, seq, input_dim)
        self.with_expert_settings(
            hidden_dim,
            out_batch,
            n_groups,
            n_experts,
            expert_capacity=capacity,
            use_blockwise_kernel=use_blockwise_kernel,
            train_capacity_factor=capacity_factor
        )
        self.with_mesh_settings(mesh_spec)
        if is_unit:
            configs = self.build_test_configs_unit()
        else:
            configs = self.build_test_configs_integ()
        return configs
    
def get_training_configs(is_unit: bool = False):
    builder = TestConfigBuilder()

    test_configs = []

    for (batch, seq, input_dim,  hidden_dim, n_experts, n_groups, out_batch, capacity, mesh_spec) in builder.build_grid_space():
        if batch % out_batch != 0:
            continue
        config = builder.build_configs(batch, seq, input_dim,  hidden_dim, n_experts, n_groups, out_batch, capacity, mesh_spec, is_unit=is_unit)
        name = f"MoE_b{batch}_s{seq}_i{input_dim}_h{hidden_dim}_e{n_experts}_g{n_groups}_ob{out_batch}_ec{capacity}_mesh{mesh_spec}"
        test_configs.extend([(name + cfg.prefix, cfg) for cfg in config])

    return test_configs

class ModuleTester:
    def _fwd_call(self, layer, state, inputs):
        return F(
                layer,
                is_training=True,
                prng_key=jax.random.PRNGKey(123),
                state=state,
                inputs=inputs,
        )

    def _test_fwd_internal(self, cfg, assert_outputs=True):
        cfg.init()
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
        if assert_outputs:
            self.assertNestedAllClose(jax.device_get(test_output), jax.device_get(golden_output))

    def _test_bwd_internal(self, cfg):
        cfg.init()
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