# Concepts in the AXLearn Library

## Table of Contents

| Section | Description |
| - | - |
| [Config Library](#introduction-to-the-config-library) | The AXLearn Config Library. |
| [Module Hierarchy](#introduction-to-the-module-hierarchy) | AXLearn Modules and the Invocation Stack. |
| [SPMD Trainer](#spmd-trainer) | The SPMD Trainer. |

<br>

## Introduction to the Config Library

AXLearn is designed with composability in mind: one should be able to design complex ML models and experiments by combining reusable building blocks, either defined in AXLearn or in the broader ML ecosystem.

To use a concrete example, consider the definition of the `TransformerAttentionLayer` (as seen in the original [`Transformer`](https://arxiv.org/abs/1706.03762) architecture):

https://github.com/apple/axlearn/blob/68f1200547254d630c2e3c239ff463cd175317cf/axlearn/common/attention.py#L2005-L2034

Roughly, the layer is composed of a normalization layer, an attention implementation, and regularization layers.

By default, these are configured for the vanilla Transformer architecture (i.e., using `LayerNorm` and `MultiheadAttention`), but one can imagine adopting [GroupedQueryAttention (GQA)](https://arxiv.org/pdf/2305.13245.pdf) instead of `MHA` by swapping the `attention` layer implementation, or using [`RMSNorm`](https://arxiv.org/abs/1910.07467) instead of `LayerNorm` by swapping the `norm` layer implementation:

```python
# An example of configuring GQA and RMSNorm.

layer_cfg = TransformerAttentionLayer.default_config().set(
   attention=GroupedQueryAttention.default_config(),
   norm=RMSNorm.default_config(),
)
```

Above, the `default_config()` classmethod is used to construct a **config instance** for an AXLearn layer. This instance is a _partially specified_ object -- that is, not all properties of the layer need to be known at the time of config creation.

The user can incrementally `set()` attributes of the config, possibly using standard Python constructs like functions, loops, etc. For example, we can build a Transformer stack:

```python
# A contrived example of building a Transformer stack.

stack_cfg = StackedTransformerLayer.default_config().set(num_layers=12)

layer_cfgs = []
for i in range(stack_cfg.num_layers):
   layer_cfgs.append(build_layer_config(i))

stack_cfg.set(layer=layer_cfgs)
```

Once a config is fully specified, the user can materialize the layer by invoking `instantiate()`:

```python
# An example of instantiating a layer from a config.

stack: StackedTransformerLayer = stack_cfg.instantiate(...)
```

The config library will validate that all required fields have been set. If successful, the result will be an instance of the `StackedTransformerLayer` class. One can view `stack_cfg` as an object factory, where `instantiate()` produces unique class instances configured by `stack_cfg`.

Hopefully, this gives some basic intuition about how AXLearn leverages composition for building complex models and experiments. More details on the [config API](#config-apis) below.

<br>

### Configurable Functions and Classes

AXLearn is also designed to be an open system. To this end, **the config library supports configuring arbitrary Python functions and classes**, not just AXLearn modules.

To demonstrate what this means, consider a situation where we have some third-party Transformer layer implementation **not** written for AXLearn (such as [Hugging Face transformers](https://github.com/huggingface/transformers)):

```python
# A contrived example of a third-party Transformer layer implementation.

class ThirdPartyTransformerLayer:

   def __init__(self, config: PretrainedConfig, ...):
      ...
```

Using the [`config_for_class`](#config_for_class) utility, we can dynamically generate a config from the class `__init__` signature:

```python
# Generate a config using `config_for_class`.
custom_layer_cfg = config_for_class(ThirdPartyTransformerLayer)

# Set some of the attribute(s).
custom_layer_cfg.set(config=PretrainedConfig(...))
```

This allows composing the layer with an AXLearn model, such as the `StackedTransformerLayer` from above:

```python
stack_cfg = StackedTransformerLayer.default_config().set(
   layer=custom_layer_cfg,
)
```

Alternatively, using the [`config_for_function`](#config_for_function) utility, we can dynamically generate a config from a function signature:

```python
def layer_from_pretrained(config: PretrainedConfig, ...):
   return ThirdPartyTransformerLayer.from_pretrained(config, ...)

# Generate a config using `config_for_function`.
custom_layer_cfg = config_for_function(layer_from_pretrained)

# Set some of the attribute(s).
custom_layer_cfg.set(config=PretrainedConfig(...))

# Compose with other configs.
stack_cfg = StackedTransformerLayer.default_config().set(
   layer=custom_layer_cfg,
)
```

In general, as long as layer implementations abide by the same config APIs, they are interoperable in the config system. For a concrete example of interoperating with third-party libraries, see [`axlearn.huggingface.HfModuleWrapper`](https://github.com/apple/axlearn/blob/68f1200547254d630c2e3c239ff463cd175317cf/axlearn/huggingface/hf_module.py#L69-L108), which wraps Hugging Face Flax modules to be used within AXLearn.

<br>

### Beyond Machine Learning

Note that while we have provided examples of the config library in the context of neural network layers, **the config library itself is agnostic to ML applications**.

Indeed, many of AXLearn's cloud infrastructure components are also configured in a similar way as the layers above. See the base [`axlearn.cloud.common.Job`](https://github.com/apple/axlearn/blob/c84f50e6cba467ce5c2096d0cba3dce4c73f897a/axlearn/cloud/common/job.py#L16) definition as an example.

<br>

## Introduction to the Module Hierarchy

While configs offer a way to compose configurable objects into possibly complex hierarchies, they do not describe the behavior of these objects, such as:
- How the config values are used;
- How the configs are propagated from parent to child;
- What internal state each object is associated with (such as neural network weights).

Most AXLearn layers are implemented as subclasses of `Module`, which provides functionality to bridge these gaps.

### Module

A `Module` can be viewed abstractly as a node in an object tree. It has several key properties:
* Each `Module`, except the tree root, has a parent and zero or more children, which can be accessed
through the `parent` and `children` methods, respectively.
* A `Module` must have a `name` unique among its siblings. This allows a `Module` to have
a unique `path()` in its hierarchy.
* A `Module` is a subclass of `Configurable` and therefore is created by first building a
`Config` object.

The anatomy of a `Module` may be familiar if you have seen other layer definitions, either above or in [Getting Started](01-start.md):

https://github.com/apple/axlearn/blob/c84f50e6cba467ce5c2096d0cba3dce4c73f897a/axlearn/common/module.py#L401-L413

As we can see, a `Module` is associated with a `@config_class` describing the configurable attributes of the `Module`.

To create a root module, we first construct a config instance via `default_config()`. This allows us to configure the module as needed (e.g. as described [in the config introduction](#introduction-to-the-config-library)).
Once ready, we can call the `instantiate()` method to produce a `Module` instance.

```python
cfg: FooModule.Config = FooModule.default_config().set(name="foo", vlog=1, ...)
foo_module: FooModule = cfg.instantiate(parent=None)
```

To create a child module, use the `_add_child()` method, usually in the parent module's
`__init__()` method. For example, to create children with names "bar1" and "bar2" in a `FooModule`:

```python
class FooModule(Module):

   @config_class
   class Config(Module.Config):
      ...

   def __init__(self, ...):
      bar_cfg: BarModule.Config = ...

      # Add child modules by invoking `_add_child` with a name and a fully-specified config.
      self._add_child("bar1", bar_cfg)
      self._add_child("bar2", bar_cfg)
```

When the `cfg: FooModule.Config` is instantiated above, the _entire_ `FooModule` hierarchy is instantiated via these `_add_child()` calls.

Once a `Module` is constructed, we can access child modules by attribute access by name. By default, invoking a child module invokes its `forward` method, but we can also directly invoke other methods on these child modules.

```python
class FooModule(Module):
   ...

   def forward(self, x: Tensor) -> Tensor:
      # Child modules can be accessed as attributes.
      # By default, this is equivalent to `self.bar1.forward(x)`.
      y = self.bar1(x)

      # We can also invoke other methods on the child modules.
      z = self.bar2.my_method(y)

      # Return some outputs.
      return z
```

<br>

### Invoking Modules and the InvocationContext

In the spirit of JAX's functional API, `Module`s are themselves stateless.

When we invoke a method of a `Module`, the caller passes in "side inputs" such as module states (e.g., layer parameters), PRNG key, and possibly other information. The outputs include not only the method results, but also "side outputs" such as summaries and state updates.

The explicit specification of side inputs and outputs allows `Module` method invocations to be pure function calls and hence can be subject to JAX transformations such as `jax.grad`.

On the other hand, explicitly passing side inputs and outputs complicates the method APIs. To keep the API definition simple, we introduce the concept of `InvocationContext` to encapsulate the side inputs and outputs. When one Module method invokes another, `InvocationContext`s form a global (per-thread) stack. This is analogous to the traditional [call stack](https://en.wikipedia.org/wiki/Call_stack) that you may be familiar with.

> In most cases, you should not have to directly interact with `InvocationContext`s.

The `InvocationContext` has the following structure:

https://github.com/apple/axlearn/blob/c84f50e6cba467ce5c2096d0cba3dce4c73f897a/axlearn/common/module.py#L140-L153

As we can see, `InvocationContext` also forms a hierarchy, where each context except for the root context is associated with a parent.
Each context is also associated with a `Module`, `state`, and `output_collection`, which are analogous to a layer implementation, its corresponding layer weights, and auxiliary outputs which are not convenient to bubble up via the traditional call stack.

To invoke a `Module`, one must construct the root `InvocationContext`, commonly via the [`functional`](https://github.com/apple/axlearn/blob/c84f50e6cba467ce5c2096d0cba3dce4c73f897a/axlearn/common/module.py#L725) API:

```python
from axlearn.common.module import functional as F

# Invoke `foo_module.forward` via functional API.
outputs, output_collection = F(
   foo_module,
   # Specify parameters for `foo_module` and its descendants.
   state={"bar1": {...}, "bar2": {...}},
   # Specify inputs to `forward`.
   inputs={"x": ...},
   ...
)
```

As the invocation traverses down the module hierarchy (i.e. as we invoke methods on child modules), new `InvocationContext`s will be pushed to the stack; as the invocations return, `InvocationContext`s will be popped from the stack.

Note that the `output_collection` is accumulated throughout the entire module hierarchy and returned as an output of the `functional` API alongside the standard function return values. This makes it convenient to return values from arbitrary points in the module hierarchy, and is commonly used to log training-time summaries via `add_summary()`.

<br>

### BaseLayer

A `BaseLayer` is a type of `Module` with trainable parameters as `Module` states and provides convenience APIs to define such parameters, including how they are partitioned[^1] and initialized.

https://github.com/apple/axlearn/blob/c84f50e6cba467ce5c2096d0cba3dce4c73f897a/axlearn/common/base_layer.py#L129-L162

The layer parameters are represented by the type `Nested[Tensor]`, a nested `dict` hierarchy corresponding
to the module tree with tensors as leaf values.

> Since the parameters are technically not owned by the `BaseLayer` instance, it is possible to create multiple sets of parameters with a given layer instance and decide which set of parameters to use on each invocation.
This feature is often used to optimize quantization or sparsification of models.

The `initialize_parameters_recursively()` method returns a `Nested[Tensor]` with parameters
initialized according to the `param_init` field in the layer's config.
As the name implies, `initialize_parameters_recursively()` also invokes child layers'
`initialize_parameters_recursively()` methods.

To specify which parameters to create, a layer can override its `_create_layer_parameter_specs()`
method. For example, the `Linear` layer creates a `weight` parameter tensor and optionally a `bias`
tensor:

https://github.com/apple/axlearn/blob/c84f50e6cba467ce5c2096d0cba3dce4c73f897a/axlearn/common/layers.py#L544-L557

In many cases, these parameter specs have already been defined for you in the core AXLearn layers.

[^1]: See https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html for more information about
partition specification.

<br>

## SPMD Trainer

The `SpmdTrainer` is a `Module` that contains the entire module hierarchy required for training and
evaluating a model.

It is often the root module, consisting of the following child modules:
* A `model` module containing the layers.
* A `learner` module containing the optimizer, learning rate schedule, etc.
* An `input` module with a `dataset()` method that returns an iterator yielding input batches for training.
* Zero or more `evaler` modules representing evaluation.
* A `checkpointer` module for saving model checkpoints.
* A `summary_writer` module for writing tensorboard summaries.

### Input Batch Sharding

When using `SpmdTrainer`, it is common to read and process inputs across all processes and hosts.
For the most common use case where you want each process to have an equal portion of the input batch, this process is mostly transparent to the user.
For more complex use cases, it can be helpful to have a general idea of the what is happening behind the scenes.

When using AXLearn's support for TFDS inputs, the typical way input batch sharding works is:

1. You specify the split for the Tensorflow dataset you want each process to have either
   explicitly using the `read_config` option of `input_data.tfds_dataset()` or
   let it default to splitting evenly per process.
  https://github.com/apple/axlearn/blob/c00c632b99e6a2d87ee7ba94f295b39e0871a577/axlearn/common/input_tf_data.py#L205
  See `input_tf_data.tfds_read_config()` for an example of how to construct a suitable value for
  `read_config` that sets per-process splits.
https://github.com/apple/axlearn/blob/c00c632b99e6a2d87ee7ba94f295b39e0871a577/axlearn/common/input_tf_data.py#L87-L98
2. In each step, each process reads in the data specified by its split, but it is only a local array
   initially.
3. `SpmdTrainer` combines these local arrays into a globally sharded array using
   `utils.host_to_global_device_array()` before passing the global input batch to `_run_step()`.
https://github.com/apple/axlearn/blob/c00c632b99e6a2d87ee7ba94f295b39e0871a577/axlearn/common/trainer.py#L420
https://github.com/apple/axlearn/blob/c00c632b99e6a2d87ee7ba94f295b39e0871a577/axlearn/common/utils.py#L496

<br>

## Config APIs

### ConfigBase

`ConfigBase` is the base class in the config library.
It is usually not used directly, but through [`Configurable`](#configurable), [`config_for_class`](#config_for_class), or [`config_for_function`](#config_for_function).

Each subclass of `ConfigBase` is defined by a set of fields, where each field has a name, a value type, and a default value, which can be set to `REQUIRED` to indicate that user must set the value explicitly.

### Configurable

`Configurable` is the base class of `Module` and represents an object that can be created by
"instantiating" a `Configurable.Config` (aka `InstantiableConfig`) object.

A `Configurable.Config` therefore represents an object factory.
Here the object can be a `Module` or a third-party object, such as an optax optimizer.

### config_for_class

`config_for_class(cls)` inspects the `__init__()` signature of the given `cls` and
creates a config object that can be used to instantiate instances of type `cls`.

This allows users to specify how to create third-party objects that are not subclasses of
`Configurable`, such as Flax modules.

### config_for_function

`config_for_class(fn)` inspects the function signature of the given `fn` and creates a config
object that can be used to invoke `fn` with the given arguments.

This allows users to specify how to create third-party functions such as `optax.sgd`.
