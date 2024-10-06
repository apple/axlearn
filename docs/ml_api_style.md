# ML API Styles

_Originally published at https://tinyurl.com/ml-api-styles, Nov 2021._

API design (including configuration) is an important but often overlooked aspect of machine
learning (ML) development. A flexible, readable API helps speeding up iterations, reducing mistakes,
and improving reproducibility.

This document discusses some common issues we have seen in API design, in particular, the author’s
own mistakes, and offers corresponding recommendations. We do not claim originality of the
recommendations, which came from many collaborators.

While these API design issues are often not specific to ML, this document focuses on ML development
because (1) ML systems often have a lot of hyper-parameters, which bring a lot of complexity to
their APIs; (2) ML systems tend to evolve rapidly to accommodate new ideas or requirements; (3) it’s
useful to focus on a specific area with concrete examples.

[Use Configs to Customize Module Creation](#use-configs-to-customize-module-creation)

[Configure Object Instances, Not Object Types](#configure-object-instances-not-object-types)

[Configuration of Composite Modules](#configuration-of-composite-modules)

[The Config Factory Pattern](#the-config-factory-pattern)

[Avoid Multiple Positional Arguments](#avoid-multiple-positional-arguments)

[Avoid Returning a Tuple of Values](#avoid-returning-a-tuple-of-values)

[Naming: Consistency and Generality](#naming-consistency-and-generality)

[Keep Default Values Simple](#keep-default-values-simple)

[Avoid Booleans in APIs](#avoid-booleans-in-apis)

## Use Configs to Customize Module Creation

A ML system consists of a hierarchy of modules. It will bring a lot of flexibility if users can
customize how each module is created. This can be achieved by using configs to capture the arguments
used to create modules, as opposed to using ephemeral arguments, because config objects can be
logged and manipulated. The config of a composite module can be specified with a hierarchy of
configs of sub-modules (see “Configuration of Composite Modules” below).

DO NOT:

```python
encoder = TransformerEncoder(num_layers=..., input_dim=...)
```

DO:

```python
encoder = instantiate(encoder_config)
```

## Configure Object Instances, Not Object Types

It is more flexible when each instance of an object type can be configured independently than to
only make the object type configurable. This is following the same philosophy to minimize the use of
global variables.

DO NOT:

```python
@configurable
def my_dropout(inputs, ratio):
    return ...

my_dropout.ratio = 0.1  # affects all users of my_dropout
```

DO:

```python
my_config.dropout.ratio = 0.1  # only affects my_config
```

## Configuration of Composite Modules

When defining the config for a composite model, **avoid “leaking” configurations of sub-modules into
the composite module**.

For example, suppose a Transformer encoder contains the following sub-modules: a token embedding
layer, a positional embedding layer, and a stack of Transformer layers. One may be tempted to define
the config as follows:

DO NOT:

```python
class TransformerEncoderConfig:
    # The configuration of the embedding table.
    vocab_size: int
    embedding_dim: int

    # The configuration of relative positional embeddings.
    num_relative_position_buckets: int
    max_relative_position_distance: int

    # The configuration of transformer layers.
    num_transformer_layers: int
    num_attention_heads: int
    feed_forward_hidden_dim: int
    dropout: float
```

DO:

```python
class TransformerEncoderConfig:
    token_embedding: TokenEmbeddingLayer.Config
    positional_embedding: PositionalEmbeddingLayer.Config
    num_transformer_layers: int
    transformer_layer: TransformerLayer.Config
```

The recommended style does not bind the composite layer configuration to specific implementations of
sub-modules such as a relative positional embedding layer that supports num_buckets and
max_distance and thus allows users to experiment with different types of sub-modules.

On the other hand, configurations that determine a module's input/output shapes should be considered
part of the module API and therefore defined in the module's own config, rather than in its
submodules' configs. For example,

DO NOT:

```python
class EncoderConfig:
    # Let output_proj.output_dim determine the output dim of Encoder.
    # This is leaking the implementation details of the Encoder,
    # because its users need to assume that output_proj is the final layer
    # of the Encoder.
    output_proj: Linear.Config
```

DO:

```python
class EncoderConfig:
    output_dim: int
    output_proj: Linear.Config

    def __init__(self, ...):
        cfg = self.config
        # Set `output_proj.output_dim` to cfg.output_dim.
        self._add_child("output_proj",
                        cfg.output_proj.set(output_dim=cfg.output_dim))
```

Further, if a dim is shared by multiple sub-modules, e.g., as the output_dim of one module and the
input_dim of the subsequent module, it should also be configured in the parent module config.

## The Config Factory Pattern

Config factory is a useful design pattern to decouple the API of general-purpose modules from
specific use cases.

For example, the recommended style of TransformerEncoder configuration does not assume any specific
type of positional embedding or dropout ratio, but for specific models we can define a narrower and
more concise API with assumptions on specific implementation choices. For example, MyEncoder can be
configured by:

```python
class MyEncoderConfig:
    model_dim = 512
    num_layers = 6
    dropout = 0.1 # MyEncoder uses dropout=0.1 by default

    def build(my_config: MyEncoderConfig) -> TransformerEncoderConfig:
        """Translates a special-purpose MyEncodeConfig

        ...to a general-purpose TransformerEncoderConfig.
        """
        config = TransformerEncoderConfig()
        for i in range(my_config.num_layers):
            # All layers in MyEncoder have input_dim=output_dim=model_dim.
            layer_config = TransformerLayerConfig(
                input_dim=my_config.model_dim, output_dim=my_config.model_dim)

        # MyEncoder’s feed-forward hidden dim is proportional to its model_dim.
        layer_config.feed_forward.hidden_dim = my_config.model_dim * 16

        # MyEncoder uses the same dropout for attention and feed-forward.
        layer_config.attention.dropout = my_config.dropout
        layer_config.feed_forward.dropout = my_config.dropout

        config.layers.append(layer_config)
        return config
```

Notice that the factory builds the encoder config object, rather than the encoder directly. This
allows its users to further customize it and log the output config if necessary.

## Avoid Multiple Positional Arguments

Avoid long lists of positional arguments. Positional arguments make call sites harder to read. They
also make it harder to extend the API with new arguments in the most natural ordering. Instead,
**limit the number of positional arguments to <= 1 and use keyword arguments for the rest**.

DO NOT:

```python
def attention(query, key, value, query_mask, kv_mask):
```

DO:

```python
def attention(*, query, key, value, query_mask, kv_mask):
```

## Avoid Returning a Tuple of Values

When a function returns multiple values, **prefer returning a dict or a dataclass
over returning a tuple**. This allows the function to be extended to return more values
in the future without breaking existing callers. Named return fields also make the code less
error-prone.

In particular, we should avoid returning a tuple of dynamic number of elements depending on input
arguments. For example:

DO NOT:

```python
return (data, mask, atten_probs) if return_atten_probs else (data, mask)
```

DO:

```python
return Output(data=data, mask=mask, atten_probs=atten_probs if return_atten_probs else None)
```

## Naming: Consistency and Generality

Respect the conventions of the repository, which should take precedence over personal preferences.
Sometimes there are multiple reasonable names, e.g., “input_dim”, “num_input_channels”,
“num_input_nodes”---pick one consistently across modules. Also pay attention to details such as
singular vs. plural, e.g., “padding” vs. “paddings”.

Prefer generic names over implementation specific ones, e.g., “norm" instead of
“layer_norm” since in the future we may want to use a normalization layer other than LayerNorm.

## Keep Default Values Simple

For general-purpose modules, the default configuration values should reflect the simplest scenarios,
not hyperparameters of specific models. Optional features such as dropout should be disabled by
default.

DO NOT:

```python
dropout = 0.1
```

DO:

```python
dropout = 0
```

Or require the user to set it explicitly:

```python
dropout = None  # must be set explicitly.
```

A potential downside of decoupling the API from specific models is that configurations tend to get
more verbose. Consider applying the “config factory” pattern to this problem.

## Avoid Booleans in APIs

Boolean arguments in the API makes it difficult to extend. **It is often better to use enums instead
of booleans.**

DO NOT:

```python
# If true, use GeLU as the activation function. Otherwise use ReLU.
use_gelu: bool = False
```

Boolean args can lead to an awkward API when we need to extend it to accommodate more than two
possibilities:

```python
# If true, use SeLU as the activation function. Otherwise use ReLU.
# WARNING: use_gelu and use_selu cannot be true at the same time.
use_selu: bool = False
```

DO:

```python
activation_function = RELU
```
