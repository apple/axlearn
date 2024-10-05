# ML API Styles

Self-link: [https://tinyurl.com/ml-api-styles](https://tinyurl.com/ml-api-styles)

Ruoming Pang ([ruoming@gmail.com](mailto:ruoming@gmail.com))
With feedback from Zhifeng Chen, Yonghui Wu, Drew Frank, Chris DuBois, and Tom Nickson

November 2021
*Status: Reviewed*

API design (including configuration) is an important but often overlooked aspect of machine learning (ML) development. A flexible, readable API helps speeding up iterations, reducing mistakes, and improving reproducibility.

This document discusses some common issues we have seen in API design, in particular, the author’s own mistakes, and offers corresponding recommendations. We do not claim originality of the recommendations, which came from many collaborators.

While these API design issues are often not specific to ML, this document focuses on ML development because (1) ML systems often have a lot of hyper-parameters, which bring a lot of complexity to their APIs; (2) ML systems tend to evolve rapidly to accommodate new ideas or requirements; (3) it’s useful to focus on a specific area with concrete examples.

[Use Configs to Customize Module Creation](#use-configs-to-customize-module-creation)

[Configure Object Instances, Not Object Types](#configure-object-instances,-not-object-types)

[Configuration of Composite Modules](#configuration-of-composite-modules)

[The Config Factory Pattern](#the-config-factory-pattern)

[Avoid Multiple Positional Arguments](#avoid-multiple-positional-arguments)

[Returning a Tuple of Values](#returning-a-tuple-of-values)

[Naming: Consistency and Generality](#naming:-consistency-and-generality)

[Keep Default Values Simple](#keep-default-values-simple)

[Booleans in API](#booleans-in-api)

## Use Configs to Customize Module Creation {#use-configs-to-customize-module-creation}

A ML system consists of a hierarchy of modules. It will bring a lot of flexibility if users can customize how each module is created. This can be achieved by using configs to capture the arguments used to create modules, as opposed to using ephemeral arguments, because config objects can be logged and manipulated. The config of a composite module can be specified with a hierarchy of configs of sub-modules (see “Configuration of Composite Modules” below).

DO NOT:

  encoder \= TransformerEncoder(num\_layers=..., input\_dim=...)

DO:

  encoder \= instantiate(config.encoder)

## Configure Object Instances, Not Object Types {#configure-object-instances,-not-object-types}

It is more flexible when each instance of an object type can be configured independently than to only make the object type configurable. This is following the same philosophy to minimize the use of global variables.

DO NOT:

  @configurable
  def my\_dropout(inputs, ratio):
    return ...

  my\_dropout.ratio \= 0.1  \# affects all users of my\_dropout

DO:

  my\_config.dropout.ratio \= 0.1  \# only affects my\_config

## Configuration of Composite Modules {#configuration-of-composite-modules}

When defining the config for a composite model, **avoid “leaking” configurations of sub-modules into the composite module**.

For example, suppose a Transformer encoder contains the following sub-modules: a token embedding layer, a positional embedding layer, and a stack of Transformer layers. One may be tempted to define the config as follows:

DO NOT:

  class TransformerEncoderConfig:
      \# The configuration of the embedding table.
      vocab\_size: int
      embedding\_dim: int

      \# The configuration of relative positional embeddings.
      num\_relative\_position\_buckets: int
      max\_relative\_position\_distance: int

      \# The configuration of transformer layers.
      num\_transformer\_layers: int
      num\_attention\_heads: int
      feed\_forward\_hidden\_dim: int
      dropout: float

DO:

  class TransformerEncoderConfig:
      token\_embedding: TokenEmbeddingLayer.Config
      positional\_embedding: PositionalEmbeddingLayer.Config
      num\_transformer\_layers: int
      transformer\_layer: TransformerLayer.Config

The recommended style does not bind the composite layer configuration to specific implementations of sub-modules such as a relative positional embedding layer that supports num\_buckets and max\_distance and thus allows users to experiment with different types of sub-modules.

On the other hand, configurations that determine a module's input/output shapes should be considered part of the module API and therefore defined in the module's own config, rather than in its submodules' configs. For example,

DO NOT:

  class EncoderConfig:
      \# Let output\_proj.output\_dim determine the output dim of Encoder.
      \# This is leaking the implementation details of the Encoder,
      \# because its users need to assume that output\_proj is the final layer
      \# of the Encoder.
      output\_proj: Linear.Config

DO:

  class EncoderConfig:
      output\_dim: int
      output\_proj: Linear.Config

      def \_\_init\_\_(...):
          cfg \= self.config
          \# Set \`output\_proj.output\_dim\` to cfg.output\_dim.
          self.\_add\_child("output\_proj",
                          cfg.output\_proj.set(output\_dim=cfg.output\_dim))

Further, if a dim is shared by multiple sub-modules, e.g., as the output\_dim of one module and the input\_dim of the subsequent module, it should also be configured in the parent module config.

## The Config Factory Pattern {#the-config-factory-pattern}

Config factory is a useful design pattern to decouple the API of general-purpose modules from specific use cases.

For example, the recommended style of TransformerEncoder configuration does not assume any specific type of positional embedding or dropout ratio, but for specific models we can define a narrower and more concise API with assumptions on specific implementation choices. For example, MyEncoder can be configured by:

  class MyEncoderConfig:
    model\_dim \= 512
    num\_layers \= 6
    dropout \= 0.1  \# MyEncoder uses dropout=0.1 by default

  def build(my\_config: MyEncoderConfig) \-\> TransformerEncoderConfig:
    """Translates a special-purpose MyEncodeConfig
       ...to a general-purpose TransformerEncoderConfig.
    """
    config \= TransformerEncoderConfig()
    for i in range(my\_config.num\_layers):
      \# All layers in MyEncoder have input\_dim=output\_dim=model\_dim.
      layer\_config \= TransformerLayerConfig(
          input\_dim=my\_config.model\_dim, output\_dim=my\_config.model\_dim)

      \# MyEncoder’s feed-forward hidden dim is proportional to its model\_dim.
      layer\_config.feed\_forward.hidden\_dim \= my\_config.model\_dim \* 16

      \# MyEncoder uses the same dropout for attention and feed-forward.
      layer\_config.attention.dropout \= my\_config.dropout
      layer\_config.feed\_forward.dropout \= my\_config.dropout

      config.layers.append(layer\_config)
    return config

Notice that the factory builds the encoder config object, rather than the encoder directly. This allows its users to further customize it and log the output config if necessary.

## Avoid Multiple Positional Arguments {#avoid-multiple-positional-arguments}

Avoid long lists of positional arguments. Positional arguments make call sites harder to read. They also make it harder to extend the API with new arguments in the most natural ordering. Instead, **limit the number of positional arguments to \<= 1 and use keyword arguments for the rest**.

DO NOT:

  def attention(query, key, value, query\_mask, kv\_mask):

DO:

  def attention(\*, query, key, value, query\_mask, kv\_mask):

## Returning a Tuple of Values {#returning-a-tuple-of-values}

When a function returns multiple values, **prefer returning a dict, or even better, a dataclass or namedtuple over returning a tuple**. This allows the function to be extended to return more values in the future without breaking existing callers. Named return fields also make the code less error-prone.

In particular, we should avoid returning a tuple of dynamic number of elements depending on input arguments. For example:

DO NOT:

  return (data, mask, atten\_probs) if return\_atten\_probs else (data, mask)

DO:

  return Output(data=data, mask=mask,
                atten\_probs=atten\_probs if return\_atten\_probs else None)

## Naming: Consistency and Generality {#naming:-consistency-and-generality}

Respect the conventions of the repository, which should take precedence over personal preferences. Sometimes there are multiple reasonable names, e.g., “input\_dim”, “num\_input\_channels”, “num\_input\_nodes”---pick one consistently across modules. Also pay attention to details such as singular vs. plural, e.g., “padding” vs. “paddings”.

Prefer generic names over implementation specific ones, e.g., “normalization\_layer” instead of “layer\_norm” since in the future we may want to use a normalization layer other than LayerNorm.

## Keep Default Values Simple {#keep-default-values-simple}

For general-purpose modules, the default configuration values should reflect the simplest scenarios, not hyperparameters of specific models. Optional features such as dropout should be disabled by default.

DO NOT:

  dropout \= 0.1,

DO:

  dropout \= 0,

Or require the user to set it explicitly:

  dropout \= None,  \# must be set explicitly.

A potential downside of decoupling the API from specific models is that configurations tend to get more verbose. Consider applying the “config factory” pattern to this problem.

## Booleans in API {#booleans-in-api}

Boolean arguments in the API makes it difficult to extend. **It is often better to use enums instead of booleans.**

DO NOT:

  \# If true, use GeLU as the activation function. Otherwise use ReLU.
  use\_gelu: bool \= False,

Boolean args can lead to an awkward API when we need to extend it to accommodate more than two possibilities :

  \# If true, use SeLU as the activation function. Otherwise use ReLU.
  \# WARNING: use\_gelu and use\_selu cannot be true at the same time.
  use\_selu: bool \= False,

DO:

  activation\_function \= RELU,
