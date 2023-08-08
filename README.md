# The AXLearn Library for Deep Learning

AXLearn is a library built on top of [JAX](https://jax.readthedocs.io/) and
[XLA](https://www.tensorflow.org/xla) to support development of large-scale deep learning models.

AXLearn takes an object-oriented approach to the software engineering challenges that arise from
building, iterating, and maintaining models.
The configuration system of the library lets users compose models from reusable building blocks and
integrate with other libraries such as [Flax](https://flax.readthedocs.io/) and
[Hugging Face transformers](https://github.com/huggingface/transformers).

AXLearn is built to scale.
It supports training of models with up to hundreds of billions of parameters across thousands of
accelerators at high utilization.
It is also designed to run on public clouds and provides tools to deploy and manage jobs and data.
Built on top of [GSPMD](https://arxiv.org/abs/2105.04663), AXLearn adopts a global computation
paradigm to allow users to describe computation on a virtual global computer rather than on a
per-accelerator basis.

AXLearn supports a wide range of applications, including natural language processing, computer
vision, and speech recognition and contains baseline configurations for training state-of-the-art
models.
