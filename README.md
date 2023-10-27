# The AXLearn Library for Deep Learning

**This library is under active development and the API is subject to change.**

## Table of Contents

| Section | Description |
| - | - |
| [Introduction](#introduction) | What is AXLearn? |
| [Getting Started](docs/01-start.md) | Getting up and running with AXLearn. |
| [Concepts](docs/02-concepts.md) | Core concepts and design principles. |
| [CLI User Guide](docs/03-cli.md) | How to use the CLI. |
| [Infrastructure](docs/04-infrastructure.md) | Core infrastructure components. |

## Introduction

Welcome to AXLearn, a powerful deep learning library built on top of JAX and XLA. AXLearn is designed to support the development of large-scale deep learning models and offers a unique approach to tackling the software engineering challenges that arise during model development and maintenance.
AXLearn is a library built on top of [JAX](https://jax.readthedocs.io/) and
[XLA](https://www.tensorflow.org/xla) to support development of large-scale deep learning models.

AXLearn takes an object-oriented approach to the software engineering challenges that arise from
building, iterating, and maintaining models.
The configuration system of the library lets users compose models from reusable building blocks and
integrate with other libraries such as [Flax](https://flax.readthedocs.io/) and
[Hugging Face transformers](https://github.com/huggingface/transformers).

 -  AXLearn is built to scale.
    - AXLearn is a powerful machine learning tool that can train models with hundreds of billions of parameters across thousands of accelerators, ensuring high utilization. It offers efficient deployment and      management of jobs and data, utilizing GSPMD and adopting a global computation paradigm, allowing computation to be described on a virtual global computer.

    - It is also designed to run on public clouds and provides tools to deploy and manage jobs and data.
  Built on top of [GSPMD](https://arxiv.org/abs/2105.04663), AXLearn adopts a global computation
  paradigm to allow users to describe computation on a virtual global computer rather than on a
  per-accelerator basis.

 -  Versatile Applications.
     - AXLearn isn't limited to a specific domain. It supports a wide range of applications, including natural language processing, computer vision, and speech recognition. Additionally, it comes with baseline configurations for training state-of-the-art models.
For in-depth information on AXLearn's core components and design, please refer to the [Concepts](docs/02-concepts.md) section. If you're eager to dive right in, check out [Getting Started](docs/01-start.md).

## Getting Started
Ready to harness the power of AXLearn? The [Getting Started](docs/01-start.md) guide will walk you through the process of setting up AXLearn and embarking on your deep learning journey.

## Concepts
To truly grasp the core components and design principles of AXLearn, head over to the [Concepts](docs/02-concepts.md) section, where you'll find valuable insights into our library's inner workings.

## CLI User Guide
Learn how to make the most of AXLearn's Command Line Interface (CLI) in the [CLI User Guide](docs/03-cli.md).

## Infrastructure
Understand the key infrastructure components of AXLearn by exploring the [Infrastructure](docs/04-infrastructure.md) section. This is where you'll discover how AXLearn handles the heavy lifting behind the scenes.
