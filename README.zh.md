# AXLearn 深度学习库

[![build-and-test](https://github.com/apple/axlearn/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/apple/axlearn/actions/workflows/build.yml)

<p>
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

**本代码库目前处于活跃开发阶段，API 可能会发生变更。**

## 目录

| 章节 | 说明 |
| - | - |
| [简介](#简介) | 什么是 AXLearn？ |
| [快速入门](docs/01-start.md) | 快速安装并启动运行 AXLearn。 |
| [核心概念](docs/02-concepts.md) | 核心概念与架构设计原则。 |
| [CLI 用户指南](docs/03-cli.md) | 如何使用命令行工具（CLI）。 |
| [基础设施](docs/04-infrastructure.md) | 核心基础设施组件。 |

## 简介

AXLearn 是一个构建在 [JAX](https://jax.readthedocs.io/) 和 [XLA](https://www.tensorflow.org/xla) 之上的深度学习库，旨在支持大规模深度学习模型的研发。

AXLearn 采用面向对象的方法来应对构建、迭代和维护模型过程中面临的软件工程挑战。该库的配置系统允许用户通过可复用的构建模块组合模型，并与 [Flax](https://flax.readthedocs.io/) 和 [Hugging Face transformers](https://github.com/huggingface/transformers) 等主流深度学习库无缝集成。

AXLearn 专为超大规模扩展而生。它支持在数千个硬件加速器上以高利用率训练高达数千亿参数的模型。同时，它专为在公有云上运行而设计，并提供用于部署和管理训练任务与数据的工具。AXLearn 构建于 [GSPMD](https://arxiv.org/abs/2105.04663) 之上，采用全局计算范式（Global Computation Paradigm），使用户能够在虚拟全局计算机上描述计算逻辑，而无需针对单个加速器逐一编写底层调度代码。

AXLearn 广泛支持自然语言处理、计算机视觉和语音识别等多种应用场景，并内置了用于训练前沿（State-of-the-art）模型的基线配置。

如需详细了解 AXLearn 的核心组件与架构设计，请参阅[核心概念](docs/02-concepts.md)；如果您希望立即动手实践，请参阅[快速入门](docs/01-start.md)。

## PyPI 发布版本

有关 PyPI 发布流程、版本管理以及安装说明，请参阅 [RELEASE.md](RELEASE.md)。

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月13日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
