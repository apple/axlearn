# Getting Started

## Table of Contents

| Section | Description |
| - | - |
| [Installing Dependencies](#installing-dependencies) | Setup instructions. |
| [A Short Tutorial](#a-short-tutorial) | Writing an experiment from scratch. |
| [Launching Training](#launching-training) | Launching training with cloud TPUs. |
| [Next Steps](#next-steps) | Where to go next. |

<br>

## Installing Dependencies

The installation steps depend on your machine's hardware, and whether you plan to develop AXLearn.

### Pre-requisites

If you use an Intel (x86) machine, we recommend installing in a virtual environment, e.g. with [conda](https://conda.io).

```shell
conda create -n axlearn python=3.10
conda activate axlearn
```

If you use an Apple Silicon machine, please follow these instructions instead:

<details>
<summary>Expand for instructions</summary>

For Apple Silicon machines, we will install native versions of Python and Python packages using Miniforge.

We need Xcode to build packages like `tensorstore`. Please install Xcode from the App Store if you haven't already.

```shell
# Install the arm64 version of Miniforge3 + Python 3.10.
curl -L -o miniforge.sh https://github.com/conda-forge/miniforge/releases/download/24.7.1-0/Miniforge3-24.7.1-0-MacOSX-arm64.sh
bash miniforge.sh -u

# Create a conda environment.
conda create -n axlearn python=3.10
conda activate axlearn

# Install tensorflow following https://developer.apple.com/metal/tensorflow-plugin.
conda install -c apple tensorflow-deps

# If you do NOT have bazel installed.
# Note that the default ./oss_scripts/install_bazel.sh installs the x86 version.
brew install bazelisk

# Manually build tensorflow-text until a collaborator build is available.
# This was tested using clang version 15 - you may get non-working wheels with earlier versions of clang.
mkdir ~/builds && git clone https://github.com/tensorflow/text.git ~/builds/text
# Install tensorflow prior to building.
pip install 'tensorflow==2.17.1'
cd ~/builds/text && git checkout v2.17.0

# Build tensorflow-text.
./oss_scripts/run_build.sh
pip install ./tensorflow_text-2.17.0-cp310-cp310-macosx_*_arm64.whl
```
</details>

<br>

### Installation (User)

This section is intended for users who **do not** intend to develop AXLearn, but rather use it as a package.

To install on Intel (x86) machines, simply run:
```shell
pip install 'axlearn[core]'
```

To install on Apple Silicon machines, make sure you have followed the required [pre-requisites](#pre-requisites) above. Then, install using:
```shell
pip install 'axlearn[core,apple-silicon]'
```

By default, AXLearn comes with tooling to launch jobs to Google Cloud Platform (GCP). To install them, run:
```shell
pip install 'axlearn[gcp]'
```

<br>

### Installation (Developer)

This section is for users who **do** intend to develop AXLearn, e.g. by submitting PRs.

<details>
<summary>Expand for instructions</summary>

Instead of installing from `pip`, please fork the repo first, and then clone the fork.

```shell
# Clone your fork of the repo.
git clone https://github.com/<username>/axlearn
cd axlearn
```

In order to iterate locally and run tests, install the package in editable mode along with `dev` dependencies:
```shell
pip install -e '.[core,dev]'
```

If you intend to launch jobs to GCP, install `gcp` dependencies:
```shell
pip install -e '.[gcp]'
```

We also recommend setting up pre-commit hooks to run some CI checks locally:
```shell
pre-commit install --hook-type pre-commit
```

These checks will run automatically when you `git commit`, but you can also run pre-commit directly (please refer to the [pre-commit](https://pre-commit.com/) docs for more information):
```shell
pre-commit run -a
```

We use [pytype](https://github.com/google/pytype) for static type checking:
```shell
# This can take a while, so we exclude it from pre-commit.
pytype -j auto .
```

To run tests (please refer to the [pytest](https://docs.pytest.org) docs for more information):
```shell
pytest axlearn/common/config_test.py

# To set logging level:
pytest --log-cli-level=INFO axlearn/common/config_test.py

# To test a specific pattern of tests:
pytest axlearn/common/config_test.py -k "test_invalid"

# Run tests with 4 processes and specific markers:
pytest -n 4 -v -m "not (gs_login or tpu)" axlearn/common/
```

</details>

<br>

## A Short Tutorial

> This section walks through writing an AXLearn experiment from scratch. For the impatient, skip ahead to [launching training](#launching-training) to start training a model with an existing recipe.

AXLearn experiments have a standard anatomy. At a high level, we will need to define:
- The model architecture.
- The training and evaluation data.
- The evaluation metrics.
- The optimizer.

AXLearn comes with many reusable building blocks for building an experiment.

Let's walk through training [ResNet](https://arxiv.org/abs/1512.03385) on [ImageNet](https://arxiv.org/abs/1409.0575) as an example. All of the code is available under `axlearn/experiments/vision/resnet`.

If you plan to follow along with the code, create a new file `axlearn/experiments/tutorial.py` with the following skeleton:
```python
# Imports to be added here...

def resnet_imagenet_trainer():
    # Code to be added here...
    ...
```

<br>

### Model Architecture

Since ImageNet is a classification task, we will start with an image classification model. Thankfully, a skeleton already exists in the `vision.image_classification` module:

```diff
+from axlearn.vision import image_classification

def resnet_imagenet_trainer():
+    # Construct an image classifier config.
+    model_cfg = image_classification.ImageClassificationModel.default_config()
```

To use this model, let's look at the definition of `ImageClassificationModel.Config`. You may notice that it has a couple required fields:
https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/vision/image_classification.py#L16-L28

- The first is the `backbone`, which is the underlying model architecture we want to use for computing the image embeddings.
- The second is `num_classes`, which is the number of class labels in our dataset.

Since we are interested in ResNet on ImageNet, we can modify the above code like so:
```diff
from axlearn.vision import image_classification
+from axlearn.vision import resnet

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    model_cfg = image_classification.ImageClassificationModel.default_config()
+   model_cfg.backbone = resnet.ResNet.resnet50_config()
+   model_cfg.num_classes = 1000
```

This will use a vanilla ResNet-50 backbone for classifying the `1000` classes in ImageNet. `ImageClassificationModel` will handle the rest, such as extract embeddings from the backbone, computing the logits, and computing the loss and other metrics.

We can further customize the model by setting additional configs. For example, ResNet models commonly use He initialization:
```diff
+import math
+from axlearn.common import param_init
from axlearn.vision import image_classification
from axlearn.vision import resnet

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    model_cfg = image_classification.ImageClassificationModel.default_config()
    model_cfg.backbone = resnet.ResNet.resnet50_config()
    model_cfg.num_classes = 1000
+   model_cfg.param_init = param_init.DefaultInitializer.default_config().set(
+       init_by_param_name={
+           param_init.PARAM_REGEXP_WEIGHT: param_init.WeightInitializer.default_config().set(
+               fan="fan_out",
+               distribution="normal",
+               scale=math.sqrt(2),
+           )
+       }
+   )
```

If you want, you can also customize the ResNet backbone:
```diff
def resnet_imagenet_trainer():
    # Construct an image classifier config.
    model_cfg = image_classification.ImageClassificationModel.default_config()
-   model_cfg.backbone = resnet.ResNet.resnet50_config()
+   model_cfg.backbone = resnet.ResNet.resnet50_config().set(
+       hidden_dim=123,
+       num_blocks_per_stage=[1, 2, 3],
+       ...
+   )
    ...
```

Or, you can just as easily switch to a different backbone:
```diff
def resnet_imagenet_trainer():
    # Construct an image classifier config.
    model_cfg = image_classification.ImageClassificationModel.default_config()
-   model_cfg.backbone = resnet.ResNet.resnet50_config()
+   model_cfg.backbone = resnet.ResNet.resnet101_config()
    ...
```

You can refer to the [`ResNet.Config`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/vision/resnet.py#L436-L452) definition for more details on the ResNet config API.

For simplicity, we will use the default values of our ResNet-50 backbone.

<br>

### Training and Evaluation Data

Next, we will define an input pipeline for reading ImageNet. As of writing, the most well-supported[^1] option uses [Tensorflow Datasets](https://www.tensorflow.org/datasets) (TFDS), which may already be familiar to you (if not, that's completely fine).

As before, we can leverage existing building blocks in AXLearn. This time we can reuse the `ImagenetInput` under the `vision.input_image` module:
```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
+from axlearn.vision import input_image

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

+   # Construct an input pipeline config.
+   input_cfg = input_image.ImagenetInput.default_config()
```

Taking a peek under the hood, we can see that `ImagenetInput.default_config()` constructs a `tfds_dataset`, and applies some standard processing like cropping and resizing:
https://github.com/apple/axlearn/blob/70eb15ffe7285cb97287b8665b91559e4b23726e/axlearn/vision/input_image.py#L383-L395

Like before, let's inspect the config API inherited from its parent class, the `Input.Config`:
https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/input_tf_data.py#L847-L879

The main required fields are:
- `is_training`: a bool indicating whether the dataset is used for training[^2].
- The `source`: a function[^3] that returns a dataset (in this case, a `tf.data.Dataset`).
- The `processor`: a function that takes a dataset, and outputs another dataset.
- The `batcher`: a function that takes a dataset, and outputs a batched dataset.

The `ImagenetInput.default_config()` fills in these required configs for you, using reasonable defaults in context of ImageNet processing.

Note that each of `source`, `processor`, and `batcher` are themselves configs.
This allows us to configure their properties with minimal changes:

```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
from axlearn.vision import input_image
+from axlearn.common import input_tf_data
+from axlearn.common import config

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

    # Construct an input pipeline config.
    input_cfg = input_image.ImagenetInput.default_config()
+   input_cfg.source.set(
+       split="train",
+       read_config=config.config_for_function(input_tf_data.tfds_read_config).set(
+           decode_parallelism=128,
+       ),
+   )
+   input_cfg.batcher.global_batch_size = 1024
```

Above, we independently set `source.split = "train"` to read the `"train"` data split, and set `batcher.global_batch_size = 1024` to configure the batch size across all training hosts.

For efficiency reasons, we also set the [`decode_parallelism`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/input_tf_data.py#L88-L89) when reading the dataset.

You may be wondering what `config.config_for_function` is. We will cover more details in [concepts](02-concepts.md), but at a high level, it dynamically generates a config from a function signature (in this case `input_tf_data.tfds_read_config`). This allows any arbitrary function to interoperate with the config system.

This also gives insight into the config API for `input_tf_data.tfds_read_config` -- it is simply the arguments of the function itself:
https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/input_tf_data.py#L66-L73

As you can see from the above example, we've configured the `decode_parallelism` parameter to be `128`.

As another example, we can switch to the newer [ImageNetV2](https://www.tensorflow.org/datasets/catalog/imagenet_v2) dataset with the following changes:
```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
from axlearn.vision import input_image
from axlearn.common import input_tf_data
from axlearn.common import config

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

    # Construct an input pipeline config.
    input_cfg = input_image.ImagenetInput.default_config()
    input_cfg.source.set(
        split="train",
        read_config=config.config_for_function(input_tf_data.tfds_read_config).set(
            decode_parallelism=128,
        ),
    )
    input_cfg.batcher.global_batch_size = 1024

+   # Swaps the dataset name to the newer ImageNetV2.
+   input_cfg.source.dataset_name = "imagenet_v2/matched-frequency"
```

This will apply the same input processing as before, except to `imagenet_v2` instead of `imagenet2012`.
Hopefully, this provides some basic intuition about the design of the config system, which is prevalent in the AXLearn codebase.

For now, let's stick to the original `imagenet2012`.

[^1]: Note that it's possible to use other types of data processing pipelines (e.g. [torch dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)). AXLearn is designed to be an open system.
[^2]: This affects things like whether we should shuffle the dataset or not.
[^3]: More accurately, it is a config that instantiates to such a function, but more on that in [concepts](02-concepts.md).

<br>

### Evaluation Metrics

A crucial part of any experiment is evaluating how well the model is learning.

AXLearn provides an evaler implementation called `SpmdEvaler`, which has the following config API:
https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/evaler.py#L468-L491

As we can see, we only need to provide an `input` source to use it. Among other things, it already comes with a `summary_writer` to log evaluation metrics, and a basic `metric_calculator` to compute metrics for the summary writer.

In fact, we already have most of the pieces ready:
- We have an input config which can read `imagenet2012`. We just need to tweak it slightly to read the `"validation"` split, instead of the `"train"` split.
- We have a classification model `ImageClassificationModel`, which already comes with basic [metrics](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/layers.py#L1304-L1310) like `accuracy`, `perplexity` and `cross_entropy_loss`.
- The `summary_writer` is already capable of logging these metrics to [tensorboard](https://www.tensorflow.org/tensorboard).

By now, you may already have an idea of how to construct the evaler input:
```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
from axlearn.vision import input_image
from axlearn.common import input_tf_data
from axlearn.common import config

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

    # Construct an input pipeline config.
    input_cfg = input_image.ImagenetInput.default_config()
    input_cfg.source.set(
        split="train",
        read_config=config.config_for_function(input_tf_data.tfds_read_config).set(
            decode_parallelism=128,
        ),
    )
    input_cfg.batcher.global_batch_size = 1024

+   # Construct an input pipeline config for evaluation.
+   eval_input_cfg = input_image.ImagenetInput.default_config()
+   eval_input_cfg.source.split = "validation"
+   eval_input_cfg.batcher.global_batch_size = 80
```

Because we often do not want to shuffle the eval dataset, we also disable shuffling using the utility `input_tf_data.disable_shuffle_recursively`.

```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
from axlearn.vision import input_image
from axlearn.common import input_tf_data
from axlearn.common import config

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

    # Construct an input pipeline config.
    input_cfg = input_image.ImagenetInput.default_config()
    input_cfg.source.set(
        split="train",
        read_config=config.config_for_function(input_tf_data.tfds_read_config).set(
            decode_parallelism=128,
        ),
    )
    input_cfg.batcher.global_batch_size = 1024

    # Construct an input pipeline config for evaluation.
    eval_input_cfg = input_image.ImagenetInput.default_config()
    eval_input_cfg.source.split = "validation"
    eval_input_cfg.batcher.global_batch_size = 80
+   input_tf_data.disable_shuffle_recursively(eval_input_cfg)
```

We can then construct the `SpmdEvaler` config, which takes the eval input as a child:
```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
from axlearn.vision import input_image
from axlearn.common import input_tf_data
from axlearn.common import config
+from axlearn.common import evaler

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

    # Construct an input pipeline config.
    input_cfg = input_image.ImagenetInput.default_config()
    input_cfg.source.set(
        split="train",
        read_config=config.config_for_function(input_tf_data.tfds_read_config).set(
            decode_parallelism=128,
        ),
    )
    input_cfg.batcher.global_batch_size = 1024

    # Construct an input pipeline config for evaluation.
    eval_input_cfg = input_image.ImagenetInput.default_config()
    eval_input_cfg.source.split = "validation"
    eval_input_cfg.batcher.global_batch_size = 80
    input_tf_data.disable_shuffle_recursively(eval_input_cfg)

+   # Construct the evaler.
+   evaler_cfg = evaler.SpmdEvaler.default_config()
+   evaler_cfg.input = eval_input_cfg
```

A minor note is that the default evaler runs every step, which may be more frequent than we'd like. We can tweak the eval rate by customizing the eval policy:
```diff
from axlearn.vision import image_classification
from axlearn.vision import resnet
from axlearn.vision import input_image
from axlearn.common import input_tf_data
from axlearn.common import config
from axlearn.common import evaler

def resnet_imagenet_trainer():
    # Construct an image classifier config.
    ...

    # Construct an input pipeline config.
    input_cfg = input_image.ImagenetInput.default_config()
    input_cfg.source.set(
        split="train",
        read_config=config.config_for_function(input_tf_data.tfds_read_config).set(
            decode_parallelism=128,
        ),
    )
    input_cfg.batcher.global_batch_size = 1024

    # Construct an input pipeline config for evaluation.
    eval_input_cfg = input_image.ImagenetInput.default_config()
    eval_input_cfg.source.split = "validation"
    eval_input_cfg.batcher.global_batch_size = 80
    input_tf_data.disable_shuffle_recursively(eval_input_cfg)

    # Construct the evaler.
    evaler_cfg = evaler.SpmdEvaler.default_config()
    evaler_cfg.input = eval_input_cfg
+   evaler_cfg.eval_policy.n = 12_510  # Eval roughly every 10 epochs.
```

This will cause the evaler to run every 12.5k steps instead, roughly every 10 epochs.

<br>

### Optimizer

Next, we will need to define an optimizer. AXLearn comes with a variety of default implementations in `common.optimizers`. For this example, we'll use standard Stochastic Gradient Descent (SGD) with a weight decay of `1e-4` and momentum of `0.9`, mostly following the original paper.

```python
from axlearn.common import config, optimizers

def resnet_imagenet_trainer():
    # Other code redacted for simplicity...

    # Construct the optimizer config.
    optimizer_cfg = config.config_for_function(optimizers.sgd_optimizer).set(
        decouple_weight_decay=False,  # Set to False to match Torch behavior.
        momentum=0.9,
        weight_decay=1e-4,
    )
```

As before, we use the `config.config_for_function` utility to dynamically generate a config from the `optimizers.sgd_optimizer` function signature:
https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/optimizers.py#L507-L514

Among these parameters is the `learning_rate`, which we still haven't configured yet.
Out of the box, AXLearn comes with a variety of learning rate schedules in `common.schedule`.
An appropriate learning rate schedule for our use-case (and batch size we intend to use) is a linear warmup to a peak learning rate of `0.4` followed by a cosine decay:

```diff
from axlearn.common import config, optimizers
+from axlearn.common import schedule

def resnet_imagenet_trainer():
    # Other code redacted for simplicity...

    # Construct the optimizer config.
+   learning_rate = config.config_for_function(schedule.cosine_with_linear_warmup).set(
+       peak_lr=0.4,
+       max_step=112_590,  # Roughly 90 epochs.
+       warmup_steps=6_255,  # Roughly 5 epochs.
+   )
    optimizer_cfg = config.config_for_function(optimizers.sgd_optimizer).set(
+       learning_rate=learning_rate,
        decouple_weight_decay=False,  # Set to False to match Torch behavior.
        momentum=0.9,
        weight_decay=1e-4,
    )
```

One caveat of applying weight decay naively is that we regularize all parameters globally. Empirically, we find that regularizing the BatchNorm[^4] parameters hurts model performance, so we exclude them from weight decay:

```diff
from axlearn.common import config, optimizers
from axlearn.common import schedule

def resnet_imagenet_trainer():
    # Other code redacted for simplicity...

    # Construct the optimizer config.
    learning_rate = config.config_for_function(schedule.cosine_with_linear_warmup).set(
        peak_lr=0.4,
        max_step=112_590,  # Roughly 90 epochs.
        warmup_steps=6_255,  # Roughly 5 epochs.
    )
+    per_param_decay = config.config_for_function(optimizers.per_param_scale_by_path).set(
+        description="weight_decay_scale",
+        scale_by_path=[
+            (".*norm.*", 0),  # Exclude the norm parameters from weight decay.
+        ],
+    )
    optimizer_cfg = config.config_for_function(optimizers.sgd_optimizer).set(
        learning_rate=learning_rate,
        decouple_weight_decay=False,  # Set to False to match Torch behavior.
        momentum=0.9,
        weight_decay=1e-4,
+       weight_decay_per_param_scale=per_param_decay,
    )
```

[^4]: [BatchNorm](https://arxiv.org/abs/1502.03167) is used throughout the ResNet architecture by default. We did not need to configure it explicitly.

<br>

### Putting Everything Together

We are now ready to put all the pieces together! The glue that makes everything work is the [`SpmdTrainer`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/trainer.py#L62). As the name implies, it runs the main training loop using the components that we've defined above.

Once again, we can get an idea of how to use this component by inspecting its config API:
https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/trainer.py#L62-L128

This particular module is quite complex, but feel free to refer to the inline documentation and comments for more details.
We can also construct it in a familiar fashion:

```python
from axlearn.common import trainer, learner

def resnet_imagenet_trainer():
    # Other code redacted for simplicity...

    # Construct the trainer config.
    trainer_cfg = trainer.SpmdTrainer.default_config()
    trainer_cfg.model = model_cfg
    trainer_cfg.input = input_cfg
    trainer_cfg.evalers = {"eval_validation": evaler_cfg}
    trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
    trainer_cfg.max_step = 112_590  # Roughly 90 epochs.
```

The code block plugs in the model, input, evalers, and other components that we have already defined.

Note that we have wrapped the optimizer with a [`Learner.Config`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/common/learner.py#L121-L149). The Learner internally uses the optimizer to update model params, and acts as an intermediary to the trainer.

For some basic book-keeping, we also configure the frequency of checkpoints and summaries:
```diff
from axlearn.common import trainer, learner

def resnet_imagenet_trainer():
    # Other code redacted for simplicity...

    # Construct the trainer config.
    trainer_cfg = trainer.SpmdTrainer.default_config()
    trainer_cfg.model = model_cfg
    trainer_cfg.input = input_cfg
    trainer_cfg.evalers = {"eval_validation": evaler_cfg}
    trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
    trainer_cfg.max_step = 112_590  # Roughly 90 epochs.

+   # Define checkpoint frequency and summary frequency.
+   trainer_cfg.checkpointer.save_policy.n = 12_510
+   trainer_cfg.checkpointer.keep_every_n_steps = 12_510
+   trainer_cfg.summary_writer.write_every_n_steps = 100
```

The last step is to expose the `trainer_cfg` in a format that is understood by the AXLearn CLI.
We'll cover this in more detail in [a later section](#launching-an-experiment), but in short, the AXLearn CLI looks for a "registry" of trainers via `named_trainer_configs`:

```diff
from axlearn.common import trainer, learner

def resnet_imagenet_trainer():
    # Other code redacted for simplicity...

    # Construct the trainer config.
    trainer_cfg = trainer.SpmdTrainer.default_config()
    trainer_cfg.model = model_cfg
    trainer_cfg.input = input_cfg
    trainer_cfg.evalers = {"eval_validation": evaler_cfg}
    trainer_cfg.learner = learner.Learner.default_config().set(optimizer=optimizer_cfg)
    trainer_cfg.max_step = 112_590  # Roughly 90 epochs.

    # Define checkpoint frequency and summary frequency.
    trainer_cfg.checkpointer.save_policy.n = 12_510
    trainer_cfg.checkpointer.keep_every_n_steps = 12_510
    trainer_cfg.summary_writer.write_every_n_steps = 100

+   # Return the final trainer config.
+   return trainer_cfg.set(name="resnet_imagenet")


+# Expose the trainer configs to the AXLearn CLI.
+def named_trainer_configs():
+   # Return a mapping from name(s) to trainer config function(s).
+   return {"ResNet-50": resnet_imagenet_trainer}
```

<br>

### Testing

While building a trainer config with Python code allows us to reuse configuration logic, a downside is that the indirections make it hard to see the effects, especially when we want to update the logic.
To address this issue, the [`golden_config_test`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/experiments/golden_config_test.py) generates the full configuration of each registered trainer config and puts them under `axlearn/experiments/testdata`.
For example, you can see the full trainer config of the ResNet50 experiment on ImageNet [here](../axlearn/experiments/testdata/axlearn.experiments.vision.resnet.imagenet_trainer/ResNet-50.txt).
This is especially useful for catching unintended changes to experiment configurations during refactoring.

To generate the golden configs for your own trainer(s), update the `golden_config_test` and run:
```bash
pytest -n auto axlearn/experiments/golden_config_test.py --update
```
For more details on golden configs, please see [concepts](02-concepts.md).

Before launching experiments into the cloud, it's also recommended to write unit tests to catch failures early. For some examples, we refer the reader to unit tests under [`axlearn/experiments/vision/resnet`](https://github.com/apple/axlearn/tree/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/experiments/vision/resnet).

<br>

### Summary

Congratulations on getting this far! Hopefully, you now have a taste of how to build experiments with AXLearn. Granted, this was a fairly quick overview of what AXLearn has to offer, and some of the content may still feel abstract or foreign. For more details on the config system or other concepts, please refer to the [concepts page](02-concepts.md).

The following section will cover how to launch your experiment in the cloud.

<br>

## Launching Training

AXLearn comes with tooling for provisioning and launching training on public clouds. This section will guide you with launching training on a [Google Cloud TPU](https://cloud.google.com/tpu/docs/intro-to-tpu).

### Pre-requisites

We assume you have:
1. `gcloud` setup, following e.g. https://cloud.google.com/sdk/docs/install.
2. A Google Cloud Platform (GCP) project. To set up a brand new GCP project with the basic resources needed, please run [this script](../axlearn/cloud/gcp/scripts/project_setup.sh).
3. TPU quota in your project. To request TPU quota, please follow [these instructions](https://cloud.google.com/tpu/docs/setup-gcp-account#prepare-to-request).
4. At least one Google Cloud Storage (GCS) bucket.

<br>

### Preparing the CLI

Please follow the instructions in the [CLI docs](03-cli.md#preparing-the-cli) to setup the CLI.

We assume that you are launching from a working directory that contains a `pyproject.toml` or `setup.py` (for instance, if you cloned the repo, you should have one already). If not, you can create a minimal `pyproject.toml`:
```toml
[project]
name = "my_project"
version = "0.0.1"
dependencies = ["axlearn"]

[project.optional-dependencies]
tpu = ["axlearn[tpu]"]
```

<br>

### Launching a Command

We can now leverage the AXLearn infrastructure to launch commands on arbitrary TPU configurations.

First, make sure you have authenticated to GCP:
```bash
# Authenticate to GCP.
axlearn gcp auth
```

We can then test a simple `v4-8` command:
```bash
# Run a dummy command on v4-8.
# Note: the "'...'" quotes are important.
axlearn gcp tpu start --name=$USER-test --tpu_type=v4-8 -- python3 -c "'import jax; print(jax.devices())'"
```

This provisions a v4-8 TPU, installs `axlearn` on it, and runs the `python3` command that comes after `--`. As the job is running, any logs from the command will be synced to GCS. Once the job is completed, the TPU resources will be torn down.

<br>

### Launching an Experiment

To launch an actual experiment, we must first define an experiment module that AXLearn understands how to parse.

There are two aspects to this:
1. By default, AXLearn looks under the `axlearn/experiments` directory for experiment modules, so we should define it there.
2. Experiment modules must expose a function `named_trainer_configs` which returns a dictionary with experiment names as keys, and [`TrainerConfigFn`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/experiments/trainer_config_utils.py#L11)s as values. As the name implies, a `TrainerConfigFn` is a function that simply returns a trainer config, similar to the one constructed [above](#putting-everything-together)[^5].

We've already packaged the ResNet on ImageNet example for you, which can be launched via:
```bash
OUTPUT_DIR=gs://path/to/$USER/experiments/resnet50-$(date +%F)
DATA_DIR=gs://path/to/tensorflow_datasets

axlearn gcp tpu start --tpu_type=v4-8 --output_dir=$OUTPUT_DIR -- \
    python3 -m axlearn.common.launch_trainer_main \
    --module=vision.resnet.imagenet_trainer --config=ResNet-50 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --jax_backend=tpu
```

If you have been following along with the code, assuming you have a file `axlearn/experiments/tutorial.py`, you can also launch your own experiment with:
```diff
OUTPUT_DIR=gs://path/to/$USER/experiments/resnet50-$(date +%F)
DATA_DIR=gs://path/to/tensorflow_datasets

axlearn gcp tpu start --tpu_type=v4-8 --output_dir=$OUTPUT_DIR -- \
    python3 -m axlearn.common.launch_trainer_main \
-   --module=vision.resnet.imagenet_trainer --config=ResNet-50 \
+   --module=tutorial --config=ResNet-50 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR --jax_backend=tpu
```

Both commands are similar to the one from the previous section except we run the trainer defined by `--module` and `--config` instead of simply printing `jax.devices()`.

The `OUTPUT_DIR` defines where to emit training artifacts like checkpoints and summaries, and the `DATA_DIR` defines where to look for datasets.

To view `tensorboard`, point to the `OUTPUT_DIR`:
```
tensorboard --logdir=$OUTPUT_DIR
```
Or, if VertexAI is [configured](#preparing-the-cli), you should also see a VertexAI Tensorboard link.

[^5]: One common question is: why return a `TrainerConfigFn` instead of, say, the trainer config itself? The reason is that a `TrainerConfigFn` allows us to defer the construction of a trainer config until we need to use it. When your project has many experiments, the cost of building all trainer configs can be non-trivial (such as when running [golden config tests](#testing)).

<br>

### Launching via Bastion

In an organization setting, it's typical to launch jobs from a centralized system, which can:
1. Constantly monitor and restart jobs as necessary.
2. Queue and schedule jobs based on priority.

AXLearn provides such an orchestrator called the "bastion", which can run in GCP with minimal dependencies.

It is often recommended to launch from the bastion. Please see the [infrastructure docs](04-infrastructure.md) for instructions on how to set it up.

<br>

## Next Steps

As a next step, we encourage you to read some of the [AXLearn Concepts](02-concepts.md) if you have not already.
While this document covers much of "how" to run experiments, the next section aims to explain the "why" behind the AXLearn design.
