# Getting Started

## Installing Dependencies

First, clone the repo. If you intend to develop AXLearn, please fork the repo first, and then clone the fork.

```shell
git clone https://github.com/apple/axlearn
cd axlearn
```

(Optional) We recommend installing within a virtual environment, e.g. with [conda](https://conda.io):
```
conda create -n axlearn python=3.9
conda activate axlearn
```

The remaining steps depend on your machine hardware:

### Intel (x86) Machines

Simply do an editable install with:

```shell
# Note: This also installs dependencies required for launching jobs to GCP.
pip install -e .[gcp]
```

Feel free to continue to the [next section](#optional-additional-setup-for-developers).

### Apple Silicon Machines

<details>
<summary>Expand for instructions</summary>

For Apple Silicon machines, we will install native versions of Python and Python packages using Miniforge.

We need Xcode to build packages like `tensorstore`. Please install Xcode from the App Store if you haven't already.

```shell
# Install the arm64 version of Miniforge3 + Python 3.9.
curl -L -O https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh

# Create a conda environment.
conda create -n axlearn python=3.9
conda activate axlearn

# Install tensorflow following https://developer.apple.com/metal/tensorflow-plugin.
conda install -c apple tensorflow-deps==2.8.0
pip install tensorflow-macos==2.8.0
pip install tensorflow-metal==0.4.0
pip install protobuf==3.20.3

# Install tensorflow-io (https://github.com/tensorflow/io/issues/1298).
mkdir ~/builds && git clone https://github.com/tensorflow/io.git ~/builds/io
cd ~/builds/io && git checkout v0.25.0
python setup.py -q bdist_wheel --project tensorflow_io_gcs_filesystem
pip install dist/tensorflow_io_gcs_filesystem-0.25.0-cp39-cp39-macosx_11_0_arm64.whl

# Install tensorflow_text (https://github.com/tensorflow/text/pull/756).
pip install https://github.com/sun1638650145/Libraries-and-Extensions-for-TensorFlow-for-Apple-Silicon/releases/download/v2.8/tensorflow_text-2.8.2-cp39-cp39-macosx_11_0_arm64.whl

# Finally, install AXLearn.
pip install -e .[apple-silicon,gcp]
```

</details>

### (Optional) Additional Setup for Developers

This section is for users who intend to develop AXLearn. If you do not plan to submit PRs, feel free to skip to the [next section](#starting-an-experiment).

<details>
<summary>Expand for instructions</summary>

In order to run tests locally, consider installing the `dev` dependencies:
```shell
pip install -e .[dev]
```

We also recommend setting up pre-commit hooks to run some CI checks locally:
```shell
pre-commit install --hook-type pre-commit
```

These checks will run automatically when you `git commit`, but you can also run pre-commit directly (please refer to the [pre-commit](https://pre-commit.com/) docs for more information):
```shell
pre-commit run -a

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

### Training and Evaluation Data

Next, we will define an input pipeline for reading ImageNet. As of writing, the most well-supported[1] option uses [Tensorflow Datasets](https://www.tensorflow.org/datasets) (TFDS), which may already be familiar to you (if not, that's completely fine).

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
- `is_training`: a bool indicating whether the dataset is used for training[2].
- The `source`: a function[3] that returns a dataset (in this case, a `tf.data.Dataset`).
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

[1] Note that it's possible to use other types of data processing pipelines (e.g. [torch dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)). AXLearn is designed to be an open system.
[2] This affects things like whether we should shuffle the dataset or not.
[3] More accurately, it is a config that instantiates to such a function, but more on that in [concepts](02-concepts.md).

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

One caveat of applying weight decay naively is that we regularize all parameters globally. Empirically, we find that regularizing the BatchNorm[1] parameters hurts model performance, so we exclude them from weight decay:

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

[1] [BatchNorm](https://arxiv.org/abs/1502.03167) is used throughout the ResNet architecture by default. We did not need to configure it explicitly.

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

### Testing

While building a trainer config with Python code allows us to reuse configuration logic, a downside is that the indirections make it hard to see the effects, especially when we want to update the logic.
To address this issue, the [`golden_config_test`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/experiments/golden_config_test.py) generates the full configuration of each registered trainer config and puts them under `axlearn/experiments/testdata`.
For example, you can see the full trainer config of the ResNet50 experiment on ImageNet [here](axlearn/experiments/testdata/axlearn.experiments.vision.resnet.imagenet_trainer/ResNet-50.txt).
This is especially useful for catching unintended changes to experiment configurations during refactoring.

To generate the golden configs for your own trainer(s), update the `golden_config_test` and run:
```bash
pytest -n auto axlearn/experiments/golden_config_test.py --update
```
For more details on golden configs, please see [concepts](02-concepts.md).

Before launching experiments into the cloud, it's also recommended to write unit tests to catch failures early. For some examples, we refer the reader to unit tests under [`axlearn/experiments/vision/resnet`](https://github.com/apple/axlearn/tree/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/experiments/vision/resnet).

### Summary

Congratulations on getting this far! Hopefully you now have a taste of how to build experiments with AXLearn. Granted, this was a fairly quick overview of what AXLearn has to offer, and some of content may still feel abstract or foreign. For more details on the config system or other concepts, please refer to the [concepts page](02-concepts.md).

The following section will cover how to launch your experiment in the cloud.

## Launching Training

AXLearn comes with tooling for provisioning and launching training on public clouds. This section will guide you with launching training on a [Google Cloud TPU](https://cloud.google.com/tpu/docs/intro-to-tpu).

### Pre-requisites

We assume you have:
1. A Google Cloud Platform (GCP) project with TPU quota.
2. At least one Google Cloud Storage (GCS) bucket.
3. `gcloud` setup, following e.g. https://cloud.google.com/sdk/docs/install.

### Preparing the CLI

To setup the CLI, we'll need to first create a config file under **one of** the following paths:
- `.axlearn.config` in the current working directory, or
- `~/.axlearn.config` in your home directory.

> Tip: You can always run `axlearn gcp config cleanup` to delete all AXLearn config files from your system.

Here's a sample config file for launching `v4-tpu`s in `us-central2-b`, under the project `my-gcp-project`. You may recognize it as a [`toml`](https://toml.io/en/) file:
```toml
[gcp."my-gcp-project:us-central2-b"]
# Basic project configs.
project = "my-gcp-project"
zone = "us-central2-b"
network = "projects/my-gcp-project/global/networks/default"
subnetwork = "projects/my-gcp-project/regions/us-central2/subnetworks/default"
# Used when launching VMs and TPUs.
service_account_email = "ml-training@my-gcp-project.iam.gserviceaccount.com"
# Used for permanent artifacts like checkpoints. Should be writable by users who intend to launch jobs.
permanent_bucket = "public-permanent-us-central2"
# Used for private artifacts, like quota files. Should be readable by users who intend to launch jobs.
private_bucket = "private-permanent-us-central2"
# Used for temporary artifacts, like logs. Should be writable by users who intend to launch jobs.
ttl_bucket = "ttl-30d-us-central2"
# (Optional) Used by the AXLearn CLI. See the CLI docs for more info.
labels = "v4-tpu"
# (Optional) Used for pushing docker images.
docker_repo = "us-docker.pkg.dev/my-gcp-project/axlearn"
# (Optional) Configure whether to use on-demand or reserved TPUs.
reserved_tpu = true
# (Optional) Configure a default Dockerfile to use when launching jobs with docker.
default_dockerfile = "Dockerfile"
# (Optional) Enable VertexAI Tensorboard support during training.
vertexai_tensorboard = "1231231231231231231"
vertexai_region = "us-central1"
```

To confirm that the CLI can find your config file, run:
```bash
axlearn gcp config list
```
If you see your project there, run:
```bash
axlearn gcp config activate
```
To set the project as active. (For more details on what this means, please refer to the [CLI docs](03-cli.md).)

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

This provisions a v4-8 TPU, installs `axlearn` on it, and runs the `python3` command that comes after `--`. As the job is running, any logs from the command will be sync'ed to GCS. Once the job completes, the TPU resources will be torn down.

### Launching an Experiment

To launch an actual experiment, we must first define an experiment module that AXLearn understands how to parse.

There are two aspects to this:
1. By default, AXLearn looks under the `axlearn/experiments` directory for experiment modules, so we should define it there.
2. Experiment modules must expose a function `named_trainer_configs` which returns a dictionary with experiment names as keys, and [`TrainerConfigFn`](https://github.com/apple/axlearn/blob/e7ef158e66e96928bda0ea0544dce172c5494d14/axlearn/experiments/trainer_config_utils.py#L11)s as values. As the name implies, a `TrainerConfigFn` is a function that simply returns a trainer config, similar to the one constructed [above](#putting-everything-together)[1].

We've already packaged the ResNet on ImageNet example for you, which can be launched via:
```bash
OUTPUT_DIR=gs://path/to/$USER/experiments/resnet50-$(date +%F)
DATA_DIR=gs://path/to/tensorflow_datasets

axlearn gcp tpu start --tpu_type=v4-8 --output_dir=$OUTPUT_DIR -- \
    python3 -m axlearn.common.launch_trainer_main \
    --module=vision.resnet.imagenet_trainer --config=ResNet-50 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR
```

If you have been following along with the code, assuming you have a file `axlearn/experiments/tutorial.py`, you can also launch your own experiment with:
```diff
OUTPUT_DIR=gs://path/to/$USER/experiments/resnet50-$(date +%F)
DATA_DIR=gs://path/to/tensorflow_datasets

axlearn gcp tpu start --tpu_type=v4-8 --output_dir=$OUTPUT_DIR -- \
    python3 -m axlearn.common.launch_trainer_main \
-   --module=vision.resnet.imagenet_trainer --config=ResNet-50 \
+   --module=tutorial --config=ResNet-50 \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR
```

Both commands are similar to the one from the previous section except we run the trainer defined by `--module` and `--config` instead of simply printing `jax.devices()`.

The `OUTPUT_DIR` defines where to emit training artifacts like checkpoints and summaries, and the `DATA_DIR` defines where to look for datasets.

To view `tensorboard`, point to the `OUTPUT_DIR`:
```
tensorboard --logdir=$OUTPUT_DIR
```
Or, if VertexAI is [configured](#preparing-the-cli), you should also see a VertexAI Tensorboard link.

[1] One common question is: why return a `TrainerConfigFn` instead of, say, the trainer config itself? The reason is that a `TrainerConfigFn` allows us to defer the construction of a trainer config until we need to use it. When your project has many experiments, the cost of building all trainer configs can be non-trivial (such as when running [golden config tests](#testing)).

### Launching via Bastion

In an organization setting, it's typical to launch jobs from a centralized system, which can:
1. Constantly monitor and restart jobs as necessary.
2. Queue and schedule jobs based on priority.

AXLearn provides such an orchestrator called the "bastion", which can run in GCP with minimal dependencies.

It is often recommended to launch from bastion. Please see the [infrastructure docs](04-infrastructure.md) for instructions on how to set it up.

## Next Steps

As a next step, we encourage you to read some of the [AXLearn Concepts](02-concepts.md) if you have not already.
While this document covers much of "how" to run experiments, the next section aims to explain the "why" behind the AXLearn design.
