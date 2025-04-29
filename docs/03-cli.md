# AXLearn CLI

## Table of Contents

| Section | Description |
| - | - |
| [Overview](#overview) | Overview of the CLI. |
| [Usage](#usage) | Usage Instructions. |

<br>

### Overview

The AXLearn CLI provides utilities for launching and managing resources across different cloud environments.

Taking inspiration from Python virtual environments, the CLI provides functionality to "activate" different cloud environments, allowing users to manage resources in these environments seamlessly.

As a motivating example, suppose that users have access to two different TPU clusters `tpu-cluster1` and `tpu-cluster2`, possibly across different availability zones.

A common usage pattern may look like:
```shell
# Authenticate to the cloud provider.
axlearn gcp auth

# "Activate" `tpu-cluster1` configuration.
axlearn gcp config activate --label=tpu-cluster1

# Launch jobs to `tpu-cluster1`.
axlearn gcp launch ...

# "Activate" `tpu-cluster2` configuration.
axlearn gcp config activate --label=tpu-cluster2

# Launch jobs to `tpu-cluster2` with the same launch command.
axlearn gcp launch ...
```

Users therefore do not need to worry about the underlying configuration details of each cluster.

While the above example uses the `gcp` commands, the CLI is agnostic to the underlying cloud provider and can support arbitrary configurations.

<br>

### Preparing the CLI

To setup the CLI, we'll need to first create a config file under at least one of the following paths:
- `.axlearn/axlearn.default.config` in the current working directory, or
- `.axlearn/.axlearn.config` in the current working directory, or
- `~/.axlearn.config` in your home directory.

To create the config, you can copy from [the template config](.axlearn/axlearn.default.config) from the root of the repo:
```shell
cp .axlearn/axlearn.default.config .axlearn/.axlearn.config
```

> Tip: In an organization setting, you can directly modify `.axlearn/axlearn.default.config`, which can be stored in the root of your git repository. This way, developers will automatically pick up the latest CLI configurations on each git pull.

> Tip: You can always run `axlearn gcp config cleanup` to delete all AXLearn config files from your system.

Here's a sample config file for launching `v4-tpu`s in `us-central2-b`, under the project `my-gcp-project`.
You may recognize it as a [`toml`](https://toml.io/en/) file:
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
# (Optional) Used by the AXLearn CLI.
labels = "v4-tpu"
# (Optional) Used for pushing docker images.
docker_repo = "us-docker.pkg.dev/my-gcp-project/axlearn"
# (Optional) Configure a default Dockerfile to use when launching jobs with docker.
default_dockerfile = "Dockerfile"
# (Optional) Enable VertexAI Tensorboard support during training.
vertexai_tensorboard = "1231231231231231231"
vertexai_region = "us-central1"
```

To confirm that the CLI can locate your config file, run:
```bash
# Lists all environments that the CLI is aware of.
$ axlearn gcp config list
No GCP project has been activated; please run `axlearn gcp config activate`.
Found default config at /path/to/axlearn/.axlearn/axlearn.default.config
Found user config at /path/to/axlearn/.axlearn/.axlearn.config
[ ] my-gcp-project:us-central2-b [v4-tpu]
```

As the output indicates, we have not yet activated a project.
To do so, run:
```bash
# Activate a specific environment.
$ axlearn gcp config activate
...
Setting my-gcp-project:us-central2-b to active.
Configs written to /path/to/axlearn/.axlearn/.axlearn.config
```

You can also directly target a config by specifying `--label`:
```bash
# Activate the environment with label "v4-tpu".
axlearn gcp config activate --label=v4-tpu
```
In this case we only have one config, so `--label` is redundant.

<br>

### Usage

The [CLI](https://github.com/apple/axlearn/blob/204f3de2650c098410749d078195e4f8db96d6f6/axlearn/cli/__init__.py) is structured as a tree and is intended to be self-documenting.
The tree can be traversed simply by invoking the CLI with `--help`.

For instance, running the root command with `--help` prints available sub-commands:
```shell
$ axlearn --help
usage: axlearn [-h] [--helpfull] {gcp} ...

AXLearn: An Extensible Deep Learning Library.

positional arguments:
  {gcp}

options:
  -h, --help  show this help message and exit
  --helpfull  show full help message and exit
```

We can traverse the tree by running a subcommand with `--help`.
```
$ axlearn gcp --help
usage: axlearn gcp [-h] [--helpfull] [--project PROJECT] [--zone ZONE]
                   {config,sshvm,sshtpu,bundle,launch,tpu,vm,bastion,dataflow,auth} ...

positional arguments:
  {config,sshvm,sshtpu,bundle,launch,tpu,vm,bastion,dataflow,auth}
    config              Configure GCP settings
    sshvm               SSH into a VM
    sshtpu              SSH into a TPU-VM
    bundle              Bundle the local directory
    launch              Launch arbitrary commands on remote compute
    tpu                 Create a TPU-VM and execute the given command on it
    vm                  Create a VM and execute the given command on it
    bastion             Launch jobs through Bastion orchestrator
    dataflow            Run Dataflow jobs locally or on GCP
    auth                Authenticate to GCP

options:
  -h, --help            show this help message and exit
  --helpfull            show full help message and exit
  --project PROJECT
  --zone ZONE
```

The leaves of the command tree may be implementation dependent. Typically, they correspond to an [abseil-py](https://github.com/abseil/abseil-py) module.[^1]
For example, `axlearn gcp launch` simply maps to:
https://github.com/apple/axlearn/blob/204f3de2650c098410749d078195e4f8db96d6f6/axlearn/cli/gcp.py#L56-L60

In some cases, they can map to shell commands. For example, `axlearn gcp auth` simply maps to:
https://github.com/apple/axlearn/blob/204f3de2650c098410749d078195e4f8db96d6f6/axlearn/cli/gcp.py#L86-L96

In general, when in doubt, run `--help`.

<br>

[^1]: You can learn more about how abseil flags work from their [Python Devguide](https://abseil.io/docs/python/guides/flags#flag-types).
