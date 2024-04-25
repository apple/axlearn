# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=python:3.9-slim

FROM ${BASE_IMAGE} AS base

RUN apt-get update
RUN apt-get install -y apt-transport-https ca-certificates gnupg curl gcc g++

# Install git.
RUN apt-get install -y git

# Install gcloud. https://cloud.google.com/sdk/docs/install
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && apt-get install google-cloud-cli -y

# Install screen and other utils for launch script.
RUN apt-get install -y jq screen ca-certificates

# Setup.
RUN mkdir -p /root
WORKDIR /root
# Introduce the minimum set of files for install.
COPY README.md README.md
COPY pyproject.toml pyproject.toml
RUN mkdir axlearn && touch axlearn/__init__.py
# Setup venv to suppress pip warnings.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies.
RUN pip install flit
RUN pip install --upgrade pip

################################################################################
# CI container spec.                                                           #
################################################################################

# Leverage multi-stage build for unit tests.
FROM base as ci

# TODO(markblee): Remove gcp,vertexai_tensorboard from CI.
RUN pip install .[dev,gcp,vertexai_tensorboard]
COPY . .

# Defaults to an empty string, i.e. run pytest against all files.
ARG PYTEST_FILES=''
# Defaults to empty string, i.e. do NOT skip precommit
ARG SKIP_PRECOMMIT=''
# `exit 1` fails the build.
RUN ./run_tests.sh $SKIP_PRECOMMIT "${PYTEST_FILES}"

################################################################################
# Bastion container spec.                                                      #
################################################################################

FROM base AS bastion

# TODO(markblee): Consider copying large directories separately, to cache more aggressively.
# TODO(markblee): Is there a way to skip the "production" deps?
COPY . /root/
RUN pip install .[gcp,vertexai_tensorboard]

################################################################################
# Dataflow container spec.                                                     #
################################################################################

FROM base AS dataflow

# Beam workers default to creating a new virtual environment on startup. Instead, we want them to
# pickup the venv setup above. An alternative is to install into the global environment.
ENV RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1
RUN pip install .[gcp,dataflow]
COPY . .

COPY --from=apache/beam_python3.9_sdk:2.52.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT ["/opt/apache/beam/boot"]


################################################################################
# TPU container spec.                                                          #
################################################################################

FROM base AS tpu

# TODO(markblee): Support extras.
ENV PIP_FIND_LINKS=https://storage.googleapis.com/jax-releases/libtpu_releases.html
# Ensure we install the TPU version, even if building locally.
# Jax will fallback to CPU when run on a machine without TPU.
RUN pip install .[tpu]
COPY . .

################################################################################
# Final target spec.                                                           #
################################################################################

FROM ${TARGET} AS final
