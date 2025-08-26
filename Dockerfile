# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS base

# Install curl and gpupg first so that we can use them to install google-cloud-cli.
# Any RUN apt-get install step needs to have apt-get update otherwise stale package
# list may occur when previous apt-get update step is cached. See here for more info:
# https://docs.docker.com/build/building/best-practices/#apt-get
RUN apt-get update && apt-get upgrade -y && apt-get install -y curl gnupg && apt clean -y

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install -y apt-transport-https ca-certificates gcc g++ \
      git screen google-perftools google-cloud-cli python3.12-venv && \
    apt clean -y

# Setup.
RUN mkdir -p /root
WORKDIR /root
# Introduce the minimum set of files for install.
COPY README.md README.md
COPY pyproject.toml pyproject.toml
RUN mkdir axlearn && touch axlearn/__init__.py
# Setup venv to suppress pip warnings.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies.
RUN pip install --upgrade pip && pip install uv flit && pip cache purge

################################################################################
# CI container spec.                                                           #
################################################################################

# Leverage multi-stage build for unit tests.
FROM base AS ci

# TODO(markblee): Remove gcp,vertexai_tensorboard from CI.
RUN uv pip install .[core,audio,orbax,dev,gcp,vertexai_tensorboard,open_api] && uv cache clean
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
RUN uv pip install .[core,gcp,vertexai_tensorboard] && uv cache clean

################################################################################
# Dataflow container spec.                                                     #
################################################################################

FROM base AS dataflow

# Beam workers default to creating a new virtual environment on startup. Instead, we want them to
# pickup the venv setup above. An alternative is to install into the global environment.
ENV RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1
RUN uv pip install .[core,gcp,dataflow] && uv cache clean
COPY . .

# Dataflow workers can't start properly if the entrypoint is not set
# See: https://cloud.google.com/dataflow/docs/guides/build-container-image#use_a_custom_base_image
COPY --from=apache/beam_python3.12_sdk:2.52.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT ["/opt/apache/beam/boot"]

################################################################################
# TPU container spec.                                                          #
################################################################################

FROM base AS tpu

ARG EXTRAS=

ENV UV_FIND_LINKS="https://storage.googleapis.com/jax-releases/libtpu_releases.html,https://storage.googleapis.com/axlearn-wheels/wheels.html"
# Ensure we install the TPU version, even if building locally.
# Jax will fallback to CPU when run on a machine without TPU.
RUN uv pip install --prerelease=allow .[core,tpu] && uv cache clean
RUN uv pip install -U --pre jax jaxlib libtpu requests -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
    && uv cache clean
RUN if [ -n "$EXTRAS" ]; then uv pip install .[$EXTRAS] && uv cache clean; fi
COPY . .

################################################################################
# GPU container spec.                                                          #
################################################################################

FROM base AS gpu

# TODO(markblee): Support extras.
ENV UV_FIND_LINKS="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html,https://storage.googleapis.com/axlearn-wheels/wheels.html"
# Enable the CUDA repository and install the required libraries (libnvrtc.so)
RUN curl -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y cuda-libraries-dev-12-8 ibverbs-utils && \
    apt clean -y
RUN uv pip install .[core,gpu] && uv cache clean
COPY . .

################################################################################
# Final target spec.                                                           #
################################################################################

FROM ${TARGET} AS final
