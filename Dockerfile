# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=ubuntu:24.04
ARG BASE_IMAGE_COLOCATED=us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:2026_02_09-python_3.12-jax_0.8.3

FROM ${BASE_IMAGE} AS base

# Install curl and gpupg first so that we can use them to install google-cloud-cli.
# Any RUN apt-get install step needs to have apt-get update otherwise stale package
# list may occur when previous apt-get update step is cached. See here for more info:
# https://docs.docker.com/build/building/best-practices/#apt-get
RUN apt-get update -qq && \
    apt-get upgrade -y -qq && \
    apt-get install -y -qq curl gnupg && \
    apt clean -y -qq

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y -qq && \
    apt-get install -y -qq apt-transport-https ca-certificates gcc g++ \
    git screen ca-certificates google-perftools google-cloud-cli python3.12-venv && \
    apt clean -y -qq

# Setup.
RUN mkdir -p /root
WORKDIR /root
# Introduce the minimum set of files for install.
RUN mkdir axlearn && touch axlearn/__init__.py
# Setup venv to suppress pip warnings.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install dependencies.
RUN pip install -qq --upgrade pip && \
    pip install -qq uv flit && \
    pip cache purge

################################################################################
# Bastion container spec.                                                      #
################################################################################

FROM base AS bastion

# TODO(markblee): Consider copying large directories separately, to cache more aggressively.
# TODO(markblee): Is there a way to skip the "production" deps?
COPY . /root/
RUN uv pip install -qq .[core,gcp,vertexai_tensorboard] && uv cache clean

################################################################################
# Dataflow container spec.                                                     #
################################################################################

FROM base AS dataflow

# Beam workers default to creating a new virtual environment on startup. Instead, we want them to
# pickup the venv setup above. An alternative is to install into the global environment.
ENV RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1
COPY pyproject.toml README.md /root/
RUN uv pip install -qq .[core,gcp,dataflow] && uv cache clean
COPY . .

# Dataflow workers can't start properly if the entrypoint is not set
# See: https://cloud.google.com/dataflow/docs/guides/build-container-image#use_a_custom_base_image
COPY --from=apache/beam_python3.12_sdk:2.69.0 /opt/apache/beam /opt/apache/beam
ENTRYPOINT ["/opt/apache/beam/boot"]

################################################################################
# TPU container spec.                                                          #
################################################################################

FROM base AS tpu

ARG EXTRAS=

# Ensure we install the TPU version, even if building locally.
# Jax will fallback to CPU when run on a machine without TPU.
COPY pyproject.toml README.md /root/
RUN uv pip install -qq --prerelease=allow .[core,tpu] && uv cache clean
RUN if [ -n "$EXTRAS" ]; then uv pip install -qq .[$EXTRAS] && uv cache clean; fi

# Generate an 8GB zero dummy file to simulate a 10GB inflection-point image
RUN dd if=/dev/zero of=/root/dummy_8gb_file bs=1M count=8000

COPY . .

################################################################################
# Colocated Python container spec.                                             #
################################################################################

FROM ${BASE_IMAGE_COLOCATED} AS colocated-python

WORKDIR /app
COPY . .

# Install the additional user-provided dependencies, strictly enforcing the rules
# from the base image's constraints file.
RUN \
    # 1. Install user-provided dependencies with modified constraints
    grep -v "^numpy" /opt/venv/server_constraints.txt | grep -v "^scipy" > /tmp/modified_constraints.txt && \
    echo "--> Installing user-provided dependencies..." && \
    uv pip install ".[core,gcp]" -c /tmp/modified_constraints.txt && \
    \
    # 2. Override numpy and scipy with specific versions
    uv pip install numpy==2.1.1 scipy==1.15.3 && \
    \
    # 3. Verify that the colocated_python_cpu_client is present.
    echo "--> Verifying JAX patch integrity..." && \
    python -c "from jax._src.lib import _jax; _jax.colocated_python_cpu_client" && \
    echo "--> JAX patch verification successful." && \
    \
    # 4. Clean the cache to keep the image slim.
    uv cache clean

################################################################################
# Final target spec.                                                           #
################################################################################

FROM ${TARGET} AS final
