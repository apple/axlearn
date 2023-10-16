# syntax=docker/dockerfile:1

# Define build arguments.
ARG TARGET=base
ARG BASE_IMAGE=python:3.9-slim

# Base image for common dependencies.
FROM ${BASE_IMAGE} AS base

RUN apt-get update
RUN apt-get install -y apt-transport-https ca-certificates gnupg curl gcc g++

# Install necessary packages in a single layer.
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates gnupg curl gcc g++ git jq screen ca-certificates && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-cli -y

# Set up the working directory and copy minimal required files.
WORKDIR /root
COPY README.md pyproject.toml axlearn/__init__.py axlearn/
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV && pip install flit && pip install --upgrade pip

################################################################################
# CI container spec.                                                           #
################################################################################

# Leverage multi-stage build for unit tests.
FROM base as ci

# TODO(markblee): Remove gcp,vertexai_tensorboard from CI.
RUN pip install .[dev,gcp,vertexai_tensorboard]

# Copy the application source code.
COPY . .

# Defaults to an empty string, i.e. run pytest against all files.
ARG PYTEST_FILES=''
# `exit 1` fails the build.
RUN ./run_tests.sh "${PYTEST_FILES}"

################################################################################
# Bastion container spec.                                                      #
################################################################################

# Bastion container for application deployment.
FROM base AS bastion

# TODO(markblee): Consider copying large directories separately, to cache more aggressively.
# TODO(markblee): Is there a way to skip the "production" deps?
COPY . /root/
# Install production dependencies.
RUN pip install .[gcp,vertexai_tensorboard]

################################################################################
# Dataflow container spec.                                                     #
################################################################################

# Dataflow container for data processing.
FROM base AS dataflow

# Install data processing dependencies.
RUN pip install .[gcp,dataflow]

# Copy the application source code.
COPY . .

################################################################################
# Final target spec.                                                           #
################################################################################

# Final target image.
FROM ${TARGET} AS final
