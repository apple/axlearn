# syntax=docker/dockerfile:1

ARG TARGET=base

FROM python:3.8-slim AS base

RUN apt-get update
RUN apt-get install -y apt-transport-https ca-certificates gnupg curl gcc g++

# Install git.
RUN apt-get install -y git

# Install gcloud. https://cloud.google.com/sdk/docs/install
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
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

RUN pip install .[dev]
COPY . .

# Defaults to an empty string, i.e. run pytest against all files.
ARG PYTEST_FILES=''
# `exit 1` fails the build.
RUN ./run_tests.sh "${PYTEST_FILES}"

################################################################################
# Final target spec.                                                           #
################################################################################

FROM ${TARGET} AS final
