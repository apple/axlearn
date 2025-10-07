#!/usr/bin/env bash

set -e -x

export TF_CPP_MIN_LOG_LEVEL=3  # Filter INFO + WARNING + ERROR, leave only FATAL
export TF_TRT_IS_AVAILABLE=0  # Suppress TF-TRT Warning: Could not find TensorRT
export TF_DISABLE_TRT=1  # Suppress TF-TRT Warning: Could not find TensorRT

# Install the package (necessary for CLI tests).
# Requirements should already be cached in the docker image.
pip install -qq -e .

exit_if_error() {
  local exit_code=$1
  shift
  printf 'ERROR: %s\n' "$@" >&2
  exit "$exit_code"
}

precommit_checks() {
  set -e -x
  pre-commit install
  pre-commit run --all-files || exit_if_error $? "pre-commit failed."
  # Run pytype separately to utilize all cpus and for better output.
  pytype -j auto axlearn || exit_if_error $? "pytype failed."
}

# Collect all background PIDs explicitly.
TEST_PIDS=()

if [[ "${1:-x}" = "--skip-pre-commit" ]] ; then
  SKIP_PRECOMMIT=true
  shift
fi

# Skip pre-commit on parallel CI because it is run as a separate job.
if [[ "${SKIP_PRECOMMIT:-false}" = "false" ]] ; then
  precommit_checks &
  TEST_PIDS[$!]=1
fi

UNQUOTED_PYTEST_FILES=$(echo $1 |  tr -d "'")
pytest -W error --durations=100 -n auto \
  -m "not (gs_login or tpu or high_cpu or fp64 or for_8_devices)" ${UNQUOTED_PYTEST_FILES} \
  --dist worksteal &
TEST_PIDS[$!]=1

JAX_ENABLE_X64=1 pytest -W error --durations=100 -n auto -m "fp64" --dist worksteal &
TEST_PIDS[$!]=1

XLA_FLAGS="--xla_force_host_platform_device_count=8" pytest -W error --durations=100 \
  -n auto -m "for_8_devices" --dist worksteal &
TEST_PIDS[$!]=1

# Use Bash 5.1's new wait -p feature to quit immediately if any subprocess fails to make error
# finding a bit easier.
while [ ${#TEST_PIDS[@]} -ne 0 ]; do
  wait -n -p PID ${!TEST_PIDS[@]} || exit_if_error $? "Test failed."
  unset TEST_PIDS[$PID]
done
