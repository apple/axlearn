#!/usr/bin/env bash
#
# Copyright © 2023 Apple Inc.
#
# Boot script for GCP TPU-VM.
#
# To inspect startup script output:
# sudo journalctl -u google-startup-scripts.service

echo "=== AXLearn start_tpu.sh ==="

# Random sleep to prevent all TPU-VMs overwhelming pypi etc for large slices.
sleep $((1 + $RANDOM % 30))

# Get env-vars
get_metadata() {
  curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1 -H "Metadata-Flavor: Google"
}

BUCKET=$(get_metadata bundle_bucket)
JOB_NAME=$(get_metadata job_name)
ZONE=$(get_metadata zone)
TPU_TYPE=$(get_metadata tpu_type)
DOCKER_REGISTRY=$(get_metadata docker_registry)
BUNDLER_TYPE=$(get_metadata bundler_type)

GS_JOB_PATH="gs://${BUCKET}/axlearn/jobs/${JOB_NAME}"
BASE_LOG_ROOT="/output/logs"
mkdir -p ${BASE_LOG_ROOT}
chmod -R 776 ${BASE_LOG_ROOT}
SETUP_LOG_PATH="${BASE_LOG_ROOT}/setup.log"

echo "Using bundler type: ${BUNDLER_TYPE}" >> ${SETUP_LOG_PATH}

tar_bundlers=("tar" "gcs")
docker_bundlers=("docker" "artifactregistry" "cloudbuild")

if [[ " ${tar_bundlers[*]} " =~ " ${BUNDLER_TYPE} " ]]; then
  echo "Running tar bundler setup..." >> ${SETUP_LOG_PATH}

  # Install screen as it's seemingly the only way to persist via gcloud ssh command.
  apt-get update
  apt-get -y install screen htop tmux ca-certificates

  if [[ $TPU_TYPE == v3* ]]; then
    # If torch is installed, symlink the Intel MKL interface.
    # This symlink is not done automatically and that causes some problems when trying to
    # use torch on v3-8 tpus.
    # Match on "torch" only not "torchvision", etc.
    if [[ `python3 -m pip list | grep "^torch "` ]]; then
      ln -s /usr/local/lib/libmkl*.so.1 /usr/local/lib/python3.8/dist-packages/torch/lib
    fi

    # The default httplib2==0.19.1 that is installed accidentally removed support for a method.
    # We have to force install the right version of httplib2.
    # https://github.com/PAIR-code/what-if-tool/issues/185
    pip install -U httplib2==0.20.2
  fi
fi

# Ensure that gcloud is installed.
while [[ ! -x $(which gcloud) ]]
do
  echo "gcloud not found, installing..." >> ${SETUP_LOG_PATH}
  # Install google-cloud-sdk via apt.
  echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
  apt-get update && apt-get -y install google-cloud-sdk
  sleep 1
done

# Run after gcloud is guaranteed to be available.
if [[ " ${docker_bundlers[*]} " =~ " ${BUNDLER_TYPE} " ]]; then
  echo "Running docker bundler setup..." >> ${SETUP_LOG_PATH}

  while [[ ! -x $(which docker) ]]
  do
    echo "docker not found, installing..." >> ${SETUP_LOG_PATH}
    apt-get update && apt-get -y install docker.io >> ${SETUP_LOG_PATH} 2>&1
    sleep 1
  done
  echo "Using docker version: $(docker --version)" >> ${SETUP_LOG_PATH} 2>&1
  # Docker auth.
  yes | gcloud auth configure-docker ${DOCKER_REGISTRY} >> ${SETUP_LOG_PATH} 2>&1
fi

# Workaround an auth bug on v5lites.
echo '' > ~/.config/gcloud/configurations/config_default
gcloud auth list >> ${SETUP_LOG_PATH}

# Set ready label (has to be in GS because cannot set labels for TPU-VM without race conditions).
# TPU request-create-time set when slice (or multi-slice) creation was triggered.
CREATE_REQUEST_TIME=$(get_metadata create_request_time)
set_ready_label() {
  REMOTE_FLAG="${GS_JOB_PATH}/tpu_vm_ready_flags/${CREATE_REQUEST_TIME}/$(hostname)-ready"
  LOCAL_FLAG="/tmp/label"
  journalctl -u google-startup-scripts.service > "${LOCAL_FLAG}"
  gsutil cp "${LOCAL_FLAG}" "${REMOTE_FLAG}"
  echo "Finished setting ready label" >> ${SETUP_LOG_PATH}
}

# Cannot set TPU-VM label without race conditions, so instead place flag in GS.
echo "Begin setting ready label" >> ${SETUP_LOG_PATH}
set_ready_label

# Copy logs.
gsutil cp ${SETUP_LOG_PATH} "${GS_JOB_PATH}/logs/setup_log-$(hostname)"
