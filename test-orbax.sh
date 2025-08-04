#!/usr/bin/env bash

set -xe

export NUM_REPLICAS=${NUM_REPLICAS:-2}
export JOBSET_NAME=${JOBSET_NAME:-$USER}
export BASTION_TIER=disabled
export GKE_CLUSTER=$(axlearn gcp config | grep gke_cluster | awk '{ print $3 }' | tr -d '"')
# Switch to tpu-v6e-256 if on scale cluster
export INSTANCE_TYPE=${INSTANCE_TYPE:-"tpu-v6e-16"}
# Switch to tpu-v6e-256-4 if on scale cluster
export MESH_SELECTOR=${MESH_SELECTOR:-"tpu-v6e-16"}
# Need to use tiktoken when saving data iterator
# export CONFIG=${CONFIG:-"fuji-8B-v3-tiktoken-flash-orbax"}
export CONFIG=${CONFIG:-"fuji-7B-v3-flash-orbaxem"}
export PROJECT_ID=$(gcloud config get project)
export TRAINER_DIR=gs://tpu-prod-env-multipod-use4

# Example for v6e-256
# MESH_SELECTOR=tpu-v6e-256-4 INSTANCE_TYPE=tpu-v6e-256 ./test-orbax.sh

# The bundle step is needed if you run on cloudtop
# uncomment if you use cloudtop
# axlearn gcp bundle --name=$JOBSET_NAME \
#         --bundler_spec=allow_dirty=True \
#         --bundler_type=artifactregistry \
#         --bundler_spec=dockerfile=Dockerfile \
#         --bundler_spec=image=tpu \
#         --bundler_spec=target=tpu

# Only enable kueue when running on scale testing cluster
# --queue=multislice-queue \
# --priority_class=very-high \
# --trainer_dir=gs://tess-checkpoints-us-west1/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
#

# Check if CONFIG ends with "orbaxem"
if [[ "$CONFIG" == *"orbaxem"* ]]; then
  echo "Running with Orbax emergency checkpointer."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
        --runner_name gke_tpu_single \
        --queue=multislice-queue \
        --priority_class=very-high \
        --name=$JOBSET_NAME \
        --instance_type=${INSTANCE_TYPE} \
        --host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --init_module=axlearn.common.checkpointer_orbax_emergency:local_ckpt_dir=/host-tmp/checkpoints \
          --module=text.gpt.c4_trainer \
          --config=${CONFIG} \
          --trainer_dir=${TRAINER_DIR}/${JOBSET_NAME} \
          --data_dir=gs://axlearn-public/tensorflow_datasets  \
          --jax_backend=tpu \
          --mesh_selector=${MESH_SELECTOR} \
          --initialization_timeout=1200 \
          --recorder_type=axlearn.cloud.gcp.measurement:goodput \
          --recorder_spec=name=goodput_${JOBSET_NAME} \
          --recorder_spec=upload_dir=${TRAINER_DIR}/summaries \
          --recorder_spec=upload_interval=30 \
          --recorder_spec=rolling_window_size=3600,7200,10800,86400 \
          --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719

else
  echo "Running Orbax regular checkpointer or AXLearn native."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
        --runner_name gke_tpu_single \
        --name=$JOBSET_NAME \
        --instance_type=${INSTANCE_TYPE} \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --module=text.gpt.c4_trainer \
          --config=${CONFIG} \
          --trainer_dir=gs://${PROJECT_ID}-axlearn/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
          --data_dir=gs://axlearn-public/tensorflow_datasets  \
          --jax_backend=tpu \
          --mesh_selector=${MESH_SELECTOR} \
          --initialization_timeout=1200 \
          --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719
fi
