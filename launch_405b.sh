#!/usr/bin/env bash

set -xe

export NUM_REPLICAS=${NUM_REPLICAS:-1}
export JOBSET_NAME=${JOBSET_NAME:-$USER}
# export BASTION_TIER=disabled
# local launch automatically sets tier to disabled
export GKE_CLUSTER=$(axlearn gcp config | grep gke_cluster | awk '{ print $3 }' | tr -d '"')
export INSTANCE_TYPE=${INSTANCE_TYPE:-"tpu-v5p-64"}
export MESH_SELECTOR=${MESH_SELECTOR:-"tpu-v5p-64"}
export CONFIG=${CONFIG:-"fuji-70B-v3-flash"}
export PROJECT_ID=$(gcloud config get project)
export TRAINER_DIR=gs://${PROJECT_ID}-axlearn
export RESERVATION=${RESERVATION:-"cloudtpu-20240716121201-595617744"}


# Only enable kueue when running on scale testing cluster
# --queue=multislice-queue \
# --priority_class=very-high \
# --trainer_dir=gs://tess-checkpoints-us-west1/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
# For goodput logging
#           --recorder_type=axlearn.cloud.gcp.measurement:goodput \
#           --recorder_spec=name=goodput_${JOBSET_NAME} \
#           --recorder_spec=upload_dir=${TRAINER_DIR}/summaries \
#           --recorder_spec=upload_interval=30 \
#           --recorder_spec=rolling_window_size=3600,7200,10800,86400 \
#           --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719

axlearn gcp launch run --cluster=$GKE_CLUSTER \
      --runner_name gke_tpu_single \
      --queue=multislice-queue \
      --reservation=${RESERVATION} \
      --name=$JOBSET_NAME \
      --instance_type=${INSTANCE_TYPE} \
      --num_replicas=${NUM_REPLICAS} \
      --bundler_spec=allow_dirty=True \
      --bundler_type=artifactregistry --bundler_spec=image=tpu \
      --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
      -- "ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
        --module=text.gpt.c4_trainer \
        --config=${CONFIG} \
        --trainer_dir=${TRAINER_DIR}/${JOBSET_NAME} \
        --data_dir=gs://axlearn-public/tensorflow_datasets  \
        --jax_backend=tpu \
        --mesh_selector=${MESH_SELECTOR} \
        --initialization_timeout=1200 \
        --trainer_crash_on_hang_timeout_seconds=300 \
        --trainer_watchdog_timeout_seconds=290 \
        --trace_at_steps=11
