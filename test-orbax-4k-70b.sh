#!/usr/bin/env bash

set -xe

export NUM_REPLICAS=${NUM_REPLICAS:-4}
export JOBSET_NAME=${JOBSET_NAME:-$USER}

# The bundle step is needed if you run on cloudtop
axlearn gcp bundle --name=$JOBSET_NAME \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu

axlearn gcp launch run --cluster=bodaborg-v6e-256-tt-c-new-2 \
        --runner_name gke_tpu_single \
        --name=$JOBSET_NAME \
        --queue=multislice-queue \
        --instance_type=tpu-v6e-256 \
        --priority_class=very-high \
        --host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --init_module=axlearn.common.checkpointer_orbax_emergency:local_ckpt_dir=/host-tmp/checkpoints \
          --module=text.gpt.c4_trainer \
          --config=fuji-70B-v2-flash-orbaxem \
          --trainer_dir=gs://tess-checkpoints-us-west1/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
          --data_dir=gs://axlearn-public/tensorflow_datasets  \
          --jax_backend=tpu \
          --mesh_selector=tpu-v6e-256-4 \
          --initialization_timeout=1200 \
          --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719
# Non orbax
# axlearn gcp launch run --cluster=bodaborg-v6e-256-tt-c \
#         --runner_name gke_tpu_single \
#         --name=$JOBSET_NAME \
#         --queue=multislice-queue \
#         --instance_type=tpu-v6e-256 \
#         --priority_class=very-high \
#         --num_replicas=${NUM_REPLICAS} \
#         --bundler_spec=allow_dirty=True \
#         --bundler_type=artifactregistry --bundler_spec=image=tpu \
#         --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
#         -- python3 -m axlearn.common.launch_trainer_main \
#           --module=text.gpt.c4_trainer \
#           --config=fuji-70B-v2-flash \
#           --trainer_dir=gs://tess-checkpoints-us-west1/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
#           --data_dir=gs://axlearn-public/tensorflow_datasets  \
#           --jax_backend=tpu \
#           --mesh_selector=tpu-v6e-256-4 \
#           --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719
