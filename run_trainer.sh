#! /bin/bash
source /shared_new/ptoulme/axlearn/venv/bin/activate
#source /shared/apoorvgu/jax-21/bin/activate
source ./setup.sh
source ./train_setup.sh

# pip install /shared/apoorvgu/libneuronxla-2.0.20240411a0-py3-none-linux_x86_64.whl
# pip install -e /shared/apoorvgu/axlearn/
echo "==============================================="
apt list | grep neuron
pip freeze | grep neuron
echo "==============================================="

# rm -rf /shared/apoorvgu/fs_drop/axlearn/test/*
# rm -rf /shared/apoorvgu/fs_drop/axlearn/compiler_dump
# rm -rf /shared/apoorvgu/fs_drop/axlearn/jax_dump
# rm -rf /shared/apoorvgu/fs_drop/axlearn/jax4_dump
OUTPUT_DIR=/shared_new/ptoulme/fs_drop/axlearn/out/
# DATA_DIR=FAKE
DATA_DIR=gs://axlearn-public/tensorflow_datasets
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B \
    --trainer_dir=$OUTPUT_DIR --data_dir=$DATA_DIR \
    --jax_backend=neuron --mesh_selector=neuron-trn1.32xlarge-64