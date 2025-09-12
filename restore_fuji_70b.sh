python3 -m axlearn.common.launch_trainer_main \
        --module=text.gpt.c4_trainer \
        --config=fuji-70B-v3-flash \
        --trainer_dir=gs://cloud-tpu-multipod-dev-uss1/axlearn-fuji-v3-70b/ \
        --data_dir=gs://axlearn-public/tensorflow_datasets  \
        --jax_backend=proxy \
        --mesh_selector=tpu-v5litepod-32-1 \
        --trace_at_steps=11
