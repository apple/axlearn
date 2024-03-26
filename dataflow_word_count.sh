# chmod +x dataflow_word_count.sh
# ./dataflow_word_count.sh

DOCKER_REPO=us-central1-docker.pkg.dev/{PROJECT_ID}/axlearn
DOCKER_IMAGE=axlearn-dataflow
DOCKER_TAG=test

axlearn gcp dataflow start \
--bundler_spec=dockerfile=Dockerfile \
--bundler_spec=repo=${DOCKER_REPO} \
--bundler_spec=image=${DOCKER_IMAGE} \
--bundler_spec=target=dataflow -- "'rm -r /tmp/output_dir; \
    python3 -m apache_beam.examples.wordcount \
        --input=gs://dataflow-samples/shakespeare/kinglear.txt \
        --output=/tmp/output_dir/outputs \
    '"
