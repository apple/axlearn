#!/bin/bash
export DOCKER_BUILDKIT=1

ECR_URI=<aws-account-id>.dkr.ecr.us-east-2.amazonaws.com
ECR_REPO=axlearn_neuronx

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin $ECR_URI
if [[ $? -eq 0 ]]; then
  docker build . -t $ECR_URI/$ECR_REPO && docker push $ECR_URI/$ECR_REPO
fi
