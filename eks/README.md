## Intro

This README explains how to build an axlearn container for running training jobs with AWS 
Trainium / AWS EC2 trn2 instances, and launch a single-node training job for Fuji-70B in 
Amazon Elastic Kubernetes Service (EKS).

These steps should be run from an x86-based Linux instance (ex: Ubuntu 22.04) hosted in the same region in which you
intend to launch your EKS cluster and Trainium instances.

If you do not yet have an EKS cluster with trn-enabled nodegroup you can refer to the following steps create a new EKS cluster
using the [Data On EKS](https://awslabs.github.io/data-on-eks/) project. Alternatively, if you already have an EKS cluster, you 
can refer to [this EKS/Trainium tutorial](https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain) 
to see how to create a Trainium enabled nodegroup and install the required Neuron & EFA k8s plugins and MPI operator. 

## Prereqs
* Install AWS CLI
* Install Docker
* Install eksctl and kubectl

## Build the axlearn container 
* Copy the Neuron binaries provided by your AWS team to the `neuron_binaries` directory
* Update `build_docker_image.sh` to reflect your AWS Account ID and ECR repo
* Run `./build_docker_image.sh` to build the axlearn container image and push it to your repo

## Launch training job
* Modify the included `launch_1node_job.yaml` to update the ECR image URIs (see lines containing `image:`).
* Launch the training job
```
kubectl apply -f ./launch_1node_job.yaml
```
* Check for running pods
```
kubectl get pods
```
* View training logs
```
kubectl logs -f YOUR_LAUNCHER_POD
```
