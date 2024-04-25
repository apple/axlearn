## Intro

This README explains how to build an axlearn container for running distributed training jobs with AWS 
Trainium / AWS EC2 trn1(n) instances, and launch a 64-node training job for Fuji-7B in Amazon Elastic Kubernetes Service (EKS).

These steps should be run from an x86-based Linux instance (ex: Ubuntu 22.04) hosted in the same region in which you
intend to launch your EKS cluster and Trainium instances.

If you do not yet have an EKS cluster with trn1n-enabled nodegroup you can refer to the following steps create a new EKS cluster
using the [Data On EKS](https://awslabs.github.io/data-on-eks/) project. Alternatively, if you already have an EKS cluster, you 
can refer to [this EKS/Trainium tutorial](https://github.com/aws-neuron/aws-neuron-eks-samples/tree/master/dp_bert_hf_pretrain) 
to see how to create a Trainium enabled nodegroup and install the required Neuron & EFA k8s plugins and MPI operator. 

Note: for best performance, please use `trn1n.32xlarge` instances (not `trn1.32xlarge`)

## Prereqs
* Install Terraform (only required if you are building a new EKS cluster via Data on EKS)
* Install AWS CLI
* Install Docker
* Install eksctl and kubectl

## Build the axlearn container 
* Copy the Neuron binaries provided by your AWS team to the `neuron_binaries` directory
* Update `build_docker_image.sh` to reflect your AWS Account ID and ECR repo
* Run `./build_docker_image.sh` to build the axlearn container image and push it to your repo

## Create an EKS cluster using Data on EKS
* Clone the Data On EKS repo `git clone https://github.com/awslabs/data-on-eks.git`
* `cd data-on-eks/ai-ml/trainium-inferentia`
* Specify your desired Data on EKS cluster configuration
```
export TV_VAR_name=doeks-jax-trn
export TF_VAR_enable_mpi_operator=true
export TF_VAR_region=us-east-2
export TF_VAR_trn1n_32xl_min_size=0
export TF_VAR_trn1n_32xl_desired_size=0
```
* Create the EKS cluster using Terraform
Note: this will launch the EKS cluster with initially empty trn1n nodegroup
```
./install.sh
```
* Update your kubeconfig file so you can use kubectl with the new cluster
```
aws eks --region us-east-2 update-kubeconfig --name doeks-jax-trn
```
* List your EKS nodegroups and look for the one associated with trn1 instances:
```
eksctl get nodegroup --cluster doeks-jax-trn --region us-east-2
```
* Scale-up your trn1 nodegroup as required
```
eksctl scale nodegroup NODEGROUP_NAME --cluster doeks-jax-trn --nodes XX --nodes-min XX --nodes-max XX --region us-east-2
```

## Launch training job
* Modify the included `launch_64node_job.yaml` to update the number of workers (should be equal to your trn1 nodegroup size)
and the ECR image URIs (see lines containing `image:`).
* Launch the training job
```
kubectl apply -f ./launch_64node_job.yaml
```
* Check for running pods
```
kubectl get pods
```
* View training logs
```
kubectl logs -f YOUR_LAUNCHER_POD
```
