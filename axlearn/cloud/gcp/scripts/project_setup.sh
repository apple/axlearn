# Sets up a brand new GCP project for AXLearn. See also the "Getting Started" docs linked in the main readme.
# Note: Running this script does NOT grant you TPU quota.
# Please follow instructions here: https://cloud.google.com/tpu/docs/setup-gcp-account#prepare-to-request


# Usage:
#    # fill out environment variables below
#    chmod +x project_setup.sh
#    ./project_setup.sh

# This will provision the following resources:
#   * A new GCP project under your organization
#   * A default VPC network
#   * Three GCS buckets
#   * A service account with the Storage Admin role


#!/bin/sh

set -e

# Define environment variables
export ORGANIZATION_ID= #your existing GCP organization
export BILLING_ACCOUNT_ID= #an existing billing account
export PROJECT_ID= #the project name you want to create or already obtained
export NETWORK_NAME=default
export TPU_REGION=
export PERMANENT_BUCKET_NAME=${PROJECT_ID}-perm
export PRIVATE_BUCKET_NAME=${PROJECT_ID}-private
export TTL_BUCKET_NAME=${PROJECT_ID}-ttl
export BUCKET_REGION=us-central1
export SERVICE_ACCOUNT_NAME=
export SERVICE_ACCOUNT_ID=${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com

# Step 1: Create a GCP project
# https://cloud.google.com/resource-manager/docs/creating-managing-projects
# Uncomment the following line if you haven't created the project
# gcloud projects create $PROJECT_ID --organization=$ORGANIZATION_ID
gcloud config set project $PROJECT_ID

# Step 2: Link the project to a billing account
# https://cloud.google.com/billing/docs/how-to/modify-project
# Uncomment the following line if you haven't created the project
# gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT_ID

# Step 3: Enable services
gcloud services enable compute.googleapis.com
gcloud services enable tpu.googleapis.com

# Step 4: Create an auto mode VPC
# https://cloud.google.com/vpc/docs/create-modify-vpc-networks#create-auto-network
if ! gcloud compute networks describe $NETWORK_NAME -q >/dev/null; then
  gcloud compute networks create $NETWORK_NAME --subnet-mode=auto
fi
# Note: certain TPUs (eg. v4) are located in a private region (eg. us-central2).
# You will need to request quota before creating a subnet in that case:
# https://cloud.google.com/tpu/docs/setup-gcp-account#prepare-to-request

# Step 5: Create GCS buckets
# https://cloud.google.com/storage/docs/creating-buckets#storage-create-bucket-cli
if ! gcloud storage buckets describe "gs://${PERMANENT_BUCKET_NAME}" -q >/dev/null; then
  gcloud storage buckets create gs://${PERMANENT_BUCKET_NAME} --location=$BUCKET_REGION --uniform-bucket-level-access
fi

if ! gcloud storage buckets describe "gs://${PRIVATE_BUCKET_NAME}" -q >/dev/null; then
  gcloud storage buckets create gs://$PRIVATE_BUCKET_NAME --location=$BUCKET_REGION --uniform-bucket-level-access
fi

if ! gcloud storage buckets describe "gs://${TTL_BUCKET_NAME}" -q >/dev/null; then
  gcloud storage buckets create gs://$TTL_BUCKET_NAME --location=$BUCKET_REGION --uniform-bucket-level-access
fi
# Note: to configure object lifecycles (eg. set a time to live), please follow instructions here:
# https://cloud.google.com/storage/docs/managing-lifecycles#permissions-console

# Step 6: Create a Service Account
# https://cloud.google.com/iam/docs/service-accounts-create#iam-service-accounts-create-gcloud
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_ID -q >/dev/null; then
  gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME
fi

# Step 7: Grant the Service Account the necessary roles
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
      --member='serviceAccount:'${SERVICE_ACCOUNT_ID} \
      --role='roles/storage.admin'

# Step 8: Print useful variables needed for AXLearn configuration (located in .axlearn/.axlearn.config)
echo "===============Set up completed===================="
echo "project=${PROJECT_ID}"
echo "network=$(gcloud compute networks list --filter="name=($NETWORK_NAME)" --uri)"
echo "subnetwork=$(gcloud compute networks subnets list --uri --filter="region:$TPU_REGION")"
echo "permanent_bucket=${PERMANENT_BUCKET_NAME}"
echo "private_bucket=${PRIVATE_BUCKET_NAME}"
echo "ttl_bucket=${TTL_BUCKET_NAME}"
