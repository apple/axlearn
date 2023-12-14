# Set up GCP Environment

## Pre-requisites
We assume you have:

1. A Google Cloud Platform (GCP) [organization](https://cloud.google.com/resource-manager/docs/creating-managing-organization)
2. A Google Cloud Platform (GCP) [billing account](https://cloud.google.com/billing/docs/how-to/find-billing-account-id)
3. Installed [gcloud CLI](https://cloud.google.com/sdk/docs/install#mac)

## Provision the resources needed for AXLearn
1. Navigate to `project_setup.sh` and fill in the environment variables
```
cd axlearn/axlearn/cloud/gcp
vim project_setup.sh
```
2. Execute shell script to provision resources programmatically
```
chmod +x project_setup.sh
./project_setup.sh
```
This step will provision the following resources:
* A new GCP project under your organization
* A default VPC network
* Three GCS buckets
* A service account with the Storage Admin role

## Next Steps
Now you are ready to [set up the AXLearn tool](01-start.md) and launch training jobs.