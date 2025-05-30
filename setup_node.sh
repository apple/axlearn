#! /bin/bash

if [ -z "$1" ]
  then
    echo "Need to pass dir to find the deb files to install. Use the script as follows."
	echo "./setup_node.sh <path_to_artifacts>"
fi

set -x
SECONDS=0

# sudo dpkg --configure -a
# # Configure Ubuntu for Neuron repository updates
# . /etc/os-release
# sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
# deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
# EOF
# wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
# 	    | sudo apt-key add -

# Update OS packages and install OS headers

sudo apt-get update - > /dev/null

sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10

sudo apt update
sudo apt install -y python3.10-venv python3.10-dev python3-pip

# Needed for TC_MALLOC fix
sudo apt-get -f install -y
sudo apt-get install -y google-perftools

# Binaries to use:
###

ENV_DROP_DIR=${1:-../mar-artifacts}

RUNTIME=$ENV_DROP_DIR/aws-neuronx-runtime-lib-*.deb
COLLECTIVES=$ENV_DROP_DIR/aws-neuronx-collectives-*.deb
TOOLS=$ENV_DROP_DIR/aws-neuronx-tools-*.deb
DKMS=$ENV_DROP_DIR/aws-neuronx-dkms_*.deb

sudo dpkg -i $RUNTIME $COLLECTIVES $TOOLS #$DKMS

# sudo apt-get install -y linux-headers-$(uname -r) || true
# sudo apt-get remove -y aws-neuronx-devtools || true

# sudo apt-get remove -y --allow-change-held-packages aws-neuronx-tools aws-neuronx-collectives aws-neuronx-runtime-lib
# # Install Neuron OS packages and dependencies
sudo dpkg -i $RUNTIME $COLLECTIVES #$TOOLS #$DKMS
# sudo apt-get -o Dpkg::Options::="--force-overwrite" install --reinstall --allow-downgrades -y aws-neuronx-dkms

# # Tracing collectives
# sudo apt-get install -y bpfcc-tools linux-headers-$(uname -r)
# sudo python3 -m pip install psutil

TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s  http://169.254.169.254/latest/meta-data/instance-id)
echo "instance_id:$INSTANCE_ID hostname:$(hostname)"
echo "runtime versions"
echo "==============================================="
echo "Dependency versions"
echo "==============================================="
apt list | grep neuron | grep installed
echo "Setup took $SECONDS s"
