FROM public.ecr.aws/ubuntu/ubuntu:22.04_stable

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP="python3.10 -m pip"

WORKDIR /neuron

COPY ./neuron_binaries /neuron/packages
COPY ./assets/nodelist_helper.py /neuron/scripts/
COPY ./assets/launch_training.sh /neuron/
ADD https://apt.repos.neuron.amazonaws.com/pool/main/a/aws-neuronx-tools/aws-neuronx-tools_2.17.1.0_amd64.deb /neuron/packages
ADD https://pip.repos.neuron.amazonaws.com/aws-neuronx-runtime-discovery/aws-neuronx-runtime-discovery-2.9.tar.gz /neuron/packages

# Install dependencies, common utils
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    jq \
    software-properties-common \
    wget \
    unzip \
    vim \
    zlib1g-dev \
    openssl \
    libssl-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    tk-dev \
    libffi-dev \
    libcap-dev \
    gnupg2 \
    gpg-agent \
    pciutils \
    python3.10-full \
    python3.10-dev \
    python3-pip \
    cython3 \
    inetutils-ping \
    google-perftools \
    patch \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# EFA Installer - required - installs libfabric (but no EFA driver) inside the container
RUN apt-get update \
    && cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \
    && wget https://efa-installer.amazonaws.com/aws-efa-installer.key && gpg --import aws-efa-installer.key \
    && cat aws-efa-installer.key | gpg --fingerprint \
    && wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz.sig && gpg --verify ./aws-efa-installer-latest.tar.gz.sig \
    && tar -xf aws-efa-installer-latest.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -fr /root/aws-efa-installer* \
    && cd $HOME \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/tmp* \
    && apt-get clean

# Install req'd packages and configure SSH for MPI operator in k8s
RUN apt-get update && apt-get install -y openmpi-bin openssh-server \
    && mkdir -p /var/run/sshd \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/tmp* \
    && apt-get clean

# Install Neuron system packages
RUN cd packages \
  && apt-get update \
  && apt install -y ./aws-neuronx-collectives-* ./aws-neuronx-runtime-lib-* ./aws-neuronx-tools* \
	&& rm -rf /var/lib/apt/lists/* \
	&& rm -rf /tmp/tmp* \
	&& apt-get clean

# Install Neuron pip packages
RUN cd packages \
  && tar xzf aws-neuronx-runtime-discovery-2.9.tar.gz \
  && $PIP install ./aws-neuronx-runtime-discovery-2.9 \
  && $PIP install ./libneuronxla-2*.whl ./neuronx_cc-2.*-cp310-*linux_x86_64.whl \
  && rm -fr /root/.cache \
  && rm -fr /neuron/packages

# Install axlearn
RUN git clone https://github.com/ptoulme-aws/axlearn.git \
  && cd axlearn \
  && git checkout aws-release-0.0.1 \
  && cd .. \
  && $PIP install -U pip \
  && $PIP install -e ./axlearn

RUN echo "export PATH=/opt/aws/neuron/bin:\$PATH" >> /root/.bashrc
ENV PATH="${PATH}:/opt/aws/neuron/bin"
