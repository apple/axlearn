#!/usr/bin/env python3
# Helper script to create a list of worker pods participating in
#   a distributed training job based on current worker's hostname
#   and the OMPI world size
import re
import os
import subprocess

if not os.environ.get("PMIX_HOSTNAME"):
    raise Exception("Error: This script should be run via mpirun on EKS")

this_host = subprocess.check_output(["hostname", "--fqdn"]).decode().strip()
s = re.search(r"^(.*-worker)-\d+(.*)", this_host)
host_prefix = s.group(1)
host_suffix = s.group(2)
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))

hosts = []
for x in range(world_size):
    hosts.append(f"{host_prefix}-{x}{host_suffix}")

print("\n".join(hosts))

