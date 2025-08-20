import pathwaysutils

print("initializing pathwaysutils")
pathwaysutils.initialize()
print("pathwaysutils initialized")

import numpy as np
import jax
from jax.experimental import colocated_python
import os
import shutil
import orbax.checkpoint as ocp

print("jax version on cpu host:", jax.__version__)

print("getting tpu devices")
tpu_devices = jax.devices()
print("tpu devices: ", tpu_devices)
print("getting cpu devices")
cpu_devices = colocated_python.colocated_cpu_devices(tpu_devices)
print("cpu devices: ", cpu_devices)

import cloudpickle

print("JAX_PLATFORMS is 'proxy'. Setting up pathways colocated python checkpointing.")
print(f" Using jax version {jax.__version__} and cloudpickle version {cloudpickle.__version__}")


print("def add_one")


@colocated_python.colocated_python
def add_one(x):
  import sys

  sys.stderr.write("In colocated python function \n")
  sys.stderr.write(f"[Colocated] jax version: {jax.__version__} \n")
  sys.stderr.write("[Colocated] add_one")
  sys.stderr.write(f"[Colocated] x:  {x} on device:  {x.device } \n")
  return x+1


print("creating input 1")
x = np.array(1)
print("putting on device")
x = jax.device_put(x, cpu_devices[0])

print("adding one to input 1")
out = add_one(x)
print("getting out")
out = jax.device_get(out)
print("out 1: ", out)

print("creating input 2")
x = np.array(5)
print("putting on device")
x = jax.device_put(x, cpu_devices[0])

assert out == 2, f"out: {out}"

print("adding one to input 2")
out = add_one(x)
print("getting out")
out = jax.device_get(out)
print("out 2: ", out)

assert out == 6, f"out: {out}"