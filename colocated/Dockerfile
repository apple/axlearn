# Use the JAX image with the custom-built sidecar as the base.
# FROM  gcr.io/cloud-tpu-multipod-dev/sujinesh_sidecar_debug@sha256:19abcd94addb6ff2749c299d6b0cc4748f27a4ab8759a18b466d0bdd3e5b71e8
FROM us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:python_3.10-jax_0.6.2
# Defines a build argument for the requirements file. This contains the user's custom
# dependencies formatted as a requirements file.
ARG REQUIREMENTS_FILE

# Set the working directory (this is already inherited)
WORKDIR /app

# Copy the user's requirements file into the image.
COPY ${REQUIREMENTS_FILE} .

# Install the additional user-provided dependencies, strictly enforcing the rules
# from the base image's constraints file.
RUN \
    # Safeguard: Explicitly fail the build if the user tries to reinstall JAX or JAXlib.
    # if grep -i -E '^jax(lib)?' ${REQUIREMENTS_FILE}; then \
    #     echo "ERROR: Your requirements file attempts to re-install 'jax' or 'jaxlib'." >&2; \
    #     echo "Please remove these lines. The base image provides a specific, custom-patched version of JAX." >&2; \
    #     exit 1; \
    # fi && \
    \
    # If the safeguard passes, proceed with the constrained installation.
    uv pip install --prerelease=allow -r ${REQUIREMENTS_FILE} -c /opt/venv/server_constraints.txt && \
    \
    # Clean the cache to keep the image slim.
    uv cache clean

# Note: The ENTRYPOINT and CMD are inherited from the base image, so they do not
# need to be redefined here. I.e. the sidecar will be launched automatically.

