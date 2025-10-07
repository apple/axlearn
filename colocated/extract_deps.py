"""A script to extract and filter dependencies from pyproject.toml."""

import toml

# Packages that have a direct dependency on jax, causing resolution conflicts.
# We will install these with --no-deps.
JAX_DEPENDENT_PACKAGES = {"chex", "optax", "flax", "aqtp", "seqio"}

with open("pyproject.toml", "r", encoding="utf-8") as f:
    pyproject = toml.load(f)

dependencies = pyproject.get("project", {}).get("dependencies", [])
optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})
core_dependencies = optional_dependencies.get("core", [])
gcp_dependencies = optional_dependencies.get("gcp", [])

all_deps = dependencies + core_dependencies + gcp_dependencies

deps_full = []
deps_nodeps = []

for dep in all_deps:
    # Normalize the package name by removing version specifiers.
    package_name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("!=")[0].strip()
    if not (dep.startswith("jax") or dep.startswith("pathwaysutils")):
        if package_name in JAX_DEPENDENT_PACKAGES:
            deps_nodeps.append(dep)
        else:
            deps_full.append(dep)

# Write the two separate requirements files.
with open("/tmp/requirements.txt", "w", encoding="utf-8") as f:
    for dep in deps_full:
        f.write(f"{dep}\n")

with open("/tmp/requirements_nodeps.txt", "w", encoding="utf-8") as f:
    for dep in deps_nodeps:
        f.write(f"{dep}\n")