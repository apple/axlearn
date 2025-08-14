load("@rules_python//python:pip.bzl", "compile_pip_requirements")

package(default_visibility = ["//visibility:public"])

# Compiles requirements.in into requirements_lock.txt.
compile_pip_requirements(
    name = "requirements",
    timeout = "moderate",  # timeout needs to explicitly set because compile takes more than 1 min.
    src = "requirements.in",
    generate_hashes = False,
    requirements_txt = "requirements_lock.txt",
)
