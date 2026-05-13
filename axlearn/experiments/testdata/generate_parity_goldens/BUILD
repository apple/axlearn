load("@rules_python//python/uv:lock.bzl", "lock")

exports_files(
    srcs = ["requirements_parity_lock.txt"],
    visibility = ["//visibility:public"],
)

lock(
    name = "requirements_parity",
    srcs = ["requirements_parity.in"],
    out = "requirements_parity_lock.txt",
    generate_hashes = False,
    args = ["--torch-backend=cpu"],
)
