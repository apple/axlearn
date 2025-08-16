# Pytype Integration for AXLearn

This document describes the pytype static type checking integration for AXLearn.

## What We Did

### 1. Created Pytype Bazel Rules

- Location: `pytype_bazel/pytype.bzl`
- BUILD file: `pytype_bazel/BUILD`
- Demo: `pytype_bazel/demo/` (working example)

**Important**: Directory is named `pytype_bazel` (not `pytype`) to avoid name collision with the pip-installed pytype package.

### 2. Added Pytype Wrapper Macros

The integration provides three wrapper macros that replace standard Bazel Python rules:

- `pytype_library()` - wraps `py_library`
- `pytype_binary()` - wraps `py_binary`
- `pytype_test()` - wraps `py_test`

### 3. Automatic Type Checking

Each macro automatically creates a corresponding `*_pytype_check` test target:

- `//my_package:my_lib` → `//my_package:my_lib_pytype_check`
- All pytype checks are tagged with `"pytype"` for easy filtering

### 4. Added Pytype Dependency

- Added `pytype` to `requirements.in`
- Updated `requirements_lock.txt` with pytype and its transitive dependencies
- Upgraded to Python **3.12** for better pytype compatibility

## How to Use

### Basic Usage

Replace standard Bazel rules with pytype equivalents:

```python
# Before
load("@rules_python//python:defs.bzl", "py_library", "py_binary", "py_test")

py_library(
    name = "my_lib",
    srcs = ["my_lib.py"],
    deps = ["@pip//numpy"],
)

# After
load("//pytype_bazel:pytype.bzl", "pytype_library")

pytype_library(
    name = "my_lib",
    srcs = ["my_lib.py"],
    deps = ["@pip//numpy"],
)
```

### Running Type Checks

```bash
# Run all pytype checks
bazel test --test_tag_filters=pytype //...

# Run pytype checks without cache
bazel test --cache_test_results=no --test_tag_filters=pytype //...

# Run specific pytype check
bazel test //my_package:my_lib_pytype_check

# Run all tests EXCEPT pytype
bazel test --test_tag_filters=-pytype //...
```

### Working Example

See `pytype_bazel/demo/` for a complete working example:

```bash
# Run the demo application (should pass pytype)
bazel run //pytype_bazel/demo:should_pass_pytype

# Test type checking (should pass)
bazel test //pytype_bazel/demo:should_pass_pytype_pytype_check

# Demo with intentional type error (will fail to demonstrate error detection)
bazel test //pytype_bazel/demo:should_fail_pytype_pytype_check
```

### Disabling Type Checking

```python
pytype_library(
    name = "my_lib",
    srcs = ["my_lib.py"],
    enable_pytype = False,  # Disable type checking
)
```

## Troubleshooting

### Name Collision Issue

**Problem**: If you name your rules directory `pytype/`, it will shadow the pip-installed pytype package.

**Solution**: Use `pytype_bazel/` or another name to avoid collision.

## Future Integration

Other Bazelized projects can use `pytype_*` rules by declaring AXLearn as a dependency and loading `pytype.bzl` from AXLearn.

```python
# In AXLearn BUILD files
load("@axlearn//pytype_bazel:pytype.bzl", "pytype_library")

pytype_library(
    name = "my_lib",
    srcs = ["my_lib.py"],
    deps = [...],
)
```
