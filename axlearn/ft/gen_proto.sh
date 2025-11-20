#!/bin/bash

# script to generate *_pb2.py file before we have a build system.
# Note that this script must run under the axlearn root directory
# Usage: bash axlearn/common/ft/gen_proto.sh

skip_lint_and_pytype() {
    file=$1
    mv $file ${file}.orig
    echo "# pylint: skip-file" >${file}
    echo "# pytype: skip-file" >>${file}
    cat ${file}.orig >>${file}
    rm ${file}.orig
}

python -m grpc_tools.protoc -I. axlearn/ft/manager.proto --python_out=. --grpc_python_out=.

for file in axlearn/ft/manager_*pb2*; do
    skip_lint_and_pytype $file
done
