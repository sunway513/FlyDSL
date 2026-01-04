#!/bin/bash
# dumpir.sh <script> <arguments>
# Example: ./dumpir.sh python tests/kernels/benchmark/matrixTranspose.py

export FLIR_DUMP_IR=1
#export FLIR_DUMP_DIR=.kernel_dumps

export PYTHONPATH="${SCRIPT_DIR}/flydsl/src:${PYTHON_PACKAGE_ROOT}:${SCRIPT_DIR}:${PYTHONPATH}"

exec "$@"
