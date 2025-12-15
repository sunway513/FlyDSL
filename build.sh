#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Set up environment
if [ -z "$MLIR_PATH" ]; then
    # Default path based on build_llvm.sh
    DEFAULT_MLIR_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)/llvm-project/buildmlir"
    if [ -d "$DEFAULT_MLIR_PATH" ]; then
        echo "MLIR_PATH not set. Using default: $DEFAULT_MLIR_PATH"
        export MLIR_PATH="$DEFAULT_MLIR_PATH"
    else
        echo "Error: MLIR_PATH not set and default location ($DEFAULT_MLIR_PATH) not found."
        echo "Please run ./build_llvm.sh first or set MLIR_PATH."
        exit 1
    fi
fi

# Note: No longer need to install MLIR Python packages separately!
# The new build system embeds all MLIR Python dependencies

# Build C++ components
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
cmake .. \
    -DMLIR_DIR="$MLIR_PATH/lib/cmake/mlir" \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_RUNTIME=OFF

# Build core targets (skip type stub generation which may fail)
echo "Building core libraries..."
cmake --build . --target RocirDialect -j$(nproc) || { echo "Failed to build RocirDialect"; exit 1; }
cmake --build . --target RocirTransforms -j$(nproc) || { echo "Failed to build RocirTransforms"; exit 1; }
cmake --build . --target MLIRCAPIRocir -j$(nproc) || { echo "Failed to build MLIRCAPIRocir"; exit 1; }
cmake --build . --target RocirPythonCAPI -j$(nproc) || { echo "Failed to build RocirPythonCAPI"; exit 1; }

# Build Python extension modules
echo "Building Python extensions..."
cmake --build . --target RocirPythonModules.extension._mlir.dso -j$(nproc) || { echo "Failed to build _mlir.dso"; exit 1; }
cmake --build . --target RocirPythonModules.extension._mlirDialectsGPU.dso -j$(nproc) || { echo "Failed to build _mlirDialectsGPU.dso"; exit 1; }
cmake --build . --target RocirPythonModules.extension._mlirDialectsLLVM.dso -j$(nproc) || { echo "Failed to build _mlirDialectsLLVM.dso"; exit 1; }
cmake --build . --target RocirPythonModules.extension._mlirGPUPasses.dso -j$(nproc) || true
cmake --build . --target RocirPythonModules.extension._mlirExecutionEngine.dso -j$(nproc) || true
cmake --build . --target RocirPythonModules.extension._rocirPasses.dso -j$(nproc) || { echo "Failed to build _rocirPasses.dso"; exit 1; }

# Build rocir-opt tool (used by run_tests.sh MLIR file tests)
cmake --build . --target rocir-opt -j$(nproc) || true

# Copy Python source files and MLIR modules
echo "Copying Python sources..."
# Core python modules (ir.py, passmanager.py, rewrite.py, etc)
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Core.Python -j$(nproc) || { echo "Failed to build MLIR core python sources"; exit 1; }
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Core.Python.Extras -j$(nproc) || true

# Upstream dialect wrappers (mlir.dialects.*) used by tests/utilities
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.builtin -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.builtin.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.arith -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.arith.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.math -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.math.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.memref -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.memref.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.func -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.func.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.cf -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.cf.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.scf -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.scf.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.gpu -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.gpu.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.vector -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.vector.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.llvm -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Dialects.llvm.ops_gen -j$(nproc) || true

# Project dialect python bindings
cmake --build . --target RocirPythonModules.sources.RocirPythonSources.rocir -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.RocirPythonSources.rocir.ops_gen -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.RocirPythonSources.rocdl -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.RocirPythonSources.rocdl.ops_gen -j$(nproc) || true

# Set up PYTHONPATH for the embedded Python package
PYTHON_PACKAGE_DIR="${BUILD_DIR}/python_packages/rocdsl"

# Ensure the python package root contains the embedded MLIR package (_mlir) and our sources (rocdsl, mlir shim).
if [ ! -d "${PYTHON_PACKAGE_DIR}" ]; then
    echo "Error: expected python package root not found: ${PYTHON_PACKAGE_DIR}"
    echo "   (Did the build generate embedded MLIR python modules?)"
    exit 1
fi

# Clean any previously overlaid sources at the root (keep embedded _mlir and include/).
find "${PYTHON_PACKAGE_DIR}" -mindepth 1 -maxdepth 1 \
    ! -name "_mlir" \
    ! -name "include" \
    -exec rm -rf {} +

# Copy RocDSL python package into the package root as rocdsl/
cp -r "${SCRIPT_DIR}/python/rocdsl" "${PYTHON_PACKAGE_DIR}/" || { echo "Failed to copy python/rocdsl"; exit 1; }

# Copy mlir shim package (optional, for backward compatibility with 'import mlir')
if [ -d "${SCRIPT_DIR}/python/mlir" ]; then
    cp -r "${SCRIPT_DIR}/python/mlir" "${PYTHON_PACKAGE_DIR}/" || { echo "Failed to copy python/mlir"; exit 1; }
fi

cd "${SCRIPT_DIR}"

echo ""
echo "✓ Build complete!"
echo "✓ rocir-opt: ${BUILD_DIR}/bin/rocir-opt"
echo "✓ Python bindings built with embedded MLIR dependencies"
echo ""
echo "Python package location: ${PYTHON_PACKAGE_DIR}"
echo ""
echo "To use Python bindings, set:"
echo "  export PYTHONPATH=${PYTHON_PACKAGE_DIR}:\$PYTHONPATH"
