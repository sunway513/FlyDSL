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
cmake --build . --target RocirPythonModules.extension._rocirPasses.dso -j$(nproc) || { echo "Failed to build _rocirPasses.dso"; exit 1; }

# Copy Python source files and MLIR modules
echo "Copying Python sources..."
cmake --build . --target RocirPythonModules.sources.MLIRPythonSources.Core -j$(nproc) || true
cmake --build . --target RocirPythonModules.sources.RocirPythonSources -j$(nproc) || true

# Manually copy rocdsl Python package
cp -r "${SCRIPT_DIR}/python/rocdsl" "${BUILD_DIR}/python_packages/" || { echo "Failed to copy Python sources"; exit 1; }

cd "${SCRIPT_DIR}"

# Set up PYTHONPATH for the embedded Python package
PYTHON_PACKAGE_DIR="${BUILD_DIR}/python_packages/rocdsl"

echo ""
echo "✓ Build complete!"
echo "✓ rocir-opt: ${BUILD_DIR}/bin/rocir-opt"
echo "✓ Python bindings built with embedded MLIR dependencies"
echo ""
echo "Python package location: ${PYTHON_PACKAGE_DIR}"
echo ""
echo "To use Python bindings, set:"
echo "  export PYTHONPATH=${PYTHON_PACKAGE_DIR}:\$PYTHONPATH"
