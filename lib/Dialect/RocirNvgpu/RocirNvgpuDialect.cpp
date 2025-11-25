#include "rocir/RocirNvgpuDialect.h"
#include "rocir/RocirNvgpuDialect.cpp.inc"

void mlir::rocir::nvgpu::RocirNvgpuDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "rocir/RocirNvgpuOps.cpp.inc"
  >();
}
