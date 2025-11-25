#include "rocir/RocirRocmDialect.h"
#include "rocir/RocirRocmDialect.cpp.inc"

void mlir::rocir::rocm::RocirRocmDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "rocir/RocirRocmOps.cpp.inc"
  >();
}
