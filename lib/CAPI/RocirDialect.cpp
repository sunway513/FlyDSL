#include "rocir-c/RocirDialect.h"
#include "rocir/RocirDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Rocir, rocir, mlir::rocir::RocirDialect)

