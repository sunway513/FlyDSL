# Simplified CMakeLists for modern MLIR

set(LLVM_TARGET_DEFINITIONS RocirDialect.td)
mlir_tablegen(RocirDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RocirDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(RocirDialectIncGen)

set(LLVM_TARGET_DEFINITIONS RocirOps.td)
mlir_tablegen(RocirOps.h.inc -gen-op-decls)
mlir_tablegen(RocirOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RocirOpsIncGen)

set(LLVM_TARGET_DEFINITIONS RocirPasses.td)
mlir_tablegen(RocirPasses.h.inc -gen-pass-decls -name Cute)
add_public_tablegen_target(RocirPassesIncGen)
