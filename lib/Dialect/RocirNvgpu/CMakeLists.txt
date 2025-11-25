add_mlir_dialect_library(RocirNvgpuDialect
  RocirNvgpuDialect.cpp
  
  DEPENDS
  RocirNvgpuDialectIncGen
  RocirNvgpuOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
