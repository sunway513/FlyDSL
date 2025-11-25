add_mlir_library(RocirTransforms
  RocirToStandard.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  DEPENDS
  RocirPassesIncGen
  
  LINK_LIBS
  RocirDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRGPUDialect
  MLIRVectorDialect
)
