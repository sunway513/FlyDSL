add_mlir_public_c_api_library(MLIRCAPIRocir
  RocirDialect.cpp
  LINK_LIBS PUBLIC
  RocirDialect
  RocirTransforms
)

