add_mlir_dialect_library(FlirDialect
  FlirDialect.cpp
  FlirOps.cpp
  FlirLayoutAlgebra.cpp
  FlirTypes.cpp
  
  DEPENDS
  FlirDialectIncGen
  FlirOpsIncGen
  FlirAttrsIncGen
  
  LINK_LIBS
  MLIRIR
)
