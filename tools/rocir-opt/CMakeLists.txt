set(LIBS
  RocirDialect
  RocirTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRMlirOptMain
)

add_llvm_tool(rocir-opt
  rocir-opt.cpp
)

target_link_libraries(rocir-opt PRIVATE ${LIBS})
llvm_map_components_to_libnames(llvm_libs Support)
target_link_libraries(rocir-opt PRIVATE ${llvm_libs})
