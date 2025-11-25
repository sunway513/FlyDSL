# Generate Python bindings for CuTe dialect

add_custom_target(RocirPythonModules)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/rocir/RocirOps.td)

# Add include path for TableGen
set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE})
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include)
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include/rocir)

# Generate Python bindings
mlir_tablegen(_rocir_ops_gen.py -gen-python-op-bindings -bind-dialect=rocir)
add_public_tablegen_target(RocirPythonOpsIncGen)
add_dependencies(RocirPythonModules RocirPythonOpsIncGen)

# Generate Python enum bindings if needed
mlir_tablegen(_rocir_enum_gen.py -gen-python-enum-bindings)
add_public_tablegen_target(RocirPythonEnumIncGen)
add_dependencies(RocirPythonModules RocirPythonEnumIncGen)

# Build Python extension module for pass registration
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_NOTFOUND
)

if(NOT pybind11_NOTFOUND)
  find_package(pybind11 CONFIG)
endif()

if(pybind11_FOUND)
  message(STATUS "Found pybind11: ${pybind11_DIR}")
  
  pybind11_add_module(_rocirPassesExt RocirPassesModule.cpp)
  
  target_include_directories(_rocirPassesExt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
  
  target_link_libraries(_rocirPassesExt PRIVATE
    RocirTransforms
    MLIRPass
    MLIRIR
  )
  
  # Install to python_bindings directory
  set_target_properties(_rocirPassesExt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_dependencies(RocirPythonModules _rocirPassesExt)
  
  message(STATUS "CuTe passes Python extension module will be built")
else()
  message(WARNING "pybind11 not found - CuTe passes will not be available in Python")
endif()

  # Build Python extension module for dialect registration
  pybind11_add_module(_rocirDialect RocirDialectModule.cpp)
  
  target_include_directories(_rocirDialect PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
  
  # No additional libraries needed for simple stub module
  # target_link_libraries(_rocirDialect PRIVATE)
  
  # Install to python_bindings directory
  set_target_properties(_rocirDialect PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_dependencies(RocirPythonModules _rocirDialect)
  
  message(STATUS "CuTe dialect Python extension module will be built")

# Fix TableGen-generated relative imports for standalone usage
# TableGen generates "from ._ods_common import ..." which assumes the code is part of mlir.dialects package
# Since we use build/python_bindings directly in PYTHONPATH, we need absolute imports
add_custom_command(
    TARGET RocirPythonOpsIncGen POST_BUILD
    COMMAND sed -i 's/from \._ods_common/from mlir.dialects._ods_common/g' 
        ${CMAKE_CURRENT_BINARY_DIR}/_rocir_ops_gen.py
    COMMENT "Fixing imports in generated Python bindings"
)

# Copy Python wrapper files to build directory at configure time
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rocir.py
    ${CMAKE_CURRENT_BINARY_DIR}/rocir.py
    COPYONLY
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "")
