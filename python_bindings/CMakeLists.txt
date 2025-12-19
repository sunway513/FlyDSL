include(AddMLIRPython)

# Use the embedded MLIR package name prefix (`_mlir.*`), matching the install
# layout under build/python_packages/rocdsl/_mlir.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

# Declare Rocir Python sources
declare_mlir_python_sources(RocirPythonSources)

# Declare Rocir dialect Python bindings
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT RocirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/RocirOps.td
  SOURCES
    dialects/rocir.py
  DIALECT_NAME rocir
  GEN_ENUM_BINDINGS
)

#
# Note: Upstream MLIR already provides Python bindings for ROCDL
# (`MLIRPythonSources.Dialects.rocdl`). We include that declared source group
# below instead of re-declaring it here.

# Declare Python extension for pass registration
declare_mlir_python_extension(RocirPythonSources.Passes
  MODULE_NAME _rocirPasses
  ADD_TO_PARENT RocirPythonSources
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    RocirRegisterPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    RocirDialect
    RocirTransforms
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIRocir
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)

# Define the Python sources to include from upstream MLIR
set(RocirMLIRPythonSources
  RocirPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.rocdl
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.gpu
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.ExecutionEngine
)

# Create common CAPI library for Python bindings
add_mlir_python_common_capi_library(RocirPythonCAPI
  INSTALL_COMPONENT RocirPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${RocirMLIRPythonSources}
)

# Set the root prefix for Python modules
set(RocirPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

# Add MLIR Python modules
add_mlir_python_modules(RocirPythonModules
  ROOT_PREFIX "${RocirPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${RocirMLIRPythonSources}"
  COMMON_CAPI_LINK_LIBS
    RocirPythonCAPI
)

option(ROCDSL_SUPPRESS_THIRD_PARTY_WARNINGS
  "Suppress warnings coming from third-party sources used to build the embedded Python bindings (MLIR python binding sources, nanobind)."
  ON
)

if(ROCDSL_SUPPRESS_THIRD_PARTY_WARNINGS AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Suppress noisy third-party warnings when building embedded Python extensions.
  # - nanobind uses some non-ISO constructs that trigger -Wpedantic
  # - MLIR python binding sources may trigger -Waddress / -Wparentheses under GCC
  set(_rocir_python_extension_targets
    RocirPythonModules.extension._mlir.dso
    RocirPythonModules.extension._rocirPasses.dso
    RocirPythonModules.extension._mlirDialectsGPU.dso
    RocirPythonModules.extension._mlirDialectsLLVM.dso
    RocirPythonModules.extension._mlirExecutionEngine.dso
    RocirPythonModules.extension._mlirGPUPasses.dso
  )

  foreach(_t IN LISTS _rocir_python_extension_targets)
    if(TARGET "${_t}")
      target_compile_options("${_t}" PRIVATE
        -Wno-pedantic
        -Wno-address
        -Wno-parentheses
      )
    endif()
  endforeach()

  # nanobind itself triggers -Wpedantic warnings under GCC (zero-size array).
  if(TARGET nanobind-static)
    target_compile_options(nanobind-static PRIVATE -Wno-pedantic)
  endif()
endif()

# Copy the pure Python sources (rocdsl package) to the build directory
add_custom_target(CopyRocirPythonSources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/python/rocdsl"
    "${MLIR_BINARY_DIR}/python_packages/rocdsl/rocdsl"
  COMMENT "Copying python/rocdsl sources to build/python_packages/rocdsl/rocdsl"
  DEPENDS RocirPythonModules
)

add_dependencies(CopyRocirPythonSources RocirPythonModules)
