//===- RocirPassesModule.cpp - CuTe Passes Python Module ------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "rocir/RocirPasses.h"

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

// Manual registration of implemented passes
inline void registerRocirToStandardPassManual() {
  std::cerr << "[RocirPassesExt] Registering rocir-to-standard pass..." << std::endl;
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return ::mlir::rocir::createRocirToStandardPass();
  });
  std::cerr << "[RocirPassesExt] Pass registered!" << std::endl;
}

PYBIND11_MODULE(_rocirPassesExt, m) {
  std::cerr << "[RocirPassesExt] Module initialization started" << std::endl;
  
  m.doc() = "CuTe transformation passes extension module";

  m.def("register_passes", []() {
    registerRocirToStandardPassManual();
  }, "Register all implemented CuTe transformation passes with MLIR");
  
  // Auto-register on module import
  registerRocirToStandardPassManual();
  
  std::cerr << "[RocirPassesExt] Module initialization completed" << std::endl;
}
