#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_rocirDialect, m) {
  m.doc() = "CuTe dialect Python bindings";
  
  // For now, this module just exists to satisfy the import
  // The actual dialect is registered via the rocir-opt tool
  // Future: implement proper Python bindings with C API
  
  m.def("available", []() { return true; }, 
        "Check if CuTe dialect bindings are available");
}
