//===- bindings.cpp - pybind11 Bindings for Runtime Utilities -*- C++ -*-===//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../runtime/include/cute_runtime.h"

namespace py = pybind11;
using namespace cute::runtime;

//===----------------------------------------------------------------------===//
// NumPy Array ↔ Device Buffer Interop
//===----------------------------------------------------------------------===//

template<typename T>
py::array_t<T> device_buffer_to_numpy(const DeviceBuffer<T>& buf) {
    auto result = py::array_t<T>(buf.size());
    buf.copy_to_host(result.mutable_data(), buf.size());
    return result;
}

template<typename T>
void numpy_to_device_buffer(DeviceBuffer<T>& buf, py::array_t<T> arr) {
    if (arr.size() != buf.size()) {
        throw std::runtime_error("Array size mismatch");
    }
    buf.copy_from_host(arr.data(), arr.size());
}

//===----------------------------------------------------------------------===//
// Python Module Definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_cute_bindings, m) {
    m.doc() = "Rocir runtime C++ bindings";
    
    //===------------------------------------------------------------------===//
    // Exceptions
    //===------------------------------------------------------------------===//
    
    py::register_exception<CuteRuntimeError>(m, "CuteRuntimeError");
    
    //===------------------------------------------------------------------===//
    // Enums
    //===------------------------------------------------------------------===//
    
    py::enum_<Arch>(m, "Arch")
        .value("SM80", Arch::SM80, "Ampere (Compute Capability 8.0)")
        .value("SM90", Arch::SM90, "Hopper (Compute Capability 9.0)")
        .value("SM100", Arch::SM100, "Blackwell (Compute Capability 10.0)")
        .export_values();
    
    py::enum_<SwizzleMode>(m, "SwizzleMode")
        .value("None", SwizzleMode::None)
        .value("Swizzle32B", SwizzleMode::Swizzle32B)
        .value("Swizzle64B", SwizzleMode::Swizzle64B)
        .value("Swizzle128B", SwizzleMode::Swizzle128B)
        .export_values();
    
    //===------------------------------------------------------------------===//
    // Device Properties
    //===------------------------------------------------------------------===//
    
    py::class_<cudaDeviceProp>(m, "DeviceProperties")
        .def_readonly("name", &cudaDeviceProp::name)
        .def_readonly("major", &cudaDeviceProp::major)
        .def_readonly("minor", &cudaDeviceProp::minor)
        .def_readonly("total_global_mem", &cudaDeviceProp::totalGlobalMem)
        .def_readonly("multi_processor_count", &cudaDeviceProp::multiProcessorCount)
        .def_readonly("max_threads_per_block", &cudaDeviceProp::maxThreadsPerBlock)
        .def_readonly("shared_mem_per_block", &cudaDeviceProp::sharedMemPerBlock);
    
    m.def("get_device_properties", &KernelExecutor::get_device_properties,
          py::arg("device_id") = 0,
          "Get CUDA device properties");
    
    //===------------------------------------------------------------------===//
    // Device Buffer
    //===------------------------------------------------------------------===//
    
    py::class_<DeviceBuffer<float>>(m, "DeviceBufferFloat")
        .def(py::init<size_t>())
        .def("copy_from_host", 
             [](DeviceBuffer<float>& self, py::array_t<float> arr) {
                 self.copy_from_host(arr.data(), arr.size());
             })
        .def("copy_to_host",
             [](const DeviceBuffer<float>& self) {
                 return device_buffer_to_numpy(self);
             })
        .def_property_readonly("size", &DeviceBuffer<float>::size);
    
    py::class_<DeviceBuffer<double>>(m, "DeviceBufferDouble")
        .def(py::init<size_t>())
        .def("copy_from_host",
             [](DeviceBuffer<double>& self, py::array_t<double> arr) {
                 self.copy_from_host(arr.data(), arr.size());
             })
        .def("copy_to_host",
             [](const DeviceBuffer<double>& self) {
                 return device_buffer_to_numpy(self);
             })
        .def_property_readonly("size", &DeviceBuffer<double>::size);
    
    //===------------------------------------------------------------------===//
    // TMA Descriptor
    //===------------------------------------------------------------------===//
    
    py::class_<TMADescriptor>(m, "TMADescriptor")
        .def(py::init<>())
        .def("initialize_2d", &TMADescriptor::initialize_2d,
             py::arg("global_ptr"),
             py::arg("dtype"),
             py::arg("global_dim_x"),
             py::arg("global_dim_y"),
             py::arg("tile_dim_x"),
             py::arg("tile_dim_y"),
             py::arg("swizzle") = SwizzleMode::Swizzle128B,
             "Initialize 2D TMA descriptor");
    
    //===------------------------------------------------------------------===//
    // Launch Configuration
    //===------------------------------------------------------------------===//
    
    py::class_<LaunchConfig>(m, "LaunchConfig")
        .def(py::init<dim3, dim3, size_t>(),
             py::arg("grid_dim"),
             py::arg("block_dim"),
             py::arg("shared_mem_bytes") = 0)
        .def_readwrite("grid_dim", &LaunchConfig::grid_dim)
        .def_readwrite("block_dim", &LaunchConfig::block_dim)
        .def_readwrite("shared_mem_bytes", &LaunchConfig::shared_mem_bytes);
    
    // dim3 helper
    py::class_<dim3>(m, "Dim3")
        .def(py::init<unsigned int, unsigned int, unsigned int>(),
             py::arg("x") = 1, py::arg("y") = 1, py::arg("z") = 1)
        .def_readwrite("x", &dim3::x)
        .def_readwrite("y", &dim3::y)
        .def_readwrite("z", &dim3::z);
    
    //===------------------------------------------------------------------===//
    // Kernel Executor
    //===------------------------------------------------------------------===//
    
    py::class_<KernelExecutor>(m, "KernelExecutor")
        .def(py::init<>())
        .def("load_cubin", &KernelExecutor::load_cubin,
             py::arg("cubin_path"),
             "Load kernel from CUBIN file")
        .def("load_ptx", &KernelExecutor::load_ptx,
             py::arg("ptx_path"),
             "Load kernel from PTX file")
        .def("set_kernel", &KernelExecutor::set_kernel,
             py::arg("kernel_name"),
             "Set kernel function to execute")
        .def("launch", &KernelExecutor::launch,
             py::arg("args"),
             py::arg("config"),
             "Launch kernel with arguments")
        .def("synchronize", &KernelExecutor::synchronize,
             "Wait for kernel completion");
    
    //===------------------------------------------------------------------===//
    // Compiler
    //===------------------------------------------------------------------===//
    
    py::class_<CuteCompiler>(m, "Compiler")
        .def(py::init<>())
        .def("set_mlir_bin_path", &CuteCompiler::set_mlir_bin_path,
             py::arg("path"),
             "Set MLIR tools installation path")
        .def("compile_to_ptx", &CuteCompiler::compile_to_ptx,
             py::arg("mlir_code"),
             py::arg("arch"),
             py::arg("opt_level") = 2,
             "Compile MLIR to PTX assembly")
        .def("compile_to_cubin", &CuteCompiler::compile_to_cubin,
             py::arg("ptx_code"),
             py::arg("arch"),
             "Compile PTX to CUBIN binary")
        .def("compile", &CuteCompiler::compile,
             py::arg("mlir_code"),
             py::arg("arch"),
             py::arg("opt_level") = 2,
             "Full compilation: MLIR → CUBIN");
    
    //===------------------------------------------------------------------===//
    // GEMM Executor
    //===------------------------------------------------------------------===//
    
    py::class_<GemmExecutor<half, half, float>>(m, "GemmExecutorHalfFloat")
        .def(py::init<size_t, size_t, size_t, Arch, bool>(),
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("arch") = Arch::SM90,
             py::arg("use_tma") = true)
        .def("compile_from_mlir", 
             &GemmExecutor<half, half, float>::compile_from_mlir,
             py::arg("mlir_code"))
        .def("load_compiled",
             &GemmExecutor<half, half, float>::load_compiled,
             py::arg("cubin_path"))
        .def("execute",
             [](GemmExecutor<half, half, float>& self,
                py::array_t<uint16_t> A,  // half as uint16
                py::array_t<uint16_t> B,
                py::array_t<float> C,
                bool is_device_ptr) {
                 self.execute(
                     reinterpret_cast<const half*>(A.data()),
                     reinterpret_cast<const half*>(B.data()),
                     C.mutable_data(),
                     is_device_ptr
                 );
                 return C;
             },
             py::arg("A"), py::arg("B"), py::arg("C"),
             py::arg("is_device_ptr") = false)
        .def_static("get_optimal_tile_size",
                    &GemmExecutor<half, half, float>::get_optimal_tile_size,
                    py::arg("M"), py::arg("N"), py::arg("K"),
                    py::arg("arch"));
    
    py::class_<GemmExecutor<float, float, float>>(m, "GemmExecutorFloat")
        .def(py::init<size_t, size_t, size_t, Arch, bool>(),
             py::arg("M"), py::arg("N"), py::arg("K"),
             py::arg("arch") = Arch::SM90,
             py::arg("use_tma") = true)
        .def("compile_from_mlir",
             &GemmExecutor<float, float, float>::compile_from_mlir,
             py::arg("mlir_code"))
        .def("load_compiled",
             &GemmExecutor<float, float, float>::load_compiled,
             py::arg("cubin_path"))
        .def("execute",
             [](GemmExecutor<float, float, float>& self,
                py::array_t<float> A,
                py::array_t<float> B,
                py::array_t<float> C,
                bool is_device_ptr) {
                 self.execute(A.data(), B.data(), C.mutable_data(), is_device_ptr);
                 return C;
             },
             py::arg("A"), py::arg("B"), py::arg("C"),
             py::arg("is_device_ptr") = false)
        .def_static("get_optimal_tile_size",
                    &GemmExecutor<float, float, float>::get_optimal_tile_size,
                    py::arg("M"), py::arg("N"), py::arg("K"),
                    py::arg("arch"));
    
    //===------------------------------------------------------------------===//
    // Generic GEMM factory function
    //===------------------------------------------------------------------===//
    
    m.def("GemmExecutor",
          [](size_t M, size_t N, size_t K,
             py::dtype dtype_a, py::dtype dtype_b, py::dtype dtype_c,
             Arch arch, bool use_tma) -> py::object {
              
              // Dispatch based on dtypes
              if (dtype_a.kind() == 'f' && dtype_a.itemsize() == 2) {
                  // FP16 input
                  return py::cast(new GemmExecutor<half, half, float>(
                      M, N, K, arch, use_tma
                  ));
              } else if (dtype_a.kind() == 'f' && dtype_a.itemsize() == 4) {
                  // FP32 input
                  return py::cast(new GemmExecutor<float, float, float>(
                      M, N, K, arch, use_tma
                  ));
              } else {
                  throw std::runtime_error("Unsupported dtype combination");
              }
          },
          py::arg("M"), py::arg("N"), py::arg("K"),
          py::arg("dtype_a"), py::arg("dtype_b"), py::arg("dtype_c"),
          py::arg("arch") = Arch::SM90,
          py::arg("use_tma") = true,
          "Create GEMM executor for given data types");
}
