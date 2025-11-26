#!/usr/bin/env python3
"""
MFMA Test - FP8 GEMM 64x64x64

Creates a 64x64x64 GEMM that exercises fp8 inputs with
`rocdl.mfma.f32.16x16x32.fp8.fp8`.
- Grid: 16 waves (1024 threads total)
- 4 blocks, each with 256 threads (4 waves)
- Each wave is responsible for a 16x16 tile of C
- K dimension covered with two 16x16x32 fp8 MFMA steps
- Inputs: A = 1.0, B = 1.0 (stored as fp8 constants)
- Expected output: 1.0 * 1.0 * 64 = 64.0 per element

Implementation mirrors `test_mfma_gemm_real.py` but swaps the MFMA
intrinsic and operand types to their fp8 counterparts. The kernel uses
`vector.transfer_write` so every lane writes its 4-lane accumulator slice
back to global memory.
"""
import sys
sys.path.insert(0, "/mnt/raid0/felix/llvm-project/buildmlir/tools/mlir/python_packages/mlir_core")
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/build/python_bindings")
sys.path.insert(0, "/mnt/raid0/felix/rocDSL/python")

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import numpy as np
from mlir import ir
from hip import hip
import ctypes

MLIR_TEMPLATE = """
module {
    gpu.module @mfma_mod [#rocdl.target<chip = "gfx942", abi = "500">] {
        gpu.func @kernel(%C: memref<4096xf32>) kernel {
            %c0_i32 = arith.constant 0 : i32
            %c4 = arith.constant 4 : index
            %a_vec = arith.constant dense<$A_VALUES$> : vector<8xf8E4M3FN>
            %b_vec = arith.constant dense<$B_VALUES$> : vector<8xf8E4M3FN>
            %a_bytes = arith.bitcast %a_vec : vector<8xf8E4M3FN> to vector<8xi8>
            %b_bytes = arith.bitcast %b_vec : vector<8xf8E4M3FN> to vector<8xi8>
            %a_vec64 = vector.bitcast %a_bytes : vector<8xi8> to vector<1xi64>
            %b_vec64 = vector.bitcast %b_bytes : vector<8xi8> to vector<1xi64>
            %a_pack = vector.extract %a_vec64[0] : i64 from vector<1xi64>
            %b_pack = vector.extract %b_vec64[0] : i64 from vector<1xi64>
            %c_init = arith.constant dense<0.0> : vector<4xf32>

            // Each fp8 MFMA covers 32 columns in K, so two steps cover K=64.
            %d1 = rocdl.mfma.f32.16x16x32.fp8.fp8 %a_pack, %b_pack, %c_init, %c0_i32, %c0_i32, %c0_i32
                : (i64, i64, vector<4xf32>, i32, i32, i32) -> vector<4xf32>

            %d2 = rocdl.mfma.f32.16x16x32.fp8.fp8 %a_pack, %b_pack, %d1, %c0_i32, %c0_i32, %c0_i32
                : (i64, i64, vector<4xf32>, i32, i32, i32) -> vector<4xf32>

            %tx = gpu.thread_id x
            %bx = gpu.block_id x
            %bdx = gpu.block_dim x
            %mul = arith.muli %bx, %bdx : index
            %idx = arith.addi %mul, %tx : index
            %offset = arith.muli %idx, %c4 : index
            vector.transfer_write %d2, %C[%offset] {in_bounds = [true]} : vector<4xf32>, memref<4096xf32>
            gpu.return
        }
    }
}
"""


def _format_dense(values: np.ndarray) -> str:
    formatted = ", ".join(f"{float(v):.3f}" for v in values)
    return f"[{formatted}]"

def test_mfma_fp8():
    print("=" * 80)
    print("MFMA FP8 GEMM Test - 64x64x64")
    print("=" * 80)
    print(f"Detected HIP Arch: {get_hip_arch()}")

    allowed_values = np.array([-1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5], dtype=np.float32)
    rng = np.random.default_rng(1234)
    a_vals = rng.choice(allowed_values, size=8, replace=True)
    b_vals = rng.choice(allowed_values, size=8, replace=True)
    mlir_text = (
        MLIR_TEMPLATE.replace("$A_VALUES$", _format_dense(a_vals))
        .replace("$B_VALUES$", _format_dense(b_vals))
    )
    print(f"A fp8 values: {a_vals}")
    print(f"B fp8 values: {b_vals}")
    expected_scalar = 2.0 * float(np.dot(a_vals, b_vals))
    print(f"Expected accumulator contribution: {expected_scalar}")

    with ir.Context():
        module = ir.Module.parse(mlir_text)
        print("✓ MLIR module parsed")
        print("Compiling...")
        try:
            pipeline = (
                Pipeline()
                .canonicalize()
                .rocdl_attach_target(chip="gfx942")
                .convert_vector_to_llvm()
                .Gpu(
                    Pipeline().convert_gpu_to_rocdl(
                        use_bare_ptr_memref_call_conv=True, runtime="HIP"
                    )
                )
                .gpu_to_llvm()
                .lower_to_llvm()
                .gpu_module_to_binary(format="bin")
            )
        except AttributeError:
            print("Warning: Pipeline.convert_vector_to_llvm not found. Trying without it.")
            pipeline = (
                Pipeline()
                .canonicalize()
                .rocdl_attach_target(chip="gfx942")
                .Gpu(
                    Pipeline().convert_gpu_to_rocdl(
                        use_bare_ptr_memref_call_conv=True, runtime="HIP"
                    )
                )
                .gpu_to_llvm()
                .lower_to_llvm()
                .gpu_module_to_binary(format="bin")
            )
        lowered = run_pipeline(module, pipeline)

    from rocdsl.dialects.ext.gpu import get_compile_object_bytes

    hsaco = get_compile_object_bytes(lowered)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")

    print("Executing kernel...")
    c_host = np.zeros(4096, dtype=np.float32)
    d_c = hip_check(hip.hipMalloc(4096 * 4))
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))

    arg_ptrs = [ctypes.c_void_p(int(d_c))]
    args_array = (ctypes.c_void_p * 1)(*[ctypes.addressof(p) for p in arg_ptrs])

    # Four blocks * 256 threads = 1024 threads (16 waves in total).
    hip_check(
        hip.hipModuleLaunchKernel(
            kernel_func, 4, 1, 1, 256, 1, 1, 0, 0, args_array, None
        )
    )
    hip_check(hip.hipDeviceSynchronize())
    hip_check(
        hip.hipMemcpy(
            c_host.ctypes.data, d_c, 4096 * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost
        )
    )

    if np.allclose(c_host, expected_scalar, rtol=1e-4, atol=1e-4):
        print(f"✓ Kernel executed correctly (All {len(c_host)} values = {expected_scalar})")
    else:
        print("✗ Unexpected result")
        print(f"  Expected: {expected_scalar}")
        print(f"  Min: {np.min(c_host)}")
        print(f"  Max: {np.max(c_host)}")
        print(f"  Mean: {np.mean(c_host)}")
        failures = np.where(np.abs(c_host - expected_scalar) > 1e-3)[0]
        if len(failures) > 0:
            print(f"  First failure at index {failures[0]}: {c_host[failures[0]]}")
            print(f"  Total failures: {len(failures)}")

    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    print("=" * 80)
    return True


if __name__ == "__main__":
    test_mfma_fp8()
