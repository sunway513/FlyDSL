#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using @gpu.func decorator pattern."""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.runtime.fp8_util import to_byte
from utils import compile_to_hsaco
import numpy as np
from mlir import ir
from mlir.dialects import vector, memref, builtin
from rocdsl.dialects.ext import arith, scf, gpu
from mlir.dialects import arith as _arith_mlir
import mlir.dialects.rocdl as rocdl
import mlir.extras.types as T
from hip import hip
import ctypes
from rocdsl.utils import SmemAllocator

def unwrap(v):
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    return v


def test_mfma_fp8_rocir():
    print("="*80)
    print("MFMA FP8 GEMM Test (@gpu.func Decorator) - 1024x1024x1280")
    print("="*80)
    
    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")

    # Constants
    M, N, K = 1024, 1024, 1280
    
    ctx = RAIIMLIRContextModule()
    
    f8 = ir.Float8E4M3FNType.get()
    f32 = ir.F32Type.get()
    
    size_c = M * N
    size_a = M * K
    size_b = N * K  # Transposed B (NxK)
    
    # 1. Initialize Allocator
    allocator = SmemAllocator(ctx, arch=gpu_arch)

    # 2. Allocate Arrays (Tile size 32x128)
    lds_a_decl = allocator.allocate_array(f8, 4096)
    lds_b_decl = allocator.allocate_array(f8, 4096)
    
    @gpu.module("mfma_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">' ])
    def gpu_mod():
        # 3. Finalize: Automatically create underlying memref.GlobalOp
        allocator.finalize()
        
        @gpu.func(emit=True)
        def kernel(
            arg_c: T.memref(size_c, T.f32()),
            arg_a: T.memref(size_a, f8),
            arg_b: T.memref(size_b, f8)
        ):
            c0_i32 = _arith_mlir.ConstantOp(ir.IntegerType.get_signless(32), ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)).result
            c128 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 128)).result
            c32 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 32)).result
            c16 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 16)).result
            c8 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 8)).result
            c4 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 4)).result
            c2 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 2)).result
            c64 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 64)).result
            c1024 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1024)).result
            c1280 = _arith_mlir.ConstantOp(ir.IndexType.get(), ir.IntegerAttr.get(ir.IndexType.get(), 1280)).result
            
            identity_map = ir.AffineMap.get_identity(1)
            
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            # 4. Get Shared Memory Pointers
            base_ptr = allocator.get_base()
            lds_a = lds_a_decl(base_ptr).get()
            lds_b = lds_b_decl(base_ptr).get()
            
            # Accumulator Init
            vec4_f32 = ir.VectorType.get([4], f32)
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, unwrap(zero_attr)).result
            
            # Global Load Indices
            
            tx_16 = _arith_mlir.MulIOp(unwrap(tx), unwrap(c16)).result
            
            row_a_local = _arith_mlir.DivUIOp(unwrap(tx_16), unwrap(c128)).result
            col_a_local = _arith_mlir.RemUIOp(unwrap(tx_16), unwrap(c128)).result
            
            bx_32 = _arith_mlir.MulIOp(unwrap(bx._value), unwrap(c32)).result
            row_a_global = _arith_mlir.AddIOp(unwrap(bx_32), unwrap(row_a_local)).result
            
            # For B (Transposed)
            row_b_local = _arith_mlir.DivUIOp(unwrap(tx_16), unwrap(c128)).result
            col_b_local = _arith_mlir.RemUIOp(unwrap(tx_16), unwrap(c128)).result
            
            by_32 = _arith_mlir.MulIOp(unwrap(by._value), unwrap(c32)).result
            row_b_global = _arith_mlir.AddIOp(unwrap(by_32), unwrap(row_b_local)).result
            
            # LDS Write Index
            lds_write_idx = tx_16
            
            vec16_f8 = ir.VectorType.get([16], f8)
            pad_f8 = _arith_mlir.ConstantOp(f8, ir.FloatAttr.get(f8, 0.0)).result
            
            # Pre-calculate LDS read indices
            wave_id = _arith_mlir.DivUIOp(unwrap(tx._value), unwrap(c64)).result
            lane_id = _arith_mlir.RemUIOp(unwrap(tx._value), unwrap(c64)).result
            
            wave_row = _arith_mlir.DivUIOp(unwrap(wave_id), unwrap(c2)).result
            wave_col = _arith_mlir.RemUIOp(unwrap(wave_id), unwrap(c2)).result
            
            lane_mod_16 = _arith_mlir.RemUIOp(unwrap(lane_id), unwrap(c16)).result
            lane_div_16 = _arith_mlir.DivUIOp(unwrap(lane_id), unwrap(c16)).result
            
            row_a_lds_base = _arith_mlir.MulIOp(unwrap(wave_row), unwrap(c16)).result
            row_a_lds = _arith_mlir.AddIOp(unwrap(row_a_lds_base), unwrap(lane_mod_16)).result
            
            col_offset_base = _arith_mlir.MulIOp(unwrap(lane_div_16), unwrap(c8)).result
            
            row_b_lds_base = _arith_mlir.MulIOp(unwrap(wave_col), unwrap(c16)).result
            row_b_lds = _arith_mlir.AddIOp(unwrap(row_b_lds_base), unwrap(lane_mod_16)).result
            
            # Main Loop K
            current_acc = acc_init
            for k in range(0, 1280, 128):
                k_const = _arith_mlir.ConstantOp(ir.IndexType.get(), k).result
                
                # Load A
                col_a_global_k = _arith_mlir.AddIOp(unwrap(k_const), unwrap(col_a_local)).result
                row_a_g_1280 = _arith_mlir.MulIOp(unwrap(row_a_global), unwrap(c1280)).result
                idx_a = _arith_mlir.AddIOp(unwrap(row_a_g_1280), unwrap(col_a_global_k)).result
                
                vec_a = vector.TransferReadOp(vec16_f8, arg_a, [unwrap(idx_a)], identity_map, unwrap(pad_f8), [True]).result
                vector.StoreOp(unwrap(vec_a), lds_a, [unwrap(lds_write_idx)])
                
                # Load B (Transposed)
                col_b_global_k = _arith_mlir.AddIOp(unwrap(k_const), unwrap(col_b_local)).result
                row_b_g_1280 = _arith_mlir.MulIOp(unwrap(row_b_global), unwrap(c1280)).result
                idx_b = _arith_mlir.AddIOp(unwrap(row_b_g_1280), unwrap(col_b_global_k)).result
                
                vec_b = vector.TransferReadOp(vec16_f8, arg_b, [unwrap(idx_b)], identity_map, unwrap(pad_f8), [True]).result
                vector.StoreOp(unwrap(vec_b), lds_b, [unwrap(lds_write_idx)])
                
                gpu.barrier()
                
                # Inner Loop
                acc = current_acc
                for ki in range(0, 128, 32):
                    ki_const = _arith_mlir.ConstantOp(ir.IndexType.get(), ki).result
                    col_lds = _arith_mlir.AddIOp(unwrap(ki_const), unwrap(col_offset_base)).result
                    
                    # A LDS Index
                    row_a_lds_128 = _arith_mlir.MulIOp(unwrap(row_a_lds), unwrap(c128)).result
                    idx_a_mfma = _arith_mlir.AddIOp(unwrap(row_a_lds_128), unwrap(col_lds)).result
                    
                    # B LDS Index
                    row_b_lds_128 = _arith_mlir.MulIOp(unwrap(row_b_lds), unwrap(c128)).result
                    idx_b_mfma = _arith_mlir.AddIOp(unwrap(row_b_lds_128), unwrap(col_lds)).result
                    
                    vec8_f8 = ir.VectorType.get([8], f8)
                    vec8_i8 = ir.VectorType.get([8], ir.IntegerType.get_signless(8))
                    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                    
                    vec_a_load = vector.LoadOp(vec8_f8, lds_a, [unwrap(idx_a_mfma)]).result
                    vec_b_load = vector.LoadOp(vec8_f8, lds_b, [unwrap(idx_b_mfma)]).result
                    
                    a_bytes = _arith_mlir.BitcastOp(unwrap(vec8_i8), unwrap(vec_a_load)).result
                    b_bytes = _arith_mlir.BitcastOp(unwrap(vec8_i8), unwrap(vec_b_load)).result
                    
                    a_vec64 = vector.BitCastOp(vec1_i64, a_bytes).result
                    b_vec64 = vector.BitCastOp(vec1_i64, b_bytes).result
                    
                    a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                    b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                    
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        vec4_f32, [unwrap(a_pack), unwrap(b_pack), unwrap(acc), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                    ).result
                    
                gpu.barrier()
                current_acc = acc

            final_acc = current_acc
            
            # Store Result
            lane_div_16 = _arith_mlir.DivUIOp(unwrap(lane_id), unwrap(c16)).result
            lane_rem_16 = _arith_mlir.RemUIOp(unwrap(lane_id), unwrap(c16)).result
            
            row_wave_base = _arith_mlir.MulIOp(unwrap(wave_row), unwrap(c16)).result
            col_wave_base = _arith_mlir.MulIOp(unwrap(wave_col), unwrap(c16)).result
            
            bx_32 = _arith_mlir.MulIOp(unwrap(bx._value), unwrap(c32)).result
            by_32 = _arith_mlir.MulIOp(unwrap(by._value), unwrap(c32)).result
            
            row_base_g = _arith_mlir.AddIOp(unwrap(bx_32), unwrap(row_wave_base)).result
            col_base_g = _arith_mlir.AddIOp(unwrap(by_32), unwrap(col_wave_base)).result
            
            for i in range(4):
                val = vector.ExtractOp(final_acc, [], [unwrap(i)]).result
                
                c_i = _arith_mlir.ConstantOp(ir.IndexType.get(), i).result
                row_offset_base = _arith_mlir.MulIOp(unwrap(lane_div_16), unwrap(c4)).result
                row_offset = _arith_mlir.AddIOp(unwrap(row_offset_base), unwrap(c_i)).result
                
                col_offset = lane_rem_16
                
                row_g = _arith_mlir.AddIOp(unwrap(row_base_g), unwrap(row_offset)).result
                col_g = _arith_mlir.AddIOp(unwrap(col_base_g), unwrap(col_offset)).result
                
                row_g_1024 = _arith_mlir.MulIOp(unwrap(row_g), unwrap(c1024)).result
                idx = _arith_mlir.AddIOp(unwrap(row_g_1024), unwrap(col_g)).result
                
                memref.StoreOp(unwrap(val), arg_c, [unwrap(idx)])
    
    print("✓ MLIR module constructed via @gpu.func decorator")
    
    # Set kernel attributes
    gpu_func_op = None
    for op in ctx.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            body_block = op.body.blocks[0] if hasattr(op.body, "blocks") else op.body
            for inner_op in body_block.operations:
                if hasattr(inner_op, 'OPERATION_NAME') and inner_op.OPERATION_NAME == "gpu.func":
                    gpu_func_op = inner_op
                    break
    
    if gpu_func_op:
        gpu_func_op.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get("256,256")
        gpu_func_op.attributes["rocdl.reqd_work_group_size"] = ir.DenseI32ArrayAttr.get([256, 1, 1])
        gpu_func_op.attributes["gpu.kernel"] = ir.UnitAttr.get()
    
    print("Compiling...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f"✓ Compiled to HSACO: {len(hsaco)} bytes")
    
    print("Executing kernel...")
    
    # Random inputs
    a_host = np.random.randint(-16, 16, size=(M, K)).astype(np.float32)
    b_host = np.random.randint(-16, 16, size=(K, N)).astype(np.float32)
    
    # Transpose B for the kernel (NxK)
    b_host_T = np.ascontiguousarray(b_host.T)
    
    a_bytes = np.array([to_byte(x) for x in a_host.flatten()], dtype=np.uint8)
    b_bytes = np.array([to_byte(x) for x in b_host_T.flatten()], dtype=np.uint8)
    
    c_host = np.zeros(size_c, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(size_a))
    d_b = hip_check(hip.hipMalloc(size_b))
    d_c = hip_check(hip.hipMalloc(size_c * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_bytes.ctypes.data, size_a, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_bytes.ctypes.data, size_b, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    
    arg_ptrs = [ctypes.c_void_p(int(d_c)), ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args_array = (ctypes.c_void_p * 3)(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Grid: 32x32 blocks. Block: 256 threads.
    hip_check(hip.hipModuleLaunchKernel(kernel_func, 32, 32, 1, 256, 1, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, size_c * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    
    # Verification
    print("Computing expected result with np.matmul...")
    expected_matrix = np.matmul(a_host, b_host)
    expected = expected_matrix.flatten()
    
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipFree(d_c))
    hip_check(hip.hipModuleUnload(hip_module))
    print("="*80)
    print(f"Max Absolute Difference: {np.max(np.abs(c_host - expected))}")
    
    if np.allclose(c_host, expected, atol=1.0):
        print(f"✓ Kernel executed correctly (Matches np.matmul)")
        return True
    else:
        print(f"✗ Unexpected result")
        print(f"  Min: {np.min(c_host)}")
        print(f"  Max: {np.max(c_host)}")
        failures = np.where(np.abs(c_host - expected) > 1.0)[0]
        if len(failures) > 0:
            print(f"  First failure at index {failures[0]}: Expected {expected[failures[0]]}, Got {c_host[failures[0]]}")
            print(f"  Total failures: {len(failures)}")
        raise ValueError("Kernel result does not match expected values")
    

if __name__ == "__main__":
    test_mfma_fp8_rocir()
