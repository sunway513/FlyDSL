#!/usr/bin/env python3
"""MFMA FP8 GEMM Test using Rocir with B preshuffle (layout 16x16, like aiter.shuffle_weight)."""

import sys
import os
import struct
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
import rocdsl.dialects.ext.rocir as rocir
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco
from rocdsl.runtime.fp8_util import to_byte
import numpy as np
from mlir import ir
from mlir.dialects import vector, memref, builtin
from rocdsl.dialects.ext import arith, scf, gpu
from mlir.dialects import arith as _arith_mlir
import mlir.dialects.rocdl as rocdl
import mlir.extras.types as T
from hip import hip
import ctypes

# Clamp to safe range for BOTH E4M3 and BF8 interpretation
FP8_MAX = 240.0 # Conservative

def e4m3fn_to_float(b: int) -> float:
    # Using Bias 8 for E4M3FNUZ (matches rocdsl.runtime.fp8_util.to_byte behavior)
    b = b & 0xFF
    if b == 0x80: return 0.0 # Negative zero in UZ
    sign = -1.0 if (b & 0x80) else 1.0
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 0: return sign * (mant / 8.0) * (2.0 ** -7) # Subnormal for Bias 8
    else: return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 8))

def per_token_fp8_quantize(arr: np.ndarray):
    max_abs = np.max(np.abs(arr), axis=1, keepdims=True).astype(np.float32)
    scale = max_abs / FP8_MAX
    scale[scale == 0] = 1.0
    scaled = arr / scale
    clipped = np.clip(scaled, -FP8_MAX, FP8_MAX)
    quantized = clipped.astype(np.float32)
    return quantized, scale.squeeze()

def shuffle_weight_np(x: np.ndarray, layout=(16, 16)):
    IN, IK = layout
    BK = IK * 2 
    K = 16
    BN = IN
    x_ = x.reshape(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.transpose(0, 1, 3, 4, 2, 5).copy()
    return x_.reshape(x.shape)

def unshuffle_weight_np(x: np.ndarray, layout=(16, 16)):
    IN, IK = layout
    BK = IK * 2 
    K = 16
    BN = IN
    x_ = x.reshape(-1, x.shape[-2] // BN, x.shape[-1] // BK, BK // K, BN, K)
    x_ = x_.transpose(0, 1, 4, 2, 3, 5).copy()
    return x_.reshape(x.shape)

def unwrap(v):
    if isinstance(v, int): return arith.constant(v, index=True).value
    if hasattr(v, "value"): return v.value
    if hasattr(v, "_value"): return v._value
    return v

def test_mfma_fp8_rocir_preshuffle():
    print("="*80)
    print("MFMA FP8 GEMM Test (B preshuffle 16x16 layout)")
    print("="*80)
    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")
    M, N = 1024, 512
    K = 768 
    ctx = RAIIMLIRContextModule()
    try:
        import _rocirPassesExt
        _rocirPassesExt.register_dialect(ctx.module.context)
        print("✓ Registered Rocir dialect")
    except Exception as e:
        print(f"Warning: Could not register Rocir dialect: {e}")
    f8 = ir.Float8E4M3FNType.get()
    f16 = ir.F16Type.get()
    f32 = ir.F32Type.get()
    size_c = M * N
    size_a = M * K
    size_b = N * K
    allocator = SmemAllocator(ctx, arch=gpu_arch)
    lds_a_decl = allocator.allocate_array(f8, 4096)
    @gpu.module("mfma_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'])
    def gpu_mod():
        allocator.finalize()
        @gpu.func(emit=True)
        def kernel(
            arg_c: T.memref(size_c, T.f32()),
            arg_a: T.memref(size_a, f8),
            arg_b: T.memref(size_b, f8)
        ):
            c_m, c_n, c_k = M, N, K
            c128, c32, c16, c8, c4, c2, c64 = 128, 32, 16, 8, 4, 2, 64
            b_bn, b_bk, b_kpack = 16, 32, 2
            b_kblocks = c_k // b_bk 
            if b_kblocks == 0: b_kblocks = 1
            c0_i32 = arith.i32(0)
            identity_map = ir.AffineMap.get_identity(1)
            # Define Layouts using Rocir
            # Layout A: (M, K) with stride (K, 1)
            layout_a = rocir.make_layout((c_m, c_k), stride=(c_k, 1))
            
            # Layout C: (M, N) with stride (N, 1)
            layout_c = rocir.make_layout((c_m, c_n), stride=(c_n, 1))
            
            # Layout B (Preshuffled): (N, K)            
            k_blocks = c_k // 32
            n_blocks = c_n // 16
            s_n_blocks = 512 * k_blocks
            
            # Flat layout for the 5 components: (n_intra, n_blk, k_intra, k_pack, k_blk)
            # We will compute these 5 coordinates manually.
            layout_b = rocir.make_layout(
                (16, n_blocks, 16, 2, k_blocks),
                stride=(16, s_n_blocks, 1, 256, 512)
            )
            
            shape_lds = rocir.make_shape(c32, c128)
            stride_lds = rocir.make_stride(c128, 1)
            layout_lds = rocir.make_layout(shape_lds, stride_lds)
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")
            by = gpu.block_id("y")
            
            base_ptr = allocator.get_base()
            lds_a = lds_a_decl(base_ptr).get()
            vec4_f32 = ir.VectorType.get([4], f32)
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            # Index arithmetic using operator overloading (ArithValue)
            tx_16 = tx * 16
            
            row_a_local = tx_16 / 128
            col_a_local = tx_16 % 128
            
            bx_32 = bx * 32
            row_a_global = bx_32 + row_a_local
            
            # row_b_local and col_b_local are same as A local for this mapping
            row_b_local = row_a_local
            col_b_local = col_a_local
            
            by_32 = by * 32
            row_b_global = by_32 + row_b_local
            
            lds_write_idx = tx_16
            
            vec16_f8 = ir.VectorType.get([16], f8)
            pad_f8 = arith.constant(0.0, type=f8)
            
            wave_id = tx / 64
            lane_id = tx % 64
            
            wave_row = wave_id / 2
            wave_col = wave_id % 2
            
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16
            
            row_a_lds_base = wave_row * 16
            row_a_lds = row_a_lds_base + lane_mod_16
            
            col_offset_base = lane_div_16 * 8
            
            row_b_lds_base = wave_col * 16
            row_b_lds = row_b_lds_base + lane_mod_16
            
            current_acc = acc_init
            for k in range(0, c_k, c128):
                col_a_global_k = col_a_local + k
                coord_a = rocir.make_coord(unwrap(row_a_global), unwrap(col_a_global_k))
                idx_a = rocir.crd2idx(coord_a, layout_a)
                vec_a = vector.TransferReadOp(vec16_f8, arg_a, [unwrap(idx_a)], identity_map, unwrap(pad_f8), in_bounds=[True]).result
                vector.StoreOp(vec_a, lds_a, [unwrap(lds_write_idx)])
                
                gpu.barrier()
                acc = current_acc
                for ki in range(0, 128, 32):
                    col_lds = col_offset_base + ki
                    
                    coord_a_lds = rocir.make_coord(unwrap(row_a_lds), unwrap(col_lds))
                    idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                    
                    vec8_f8 = ir.VectorType.get([8], f8)
                    vec8_i8 = ir.VectorType.get([8], ir.IntegerType.get_signless(8))
                    vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
                    
                    vec_a_load = vector.LoadOp(vec8_f8, lds_a, [unwrap(idx_a_mfma)]).result
                    
                    # Direct Global B Load
                    global_n_mfma = by_32 + row_b_lds
                    global_k_mfma = k + col_lds
                    
                    n_intra = global_n_mfma % 16
                    n_blk = global_n_mfma // 16
                    
                    k_intra = global_k_mfma % 16
                    k_rem = global_k_mfma // 16
                    k_pack = k_rem % 2
                    k_blk = k_rem // 2
                    
                    coord_b_mfma = rocir.make_coord(n_intra, n_blk, k_intra, k_pack, k_blk)
                    idx_b_global = rocir.crd2idx(coord_b_mfma, layout_b)
                    
                    vec_b_load = vector.TransferReadOp(vec8_f8, arg_b, [unwrap(idx_b_global)], identity_map, unwrap(pad_f8), in_bounds=[True]).result
                    
                    a_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_a_load).result
                    b_bytes = _arith_mlir.BitcastOp(vec8_i8, vec_b_load).result
                    
                    a_vec64 = vector.BitCastOp(vec1_i64, a_bytes).result
                    b_vec64 = vector.BitCastOp(vec1_i64, b_bytes).result
                    
                    a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                    b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                    
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        vec4_f32, [unwrap(a_pack), unwrap(b_pack), unwrap(acc), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                    ).result
                gpu.barrier()
                current_acc = acc
                gpu.barrier()
            
            final_acc = current_acc
            
            # Re-calculate needed indices (some variables might need re-computation if they were local to loop or expensive to keep live? No, they are fine)
            # lane_div_16 and lane_rem_16 were calculated before loop
            
            row_wave_base = wave_row * 16
            col_wave_base = wave_col * 16
            
            row_base_g = bx_32 + row_wave_base
            col_base_g = by_32 + col_wave_base
            
            for i in range(4):
                val = vector.ExtractOp(final_acc, [], [i]).result
                
                row_offset_base = lane_div_16 * 4
                row_offset = row_offset_base + i
                
                col_offset = lane_mod_16
                
                row_g = row_base_g + row_offset
                col_g = col_base_g + col_offset
                
                coord_c = rocir.make_coord(unwrap(row_g), unwrap(col_g))
                idx = rocir.crd2idx(coord_c, layout_c)
                memref.StoreOp(unwrap(val), arg_c, [unwrap(idx)])
    
    print("✓ MLIR module constructed via @gpu.func decorator")
    gpu_func_op = None
    for op in ctx.module.body.operations:
        if isinstance(op, ir.OpView) and op.OPERATION_NAME == "gpu.module":
            body_block = op.body.blocks[0] if hasattr(op.body, 'blocks') else op.body
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
    np.random.seed(42)
    a_host = np.random.rand(M, K).astype(np.float32)
    b_host = np.random.rand(K, N).astype(np.float32)
    b_host_T = np.ascontiguousarray(b_host.T)
    a_fp8, scale_a = per_token_fp8_quantize(a_host)
    b_fp8_T, scale_b = per_token_fp8_quantize(b_host_T)
    b_fp8_T_shuffle = shuffle_weight_np(b_fp8_T, layout=(16, 16)).astype(np.float32)
    
    a_bytes = np.array([to_byte(float(x)) for x in a_fp8.flatten()], dtype=np.uint8)
    b_bytes = np.array([to_byte(float(x)) for x in b_fp8_T_shuffle.flatten()], dtype=np.uint8)
    
    
    c_host = np.zeros(size_c, dtype=np.float32)
    d_a = hip_check(hip.hipMalloc(size_a + 4096))
    d_b = hip_check(hip.hipMalloc(size_b + 4096))
    d_c = hip_check(hip.hipMalloc(size_c * 4))
    hip_check(hip.hipMemset(d_a, 0, size_a + 4096))
    hip_check(hip.hipMemset(d_b, 0, size_b + 4096))
    hip_check(hip.hipMemset(d_c, 0, size_c * 4))
    hip_check(hip.hipMemcpy(d_a, a_bytes.ctypes.data, size_a, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_check(hip.hipMemcpy(d_b, b_bytes.ctypes.data, size_b, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"kernel"))
    arg_ptrs = [ctypes.c_void_p(int(d_c)), ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args_array = (ctypes.c_void_p * 3)(*[ctypes.addressof(p) for p in arg_ptrs])
    grid_x = M // 32
    grid_y = N // 32
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, 256, 1, 1, 0, 0, args_array, None))
    hip_check(hip.hipDeviceSynchronize())
    hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, size_c * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    c_host_matrix = c_host.reshape(M, N)

    # Check raw integer GEMM result using Bias 8 decoder
    a_decoded = np.array([e4m3fn_to_float(int(b)) for b in a_bytes], dtype=np.float32).reshape(M, K)
    b_decoded_shuffle = np.array([e4m3fn_to_float(int(b)) for b in b_bytes], dtype=np.float32).reshape(N, K)
    b_decoded = unshuffle_weight_np(b_decoded_shuffle, layout=(16, 16))
    
    expected_int_matrix = np.matmul(a_decoded, b_decoded.T)
    diff_int = np.abs(c_host_matrix - expected_int_matrix)
    max_diff_int = np.nanmax(diff_int)
    print(f"Max Absolute Difference (Integer GEMM): {max_diff_int}")
    if np.allclose(c_host_matrix, expected_int_matrix, atol=1.0):
         print(f"✓ Raw Integer GEMM matches")
    else:
         print(f"✗ Raw Integer GEMM mismatch")

    print("Computing expected result with np.matmul...")
    a_dequant = a_decoded * scale_a[:, None]
    b_dequant = b_decoded * scale_b[:, None]
    expected_matrix = np.matmul(a_dequant, b_dequant.T)
    c_dequant = c_host_matrix * scale_a[:, None] * scale_b[None, :]
    print(f"c_host stats: finite={np.isfinite(c_host_matrix).sum()}/{c_host_matrix.size}, min={np.min(c_host_matrix)}, max={np.max(c_host_matrix)}")
    diff = np.abs(c_dequant - expected_matrix)
    max_diff = np.nanmax(diff)
    print(f"Max Absolute Difference: {max_diff}")
    
    # Check relative error
    rel_error = diff / (np.abs(expected_matrix) + 1e-6)
    max_rel_error = np.nanmax(rel_error)
    print(f"Max Relative Error: {max_rel_error}")

    if np.allclose(c_dequant, expected_matrix, rtol=1e-2, atol=1e-2):
        print(f"✓ Kernel executed correctly (Matches np.matmul)")
    else:
        print(f"✗ Unexpected result")

if __name__ == "__main__":
    test_mfma_fp8_rocir_preshuffle()
