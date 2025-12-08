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
from tests.utils import compile_to_hsaco, perftest
from rocdsl.runtime.fp8_util import to_byte
import numpy as np
from mlir import ir
from mlir.dialects import vector, memref, builtin, llvm
from rocdsl.dialects.ext import arith, scf, gpu, buffer_ops
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
    # Recursive unwrap for nested ArithValue wrappers
    while hasattr(v, "value") or hasattr(v, "_value"):
        if hasattr(v, "_value"):
            v = v._value
        elif hasattr(v, "value"):
            v = v.value
    return v

def test_mfma_fp8_rocir_preshuffle():
    print("="*80)
    print("MFMA FP8 GEMM Test (B preshuffle 16x16 layout)")
    print("="*80)
    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")
    M, N = 32, 4096
    K = 4096
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
            
            # Create buffer resources
            a_rsrc = buffer_ops.create_buffer_resource(arg_a)
            b_rsrc = buffer_ops.create_buffer_resource(arg_b)
            c_rsrc = buffer_ops.create_buffer_resource(arg_c)
            i32_type = ir.IntegerType.get_signless(32)
            index_type = ir.IndexType.get()
            vec4_f32 = ir.VectorType.get([4], f32)
            vec8_f8 = ir.VectorType.get([8], f8)
            vec1_i64 = ir.VectorType.get([1], ir.IntegerType.get_signless(64))
            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result
            
            # Index arithmetic using operator overloading (ArithValue)
            tx_16 = tx * 16
            
            # Map threads to 16 rows (repeat every 128 threads) and 128 cols span
            row_a_local = (tx % 128) / 8  # 0..15 repeated
            col_a_local = tx_16 % 128
            
            bx_16 = bx * 16
            row_a_global = bx_16 + row_a_local
            
            by_128 = by * 128
            
            coord_store = rocir.make_coord(unwrap(row_a_local), unwrap(col_a_local))
            lds_write_idx = rocir.crd2idx(coord_store, layout_lds)
            
            vec16_f8 = ir.VectorType.get([16], f8)
            
            wave_id = tx / 64
            lane_id = tx % 64
            
            wave_row = (tx - tx)  # force 0
            wave_col = wave_id % 2
            
            lane_mod_16 = lane_id % 16
            lane_div_16 = lane_id / 16
            
            row_a_lds_base = wave_row * 0
            row_a_lds = row_a_lds_base + lane_mod_16
            
            col_offset_base = lane_div_16 * 8
            
            row_b_lds_base = wave_col * 16
            row_b_lds = row_b_lds_base + lane_mod_16
            
            # Process N tile in 32-column chunks to cover 16x128 per block
            for n_tile in range(0, 128, 32):
                current_acc = acc_init
                for k in range(0, c_k, 128):
                    col_a_global_k = col_a_local + k
                    coord_a = rocir.make_coord(unwrap(row_a_global), unwrap(col_a_global_k))
                    idx_a = rocir.crd2idx(coord_a, layout_a)
                    
                    # Global A load: keep 128-bit for efficiency then spill to LDS
                    idx_a_div4 = idx_a // 4
                    v_i32 = buffer_ops.buffer_load(a_rsrc, idx_a_div4, vec_width=4, dtype=i32_type)
                    vec_a = vector.BitCastOp(vec16_f8, v_i32).result
                    vector.StoreOp(vec_a, lds_a, [unwrap(lds_write_idx)])
                    
                    gpu.barrier()
                    acc = current_acc
                    for ki in range(0, 128, 32):
                        col_lds = col_offset_base + ki
                        
                        coord_a_lds = rocir.make_coord(unwrap(row_a_lds), unwrap(col_lds))
                        idx_a_mfma = rocir.crd2idx(coord_a_lds, layout_lds)
                        idx_a_idx = _arith_mlir.IndexCastOp(index_type, unwrap(idx_a_mfma)).result
                        
                        # LDS load: use 64-bit (vector<8xf8>) to avoid complex alignment logic
                        loaded_a = vector.LoadOp(vec8_f8, lds_a, [unwrap(idx_a_idx)]).result
                        
                        # Global B load: use buffer_load dwordx2 (64-bit)
                        global_n_mfma = by_128 + n_tile + row_b_lds
                        global_k_mfma = k + col_lds
                        
                        n_intra = global_n_mfma % 16
                        n_blk = global_n_mfma // 16
                        k_intra = global_k_mfma % 16
                        k_rem = global_k_mfma // 16
                        k_pack = k_rem % 2
                        k_blk = k_rem // 2
                        
                        coord_b_mfma = rocir.make_coord(n_intra, n_blk, k_intra, k_pack, k_blk)
                        idx_b_global = rocir.crd2idx(coord_b_mfma, layout_b)
                        
                        idx_b_div4 = idx_b_global // 4
                        loaded_b = buffer_ops.buffer_load(b_rsrc, idx_b_div4, vec_width=2, dtype=i32_type)
                        
                        a_vec64 = vector.BitCastOp(vec1_i64, loaded_a).result
                        b_vec64 = vector.BitCastOp(vec1_i64, loaded_b).result
                        
                        a_pack = vector.ExtractOp(a_vec64, static_position=[0], dynamic_position=[]).result
                        b_pack = vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result
                        
                        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            vec4_f32, [unwrap(a_pack), unwrap(b_pack), unwrap(acc), unwrap(c0_i32), unwrap(c0_i32), unwrap(c0_i32)]
                        ).result
                    
                    gpu.barrier()
                    current_acc = acc
                
                final_acc = current_acc
                
                row_wave_base = wave_row * 16
                col_wave_base = wave_col * 16
                
                row_base_g = bx_16 + row_wave_base
                col_base_g = by_128 + n_tile + col_wave_base
                
                for i in range(4):
                    val = vector.ExtractOp(final_acc, [], [i]).result
                    
                    row_offset_base = lane_div_16 * 4
                    row_offset = row_offset_base + i
                    
                    col_offset = lane_mod_16
                    
                    row_g = row_base_g + row_offset
                    col_g = col_base_g + col_offset
                    
                    coord_c = rocir.make_coord(unwrap(row_g), unwrap(col_g))
                    idx = rocir.crd2idx(coord_c, layout_c)
                    buffer_ops.buffer_store(val, c_rsrc, idx)
    
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
    grid_x = M // 16
    grid_y = N // 128

    def launch_kernel():
        hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, 256, 1, 1, 0, 0, args_array, None))

    launch_kernel()
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

    # Benchmark using shared perftest helper
    warmup = int(os.environ.get("BENCH_WARMUP", "5"))
    runs = int(os.environ.get("BENCH_RUNS", "20"))
    bytes_moved = size_a + size_b + size_c * 4
    flops = 2 * M * N * K

    @perftest
    def bench():
        return {
            "launch": launch_kernel,
            "size": size_c,  # elements of C; bandwidth uses total_bytes override
            "warmup_iters": warmup,
            "bench_iters": runs,
            "total_bytes": bytes_moved,
        }

    results = bench()
    gflops_avg = flops / (results.avg_ms / 1e3) / 1e9
    gflops_min = flops / (results.min_ms / 1e3) / 1e9

    print(f"Latency ms: avg={results.avg_ms:.3f}, min={results.min_ms:.3f}, max={results.max_ms:.3f}, std={results.std_ms:.3f}")
    print(f"Throughput: avg={gflops_avg:.2f} GFLOPS (min-based {gflops_min:.2f}), bandwidth={results.bandwidth_gbs:.2f} GB/s (bytes: {bytes_moved})")

if __name__ == "__main__":
    test_mfma_fp8_rocir_preshuffle()
