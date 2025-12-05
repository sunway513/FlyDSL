#!/usr/bin/env python3
"""Matrix Transpose Benchmark - GPU kernel with Shared Memory + Vec2 Vectorization

Optimizations:
- Shared memory tiling to improve memory coalescing
- Vec2 vectorization for global memory access (64-bit loads/stores)
- Configurable TILE_SIZE: each thread processes multiple elements
- Bank conflict avoidance through memory padding
"""

import sys
import os
sys.path.insert(0, os.path.join(os.environ.get('MLIR_PATH'), 'tools/mlir/python_packages/mlir_core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../build/python_bindings'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, rocir, arith, scf
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from mlir import ir
from mlir.dialects import memref, vector
import mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes

# Import benchmark utilities from shared tests/utils.py
from utils import BenchmarkResults, perftest, compile_to_hsaco
    
def benchmark_matrix_transpose(TILE_SIZE=8, BLOCK_TILE=32):
    """Benchmark matrix transpose kernel performance
    
    Args:
        TILE_SIZE: Number of elements each thread processes (must be multiple of 2)
        BLOCK_TILE: Tile size processed by a thread block (default 32x32)
    """
    assert TILE_SIZE % 2 == 0, "TILE_SIZE must be multiple of 2 for vec2"
    
    print("\n" + "="*80)
    print("Benchmark: Matrix Transpose (B = A^T) - Shared Memory + Vec2")
    print("Size: 4096×4096 matrix (16M floats, ~64 MB)")
    print("Memory Traffic: 2 × 4096 × 4096 × 4 bytes = ~134 MB per kernel")
    print(f"TILE_SIZE: Each thread processes {TILE_SIZE} elements ({TILE_SIZE//2} vec2)")
    print(f"BLOCK_TILE: {BLOCK_TILE}×{BLOCK_TILE}")
    print("="*80)
    
    M, N = 4096, 4096
    VEC_SIZE = 2  # vec2
    
    # Configuration
    # BLOCK_TILE passed as arg
    PAD = 2          # Shared memory padding to avoid bank conflicts
    
    # Thread block dimensions
    # Each thread handles TILE_SIZE columns, so we need BLOCK_TILE/TILE_SIZE threads in X
    BLOCK_X = BLOCK_TILE // TILE_SIZE
    BLOCK_Y = BLOCK_TILE
    ITERS = TILE_SIZE // VEC_SIZE  # Number of vec2 operations per thread
    
    print(f"Config: Block={BLOCK_X}x{BLOCK_Y}, Iters/thread={ITERS}, Smem={BLOCK_TILE}x{BLOCK_TILE+PAD}")
    
    # Compile kernel
    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    
    @gpu.module("transpose_kernels", ["#rocdl.target<abi = \"500\">"])
    def gpu_mod():
        pass
    
    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()
    
    # Shared memory: 1D layout for vectorized access
    SMEM_SIZE = BLOCK_TILE * (BLOCK_TILE + PAD)
    smem_type = T.memref(SMEM_SIZE, T.f32(), memory_space=gpu.lds_space())
    memref.global_(sym_name="tile_smem", type_=smem_type, alignment=16)
    
    # Use flat 1D memrefs as kernel parameters
    @gpu.func(emit=True)
    def matrixTranspose(A: T.memref(M * N, T.f32()), B: T.memref(N * M, T.f32())):
        smem = memref.get_global(smem_type, "tile_smem")
        
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        tx = gpu.thread_id("x")
        ty = gpu.thread_id("y")
        
        # Constants
        m_c = arith.index(M)._value
        n_c = arith.index(N)._value
        block_tile_c = arith.index(BLOCK_TILE)._value
        smem_stride_c = arith.index(BLOCK_TILE + PAD)._value
        tile_size_c = arith.index(TILE_SIZE)._value
        two_c = arith.index(2)._value
        
        vec2_type = T.vector(VEC_SIZE, T.f32())
        
        # ==================================================================
        # Phase 1: Load from global memory A using vec2
        # ==================================================================
        # Thread (tx, ty) loads from A: row = ty, col_chunk = tx
        
        row_a = (by * block_tile_c + ty)._value
        row_a_valid = (row_a < m_c)._value
        
        for i in range(ITERS):
            i_c = arith.index(i)._value
            # Col index in A: bx*BLOCK_TILE + tx*TILE_SIZE + i*VEC_SIZE
            col_a_offset = (tx * tile_size_c + i_c * two_c)._value
            col_a = (bx * block_tile_c + col_a_offset)._value
            
            col_a_valid = (col_a < n_c)._value
            col_a_end_valid = ((col_a + two_c) <= n_c)._value
            valid_load = (row_a_valid & col_a_valid & col_a_end_valid)._value
            
            with ir.InsertionPoint(scf.IfOp(valid_load.value).then_block):
                # Load vec2 from A
                g_idx = (row_a * n_c + col_a)._value
                vec_val = vector.load(vec2_type, A, 
                                     [g_idx.value if hasattr(g_idx, "value") else g_idx])
                
                # Store to smem[ty][col_a_offset]
                s_idx = (ty * smem_stride_c + col_a_offset)._value
                vector.store(vec_val, smem, [s_idx.value if hasattr(s_idx, "value") else s_idx])
                scf.yield_([])
        
        gpu.barrier()
        
        # ==================================================================
        # Phase 2: Store to global memory B using vec2
        # ==================================================================
        # Re-map threads to ensure coalesced writes to B
        # We want consecutive threads to write consecutive columns of B
        
        threads_per_row_b = BLOCK_TILE // VEC_SIZE
        rows_per_iter_val = (BLOCK_X * BLOCK_Y) // threads_per_row_b
        num_phase2_iters = BLOCK_TILE // rows_per_iter_val
        
        # Flatten thread ID: tid = ty * BLOCK_X + tx
        tid = (ty * arith.index(BLOCK_X)._value + tx)._value
        
        threads_per_row_c = arith.index(threads_per_row_b)._value
        write_row_base = (tid // threads_per_row_c)._value
        write_col = ((tid % threads_per_row_c) * two_c)._value
        
        rows_per_iter_c = arith.index(rows_per_iter_val)._value
        
        for k in range(num_phase2_iters):
            k_c = arith.index(k)._value
            curr_row_local = (write_row_base + k_c * rows_per_iter_c)._value
            
            # Global row index in B: bx*BLOCK_TILE + curr_row_local
            row_b = (bx * block_tile_c + curr_row_local)._value
            
            # Global col index in B: by*BLOCK_TILE + write_col
            col_b = (by * block_tile_c + write_col)._value
            
            row_b_valid = (row_b < n_c)._value
            col_b_valid = (col_b < m_c)._value
            col_b_end_valid = ((col_b + two_c) <= m_c)._value
            valid_store = (row_b_valid & col_b_valid & col_b_end_valid)._value
            
            with ir.InsertionPoint(scf.IfOp(valid_store.value).then_block):
                # Read from smem[write_col][curr_row_local] (transposed)
                s_idx_0 = (write_col * smem_stride_c + curr_row_local)._value
                val_0 = memref.load(smem, [s_idx_0.value if hasattr(s_idx_0, "value") else s_idx_0])
                
                one_c = arith.index(1)._value
                s_idx_1 = ((write_col + one_c) * smem_stride_c + curr_row_local)._value
                val_1 = memref.load(smem, [s_idx_1.value if hasattr(s_idx_1, "value") else s_idx_1])
                
                # Form vec2
                vec_out = vector.from_elements(vec2_type, 
                                              [val_0.value if hasattr(val_0, "value") else val_0,
                                               val_1.value if hasattr(val_1, "value") else val_1])
                
                # Store vec2 to B
                g_idx_b = (row_b * m_c + col_b)._value
                vector.store(vec_out, B, [g_idx_b.value if hasattr(g_idx_b, "value") else g_idx_b])
                scf.yield_([])
    
    ip.__exit__(None, None, None)
    
    hsaco = compile_to_hsaco(ctx.module, kernel_name="matrixTranspose")
    print(f"\nCompiled to HSACO: {len(hsaco)} bytes")
    print(f"Shared memory: {SMEM_SIZE * 4} bytes per block")
    
    # Allocate device memory
    np.random.seed(123)
    a_host_2d = np.random.randn(M, N).astype(np.float32)
    # Flatten to row-major 1D for kernel
    a_host = a_host_2d.flatten('C')  # C order = row-major
    b_host = np.zeros(N * M, dtype=np.float32)
    
    d_a = hip_check(hip.hipMalloc(M * N * 4))
    d_b = hip_check(hip.hipMalloc(M * N * 4))
    
    hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
    
    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matrixTranspose"))
    
    # Grid: each block processes BLOCK_TILE x BLOCK_TILE
    grid_x = (N + BLOCK_TILE - 1) // BLOCK_TILE
    grid_y = (M + BLOCK_TILE - 1) // BLOCK_TILE
    
    print(f"Grid: ({grid_x}, {grid_y}), Block: ({BLOCK_X}, {BLOCK_Y})")
    print(f"Total threads: {grid_x * grid_y * BLOCK_X * BLOCK_Y:,}")
    
    arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b))]
    args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])
    
    # Define kernel launch function
    def launch_kernel():
        hip_check(hip.hipModuleLaunchKernel(
            kernel_func,
            grid_x, grid_y, 1,  # grid dimensions
            BLOCK_X, BLOCK_Y, 1,  # block dimensions
            0,  # shared memory bytes (static allocation via memref.global_)
            None,  # stream
            args,
            None
        ))
        hip_check(hip.hipDeviceSynchronize())
    
    @perftest
    def run_benchmark():
        return {
            "launch": launch_kernel,
            "size": M * N,
            "total_bytes": 2 * M * N * 4,  # Read + Write
        }
    
    # Run benchmark
    results = run_benchmark()
    
    # Verify correctness
    hip_check(hip.hipMemcpy(b_host.ctypes.data, d_b, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))
    # Reshape back to 2D for comparison
    b_result_2d = b_host.reshape(N, M, order='C')  # row-major
    expected_2d = a_host_2d.T
    error = np.max(np.abs(b_result_2d - expected_2d))
    
    print(f"\n  Correctness Check:")
    print(f"  Max error: {error:.2e}")
    
    # Print benchmark results
    print(f"\n{results}")
    
    # Cleanup
    hip_check(hip.hipFree(d_a))
    hip_check(hip.hipFree(d_b))
    hip_check(hip.hipModuleUnload(hip_module))
    
    return error < 1e-5

# Pytest test function
def test_benchmark_matrix_transpose():
    """Pytest wrapper for matrix transpose benchmark."""
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Matrix Transpose with Vec2 + Shared Memory")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    # Test with TILE_SIZE=4, BLOCK_TILE=32 (optimal configuration)
    assert benchmark_matrix_transpose(TILE_SIZE=4, BLOCK_TILE=32), "Matrix transpose benchmark failed correctness check"

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Matrix Transpose Benchmark')
    parser.add_argument('--tile-size', type=int, default=4,
                       help='Elements per thread (default: 4)')
    parser.add_argument('--block-tile', type=int, default=32,
                       help='Block tile size (default: 32)')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("ROCm GPU Benchmark - Matrix Transpose with Vec2 + Shared Memory")
    print(f"GPU: {get_hip_arch()}")
    print("="*80)
    
    result = benchmark_matrix_transpose(TILE_SIZE=args.tile_size, BLOCK_TILE=args.block_tile)
    
    print("\n" + "="*80)
    if result:
        print("✓ BENCHMARK COMPLETED SUCCESSFULLY")
        sys.exit(0)
    else:
        print("⚠️ BENCHMARK FAILED CORRECTNESS CHECK")
        sys.exit(1)

