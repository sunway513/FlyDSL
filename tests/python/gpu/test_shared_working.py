#!/usr/bin/env python3
"""Shared memory matmul - FIXED VERSION"""

import sys
import os


from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import gpu, scf, rocir
from rocdsl.dialects.ext.gpu import lds_space
from rocdsl.dialects.ext import arith
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from tests.utils import compile_to_hsaco
from _mlir import ir
from _mlir.dialects import memref
import _mlir.extras.types as T
from hip import hip
import numpy as np
import ctypes
import time
import gc

M, N, K = 256, 256, 256
TILE_SIZE = 16

ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
gpu.set_container_module(ctx.module)

# Initialize Allocator
allocator = SmemAllocator(ctx, arch=get_hip_arch())

# We need to allocate enough space for TILE_SIZE x TILE_SIZE float32 elements
# SmemAllocator's allocate_array gives us a 1D array view.
s_a_decl = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)
s_b_decl = allocator.allocate_array(T.f32(), TILE_SIZE * TILE_SIZE)

@gpu.module("matmul_shared", [f'#rocdl.target<chip = "{get_hip_arch()}", abi = "500">'])
def gpu_mod():
    allocator.finalize()

ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
ip.__enter__()

@gpu.func(emit=True)
def matmul_shared(A: T.memref(M, K, T.f32()), B: T.memref(K, N, T.f32()), C: T.memref(M, N, T.f32())):
    # Get references to shared memory using Allocator
    base_ptr = allocator.get_base()
    As = s_a_decl(base_ptr)
    Bs = s_b_decl(base_ptr)
    
    tile_c = arith.index(TILE_SIZE)
    row = (gpu.block_id("y") * tile_c + gpu.thread_id("y"))
    col = (gpu.block_id("x") * tile_c + gpu.thread_id("x"))
    
    tx = gpu.thread_id("x")
    ty = gpu.thread_id("y")
    
    zero = arith.index(0)
    one = arith.index(1)
    k_c = arith.index(K)
    zero_f = arith.f32(0.0)
    
    acc = zero_f
    num_tiles = arith.index(K // TILE_SIZE)
    
    # Helper for 2D to 1D mapping: idx = y * WIDTH + x
    tile_width = arith.index(TILE_SIZE)
    
    # Rocir Layout definition
    tile_size_idx = tile_c
    one_idx = one
    
    # Shape: (TILE_SIZE, TILE_SIZE)
    tile_shape = rocir.make_shape(tile_size_idx, tile_size_idx)
    # Stride: (TILE_SIZE, 1) -> Row Major
    tile_stride = rocir.make_stride(tile_size_idx, one_idx)
    tile_layout = rocir.make_layout(tile_shape, tile_stride)

    def get_tile_idx(y, x):
        coord = rocir.make_coord(y, x)
        idx_val = rocir.crd2idx(coord, tile_layout)
        return idx_val.value if hasattr(idx_val, 'value') else idx_val

    for_tiles = scf.ForOp(zero.value, num_tiles.value, one.value, [acc.value])
    with ir.InsertionPoint(for_tiles.body):
        t = for_tiles.induction_variable
        acc_val = for_tiles.inner_iter_args[0]
        k_base = (t * tile_c)
        
        a_col = (k_base + tx)
        a_val = memref.load(A, [row.value, a_col.value])
        As.store(a_val.value, [get_tile_idx(ty.value, tx.value)])
        
        b_row = (k_base + ty)
        b_val = memref.load(B, [b_row.value, col.value])
        Bs.store(b_val.value, [get_tile_idx(ty.value, tx.value)])
        
        gpu.barrier()
        
        for_k = scf.ForOp(zero.value, tile_c.value, one.value, [acc_val.value])
        with ir.InsertionPoint(for_k.body):
            k_local = for_k.induction_variable
            acc_k = for_k.inner_iter_args[0]
            
            a_smem = As.load([get_tile_idx(ty.value, k_local.value)])
            b_smem = Bs.load([get_tile_idx(k_local.value, tx.value)])
            new_acc = (acc_k + a_smem * b_smem)
            
            scf.yield_([new_acc.value])
        
        gpu.barrier()
        scf.yield_([for_k.results[0].value if hasattr(for_k.results[0], "value") else for_k.results[0]])
    
    memref.store(for_tiles.results[0].value if hasattr(for_tiles.results[0], "value") else for_tiles.results[0], C, [row.value, col.value])

ip.__exit__(None, None, None)

print("="*80)
print("Compiling shared memory matmul...")
hsaco = compile_to_hsaco(ctx.module)
print(f" HSACO: {len(hsaco)} bytes")

np.random.seed(42)
a_host = np.random.randn(M, K).astype(np.float32) * 0.01
b_host = np.random.randn(K, N).astype(np.float32) * 0.01
c_host = np.zeros((M, N), dtype=np.float32)
expected = a_host @ b_host

d_a = hip_check(hip.hipMalloc(M * K * 4))
d_b = hip_check(hip.hipMalloc(K * N * 4))
d_c = hip_check(hip.hipMalloc(M * N * 4))

hip_check(hip.hipMemcpy(d_a, a_host.ctypes.data, M * K * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))
hip_check(hip.hipMemcpy(d_b, b_host.ctypes.data, K * N * 4, hip.hipMemcpyKind.hipMemcpyHostToDevice))

hip_module = hip_check(hip.hipModuleLoadData(hsaco))
kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"matmul_shared"))

grid_x = grid_y = (M + TILE_SIZE - 1) // TILE_SIZE

arg_ptrs = [ctypes.c_void_p(int(d_a)), ctypes.c_void_p(int(d_b)), ctypes.c_void_p(int(d_c))]
args = (ctypes.c_void_p * len(arg_ptrs))(*[ctypes.addressof(p) for p in arg_ptrs])

hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, None))
hip_check(hip.hipDeviceSynchronize())

start_time = time.time()
for _ in range(10):
    hip_check(hip.hipModuleLaunchKernel(kernel_func, grid_x, grid_y, 1, TILE_SIZE, TILE_SIZE, 1, 0, 0, args, None))
hip_check(hip.hipDeviceSynchronize())
avg_time_ms = (time.time() - start_time) * 100

hip_check(hip.hipMemcpy(c_host.ctypes.data, d_c, M * N * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost))

error = np.max(np.abs(c_host - expected))
rel_error = error / (np.max(np.abs(expected)) + 1e-8)
gbps = (2.0 * M * N * K * 4) / (avg_time_ms * 1e-3) / 1e3

print(f"\n{'='*80}")
print(f"Throughput: {avg_time_ms:.3f} ms, {gbps:.1f} GB/s")
print(f"{'='*80}\n")

if rel_error < 1e-3:
    print("\n SHARED MEMORY OPTIMIZATION CORRECT!")
else:
    print(f"\nMax error: {error:.2e}, Relative error: {rel_error:.2e}")
    print(expected[:5,:5])
    print(c_host[:5,:5])

# Cleanup HIP resources explicitly. Some HIP python bindings can crash at interpreter
# shutdown if modules/allocations are still live.
hip_check(hip.hipFree(d_a))
hip_check(hip.hipFree(d_b))
hip_check(hip.hipFree(d_c))
hip_check(hip.hipModuleUnload(hip_module))

# Ensure MLIR objects are torn down before interpreter finalization.
# Some native bindings crash if they are finalized during Python shutdown.
del allocator
del ctx
gc.collect()

# Work around a known finalization crash in some environments where native GPU/MLIR
# bindings execute callbacks while the interpreter is shutting down and the GIL is
# already released. Hard-exiting avoids running Python finalizers.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
