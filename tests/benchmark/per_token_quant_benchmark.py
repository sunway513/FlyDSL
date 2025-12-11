#!/usr/bin/env python3
"""
Benchmark: Per-Token Quantization Kernel
Reference: /data/zhimding/aiter/aiter/ops/quant.py
Based on: /data/zhimding/rocDSL/tests/benchmark/vecAdd.py
"""

import sys
import os
import argparse

sys.path.insert(
    0, os.path.join(os.environ.get("MLIR_PATH"), "tools/mlir/python_packages/mlir_core")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../build/python_bindings")
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ctypes
import numpy as np

try:
    import torch
    import aiter
    from aiter.ops.quant import per_token_quant_hip

    HAS_AITER = True
except ImportError:
    HAS_AITER = False

from hip import hip
from rocdsl.compiler.context import RAIIMLIRContextModule
from rocdsl.dialects.ext import arith, gpu, rocir
from rocdsl.dialects.ext.gpu import lds_space
from rocdsl.runtime.hip_util import hip_check, get_hip_arch
from rocdsl.utils import SmemAllocator
from mlir import ir
from mlir.dialects import (
    math as _math_mlir,
    scf,
    vector,
    memref,
    gpu as mlir_gpu,
)
import mlir.extras.types as T
from utils import compile_to_hsaco
from tests.test_common import run_perftest


def benchmark_per_token_quant(M=4096, N=8192):
    print("\n" + "=" * 80)
    print(f"Benchmark: Per-Token Quantization Performance (RocDSL) [M={M}, N={N}]")
    print("=" * 80)

    gpu_arch = get_hip_arch()
    print(f"Detected HIP Arch: {gpu_arch}")

    NUM_WARPS = 4
    BLOCK_SIZE = 64 * NUM_WARPS
    VEC_WIDTH = 32

    ELEMS_PER_THREAD = VEC_WIDTH
    ELEMS_PER_BLOCK_ITER = BLOCK_SIZE * ELEMS_PER_THREAD

    assert N % VEC_WIDTH == 0, "N must be multiple of VecWidth (32)"
    ITERS = (N + ELEMS_PER_BLOCK_ITER - 1) // ELEMS_PER_BLOCK_ITER

    total_elements = M * N
    total_bytes_rw = (M * N * 2) * 1 + (M * N * 1) + (M * 4)

    print(f"Configuration:")
    print(f"  - Shape: [{M}, {N}]")
    print(f"  - Block Size: {BLOCK_SIZE}")
    print(f"  - Total Elements: {total_elements/1e6:.2f}M")
    print(f"  - Loops per Block: {ITERS}")
    print(f"  - Est. Memory Traffic: {total_bytes_rw/1e9:.2f} GB per call")

    np.random.seed(42)
    input_data_fp16 = np.random.uniform(-5.0, 5.0, size=(M, N)).astype(np.float16)
    input_data = input_data_fp16.astype(np.float32)
    dtypeMax = 127.0
    per_token_amax = np.max(np.abs(input_data), axis=1)
    per_token_scale = per_token_amax / dtypeMax
    per_token_scale[per_token_scale == 0] = 1.0
    scale_expanded = per_token_scale[:, np.newaxis]
    output_ref = (input_data / scale_expanded).astype(np.int8)

    ctx = RAIIMLIRContextModule(allow_unregistered_dialects=True)
    gpu.set_container_module(ctx.module)
    allocator = SmemAllocator(ctx, arch=gpu_arch)
    f32_type = ir.F32Type.get()
    red_buffer_decl = allocator.allocate_array(f32_type, 64)

    @gpu.module("quant_mod", [f'#rocdl.target<chip = "{gpu_arch}", abi = "500">'])
    def gpu_mod():
        allocator.finalize()

    ip = ir.InsertionPoint.at_block_begin(gpu_mod.regions[0].blocks[0])
    ip.__enter__()

    red_type = T.memref(NUM_WARPS, T.f32(), memory_space=lds_space())
    memref.global_(sym_name="red_buffer", type_=red_type, alignment=16)

    def unwrap(v):
        return v._value if hasattr(v, "_value") else v

    @gpu.func(emit=True)
    def quant_kernel(
        input: T.memref(M * N, T.f16()),
        output: T.memref(M * N, T.i8()),
        scales: T.memref(M, T.f32()),
    ):
        tid_x = rocir.thread_idx("x")
        bid_x = rocir.block_idx("x")
        tid_linear = tid_x

        base_ptr = allocator.get_base()
        red_val = red_buffer_decl(base_ptr).get()

        f_0 = arith.f32(0.0)
        f_1 = arith.f32(1.0)
        f_127 = arith.f32(127.0)
        c_0 = arith.index(0)

        local_max = f_0
        cached_vecs = []

        vec_type_f16 = T.vector(VEC_WIDTH, T.f16())
        vec_type_f32 = T.vector(VEC_WIDTH, T.f32())

        # Load phase: 使用 rocir 坐标计算
        c_m = arith.index(M)
        c_n = arith.index(N)
        c_1 = arith.index(1)

        for i in range(ITERS):
            c_chunk_offset = arith.index(i * ELEMS_PER_BLOCK_ITER)
            c_vec_width = arith.index(VEC_WIDTH)
            thread_offset = tid_linear * c_vec_width
            col_base = c_chunk_offset + thread_offset

            is_valid = col_base < c_n

            if_load = scf.IfOp(is_valid.value, [vec_type_f16], hasElse=True)
            with ir.InsertionPoint(if_load.then_block):
                coord = rocir.make_coord(bid_x, col_base)
                layout_global = rocir.make_layout((c_m, c_n), stride=(c_n, c_1))
                linear_idx = rocir.crd2idx(coord, layout_global)

                vec_val = vector.load(vec_type_f16, input, [linear_idx.value])
                scf.YieldOp([vec_val])

            with ir.InsertionPoint(if_load.else_block):
                zero_vec = arith.constant_vector(0.0, vec_type_f16).value
                scf.YieldOp([zero_vec])

            vec_val_f16 = if_load.results[0]
            cached_vecs.append(vec_val_f16)

        for i in range(ITERS):
            vec_val_f16 = cached_vecs[i]
            vec_val = arith.extf(vec_type_f32, vec_val_f16).value
            vec_abs = _math_mlir.absf(vec_val)
            chunk_max = vector.ReductionOp(
                T.f32(), vector.CombiningKind.MAXIMUMF, vec_abs
            ).result
            local_max = arith.maximum(local_max, chunk_max).value

        current_val = arith.ArithValue(local_max)
        for s in [32, 16, 8, 4, 2, 1]:
            offset = arith.i32(s)
            width = arith.i32(64)

            shuffled_op = mlir_gpu.ShuffleOp(
                current_val.value,
                offset.value,
                width.value,
                mode=mlir_gpu.ShuffleMode.XOR,
            )
            shuffled_val = shuffled_op.results[0]
            current_val = arith.maximum(current_val, shuffled_val)

        c_64 = arith.index(64)
        c_0_idx = arith.index(0)
        warp_id = tid_linear / c_64
        lane_id = tid_linear % c_64

        is_lane_0 = lane_id == c_0_idx

        if_warp_store = scf.IfOp(is_lane_0.value)
        with ir.InsertionPoint(if_warp_store.then_block):
            memref.store(current_val.value, red_val, [warp_id.value])
            scf.YieldOp([])

        mlir_gpu.BarrierOp()

        is_thread_0 = (tid_linear == c_0_idx)

        if_block_reduce = scf.IfOp(is_thread_0.value)
        with ir.InsertionPoint(if_block_reduce.then_block):
            final_max_val = arith.ArithValue(f_0)
            for w in range(NUM_WARPS):
                c_w = arith.index(w).value
                val = memref.load(red_val, [c_w])
                final_max_val = arith.maximum(final_max_val, val)

            memref.store(final_max_val.value, red_val, [c_0.value])
            scf.YieldOp([])

        mlir_gpu.BarrierOp()

        reduced_max = memref.load(red_val, [c_0.value])
        scale = reduced_max / 127

        is_zero = scale == 0
        final_scale = arith.select(is_zero, f_1, scale)

        if_op = scf.IfOp(is_thread_0.value)
        with ir.InsertionPoint(if_op.then_block):
            bid_x_val = bid_x.value if hasattr(bid_x, "value") else bid_x
            memref.store(final_scale.value, scales, [bid_x_val])
            scf.YieldOp([])

        vec_scale = vector.BroadcastOp(vec_type_f32, final_scale.value).result
        vec_f1 = vector.BroadcastOp(vec_type_f32, f_1.value).result
        vec_f1_arith = arith.ArithValue(vec_f1)
        vec_scale_arith = arith.ArithValue(vec_scale)
        vec_inv_scale = (vec_f1_arith / vec_scale_arith).value

        for i in range(ITERS):
            c_chunk_offset = arith.index(i * ELEMS_PER_BLOCK_ITER)
            c_vec_width = arith.index(VEC_WIDTH)
            thread_offset = tid_linear * c_vec_width
            col_base = c_chunk_offset + thread_offset

            is_valid = col_base < c_n
            if_store = scf.IfOp(is_valid.value)
            with ir.InsertionPoint(if_store.then_block):
                coord = rocir.make_coord(bid_x, col_base)
                layout_global = rocir.make_layout((c_m, c_n), stride=(c_n, c_1))
                linear_idx = rocir.crd2idx(coord, layout_global)

                vec_val_f16 = cached_vecs[i]
                vec_val = arith.extf(vec_type_f32, vec_val_f16)

                vec_scaled = vec_val * vec_inv_scale
                vec_i8_type = T.vector(VEC_WIDTH, T.i8())
                vec_quant = arith.fptosi(vec_i8_type, vec_scaled)

                vector.store(vec_quant.value, output, [linear_idx.value])
                scf.YieldOp([])

    ip.__exit__(None, None, None)

    print("Compiling MLIR module...")
    hsaco = compile_to_hsaco(ctx.module)
    print(f"Compiled to HSACO: {len(hsaco)} bytes")

    input_size_bytes = M * N * 2
    output_size_bytes = M * N * 1
    scales_size_bytes = M * 4

    d_input = hip_check(hip.hipMalloc(input_size_bytes))
    d_output = hip_check(hip.hipMalloc(output_size_bytes))
    d_scales = hip_check(hip.hipMalloc(scales_size_bytes))

    hip_check(
        hip.hipMemcpy(
            d_input,
            input_data_fp16.ctypes.data,
            input_size_bytes,
            hip.hipMemcpyKind.hipMemcpyHostToDevice,
        )
    )

    hip_module = hip_check(hip.hipModuleLoadData(hsaco))
    kernel_func = hip_check(hip.hipModuleGetFunction(hip_module, b"quant_kernel"))

    arg_ptrs = [
        ctypes.c_void_p(int(d_input)),
        ctypes.c_void_p(int(d_output)),
        ctypes.c_void_p(int(d_scales)),
    ]
    args_array = (ctypes.c_void_p * len(arg_ptrs))(
        *[ctypes.addressof(p) for p in arg_ptrs]
    )

    def hip_kernel_launch():
        hip.hipModuleLaunchKernel(
            kernel_func,
            M,
            1,
            1,
            BLOCK_SIZE,
            1,
            1,
            sharedMemBytes=0,
            stream=None,
            kernelParams=args_array,
            extra=None,
        )
        hip.hipDeviceSynchronize()

    print("Running benchmark...")
    _, avg_us = run_perftest(hip_kernel_launch, num_iters=20, num_warmup=2)
    
    # Calculate metrics
    bandwidth_gbs = total_bytes_rw / (avg_us / 1e6) / 1e9
    avg_ms = avg_us / 1000
    
    results = {
        "avg_ms": avg_ms,
        "avg_us": avg_us,
        "bandwidth_gbs": bandwidth_gbs,
        "size": total_elements,
        "total_bytes": total_bytes_rw,
    }

    output_host = np.zeros((M, N), dtype=np.int8)
    scales_host = np.zeros(M, dtype=np.float32)

    hip_check(
        hip.hipMemcpy(
            output_host.ctypes.data,
            d_output,
            output_size_bytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToHost,
        )
    )
    hip_check(
        hip.hipMemcpy(
            scales_host.ctypes.data,
            d_scales,
            scales_size_bytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToHost,
        )
    )

    scale_diff = np.max(np.abs(scales_host - per_token_scale))
    output_diff = np.max(
        np.abs(output_host.astype(np.float32) - output_ref.astype(np.float32))
    )

    print(f"\nRocDSL Kernel Results:")
    print(f"  Max Scale Diff:  {scale_diff:.2e}")
    print(f"  Max Output Diff: {output_diff:.2e}")
    print(f"  {results}")

    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipFree(d_scales))
    hip_check(hip.hipModuleUnload(hip_module))

    # ========================================================================
    # Benchmark Reference Implementation (aiter)
    # ========================================================================
    if HAS_AITER:  # HAS_ATIER
        print("\n" + "=" * 80)
        print("Benchmarking Reference Implementation (aiter)")
        print("=" * 80)

        input_torch = torch.from_numpy(input_data_fp16).cuda()

        def launch_aiter():
            per_token_quant_hip(input_torch)
            torch.cuda.synchronize()

        _, aiter_avg_us = run_perftest(launch_aiter, num_iters=20, num_warmup=2)
        
        aiter_bandwidth_gbs = total_bytes_rw / (aiter_avg_us / 1e6) / 1e9
        aiter_avg_ms = aiter_avg_us / 1000
        
        aiter_results = {
            "avg_ms": aiter_avg_ms,
            "avg_us": aiter_avg_us,
            "bandwidth_gbs": aiter_bandwidth_gbs,
            "size": total_elements,
            "total_bytes": total_bytes_rw,
        }

        output_torch, scale_torch = per_token_quant_hip(input_torch)
        torch.cuda.synchronize()

        output_ref_torch = output_torch.cpu().numpy()
        scale_ref_torch = scale_torch.squeeze().cpu().numpy()

        scale_diff_ref = np.max(np.abs(scale_ref_torch - per_token_scale))
        output_diff_ref = np.max(
            np.abs(output_ref_torch.astype(np.float32) - output_ref.astype(np.float32))
        )

        print(f"\n  Reference Correctness Check:")
        print(f"  Max Scale Diff:  {scale_diff_ref:.2e}")
        print(f"  Max Output Diff: {output_diff_ref:.2e}")

        rocdsl_time = results["avg_ms"]
        aiter_time = aiter_results["avg_ms"]
        speedup = aiter_time / rocdsl_time

        print(f"\n" + "=" * 80)
        print(f"Performance Comparison:")
        print(
            f"  RocDSL:     {rocdsl_time:7.3f} ms  ({results['bandwidth_gbs']:8.2f} GB/s)"
        )
        print(
            f"  Reference:  {aiter_time:7.3f} ms  ({aiter_results['bandwidth_gbs']:8.2f} GB/s)"
        )
        print(f"  Speedup:    {speedup:7.2f}x")
        print("=" * 80)

    return output_diff <= 1.0


def test_benchmark_per_token_quant():
    """Pytest wrapper for per-token quantization benchmark."""
    print("\n" + "=" * 80)
    print("ROCm GPU Benchmark - Per-Token Quantization")
    print(f"GPU: {get_hip_arch()}")
    print("=" * 80)
    assert (
        benchmark_per_token_quant()
    ), "Per-token quantization benchmark failed correctness check"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Per-Token Quantization")
    parser.add_argument(
        "-m", "--tokens", type=int, default=4096, help="Number of tokens (M)"
    )
    parser.add_argument(
        "-n", "--hidden", type=int, default=8192, help="Hidden dimension (N)"
    )
    args = parser.parse_args()

    success = benchmark_per_token_quant(M=args.tokens, N=args.hidden)

    if not success:
        sys.exit(1)
