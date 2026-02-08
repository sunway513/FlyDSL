# CuteDSL → FlyDSL Porting Guide

> Mapping concepts, APIs, and patterns from NVIDIA CuteDSL (CUTLASS Python DSL) to AMD FlyDSL (FLIR Python DSL).

## Quick Reference

| CuteDSL (NVIDIA) | FlyDSL (AMD) |
|---|---|
| `import cutlass.cute as cute` | `from flydsl.dialects.ext import flir` |
| `@cute.jit` | `@kernel` / `@jit` (from `flydsl.lang.ir.module`) |
| `cutlass.Constexpr[int]` | Python int literal (static at IR level) |
| `cute.Tensor` | `flir.TensorView` |
| `cute.make_layout(shape, stride)` | `flir.make_layout(shape, stride)` |
| `cute.tiled_divide(layout, tiler)` | `flir.tiled_divide(layout, tiler)` |
| `cute.gemm(tiled_mma, a, b, c)` | `flir.rocm.mfma(a, b, c)` / MFMA intrinsic |
| `cute.copy(atom, src, dst)` | `flir.copy(atom, src, dst)` |
| CUDA / PTX output | HIP / ROCDL / HSACO output |
| SM80 / SM90 / SM100 | GFX942 / GFX950 / GFX90a |

---

## 1. Overview

Both CuteDSL and FlyDSL are Python DSLs for writing high-performance GPU kernels with explicit layout control. They share a common intellectual heritage from NVIDIA's CuTe layout algebra but target different hardware:

| Aspect | CuteDSL | FlyDSL |
|---|---|---|
| **Organization** | Part of NVIDIA CUTLASS | Standalone project |
| **Hardware** | NVIDIA GPUs (CUDA) | AMD GPUs (ROCm/HIP) |
| **IR backend** | CUTLASS MLIR dialects → CUDA/PTX | FLIR MLIR dialect → ROCDL → HSACO |
| **Layout algebra** | `cutlass.cute` module | `flir` MLIR dialect + Python API |
| **Kernel model** | `@cute.jit` decorator | `MlirModule` class + `@kernel`/`@jit` |
| **Memory model** | GMEM → SMEM → RMEM → TMEM | GMEM → LDS → VGPR |
| **Compilation** | Python JIT → MLIR → CUDA binary | Python → MLIR → ROCDL → HSACO binary |
| **Wave/Warp size** | 32 threads (warp) | 64 threads (wavefront) |

---

## 2. Layout System Mapping

The layout algebra is nearly identical between CuteDSL and FlyDSL. Both implement the same mathematical foundation from CuTe.

### 2.1 Core Types

| CuteDSL | FlyDSL | Notes |
|---|---|---|
| `cute.make_shape(M, N)` | `flir.make_shape(M, N)` | Identical semantics |
| `cute.make_stride(s0, s1)` | `flir.make_stride(s0, s1)` | Identical semantics |
| `cute.make_layout(shape, stride)` | `flir.make_layout(shape, stride)` | Identical semantics |
| `cute.make_coord(i, j)` | `flir.make_coord(i, j)` | Identical semantics |
| Python int literal | Python int literal | Both: static at compile time |
| `cutlass.Constexpr[int]` | Python int (always static) | FlyDSL ints are static in MLIR IR |

**CuteDSL:**
```python
import cutlass.cute as cute

shape = cute.make_shape(128, 64)
stride = cute.make_stride(1, 128)    # Column-major
layout = cute.make_layout(shape, stride)
coord = cute.make_coord(3, 5)
```

**FlyDSL:**
```python
from flydsl.dialects.ext import flir

shape = flir.make_shape(128, 64)
stride = flir.make_stride(1, 128)    # Column-major
layout = flir.make_layout(shape, stride)
coord = flir.make_coord(3, 5)
```

### 2.2 Query Operations

| CuteDSL | FlyDSL | Formula |
|---|---|---|
| `cute.size(layout)` | `flir.size(layout)` | Product of shape elements |
| `cute.cosize(layout)` | `flir.cosize(layout)` | Max index + 1 |
| `cute.rank(layout)` | `flir.rank(layout)` | Number of modes |
| `cute.size(layout, mode=[i])` | `flir.get(flir.get_shape(layout), i)` | Size of mode i |

### 2.3 Coordinate Mapping

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.crd2idx(coord, layout)` | `flir.crd2idx(coord, layout)` | Coord → linear index |
| `cute.idx2crd(idx, layout)` | `flir.idx2crd(idx, layout)` | Linear index → coord |

Both compute `index = dot(coord, stride)`.

### 2.4 Layout Algebra Operations

All layout algebra operations have identical names and semantics:

| Operation | CuteDSL | FlyDSL |
|---|---|---|
| **Composition** | `cute.composition(A, B)` | `flir.composition(A, B)` |
| **Complement** | `cute.complement(layout, cotarget)` | `flir.complement(layout, cotarget)` |
| **Coalesce** | `cute.coalesce(layout)` | `flir.coalesce(layout)` |
| **Logical Product** | `cute.logical_product(A, B)` | `flir.logical_product(A, B)` |
| **Zipped Product** | `cute.zipped_product(A, B)` | `flir.zipped_product(A, B)` |
| **Tiled Product** | `cute.tiled_product(A, B)` | `flir.tiled_product(A, B)` |
| **Flat Product** | `cute.flat_product(A, B)` | `flir.flat_product(A, B)` |
| **Raked Product** | `cute.raked_product(A, B)` | `flir.raked_product(A, B)` |
| **Blocked Product** | `cute.blocked_product(A, B)` | `flir.blocked_product(A, B)` |
| **Logical Divide** | `cute.logical_divide(A, tiler)` | `flir.logical_divide(A, tiler)` |
| **Zipped Divide** | `cute.zipped_divide(A, tiler)` | `flir.zipped_divide(A, tiler)` |
| **Tiled Divide** | `cute.tiled_divide(A, tiler)` | `flir.tiled_divide(A, tiler)` |
| **Flat Divide** | `cute.flat_divide(A, tiler)` | `flir.flat_divide(A, tiler)` |
| **Local Partition** | `cute.local_partition(layout, ...)` | `flir.local_partition(layout, ...)` |
| **Local Tile** | `cute.local_tile(layout, ...)` | `flir.local_tile(layout, ...)` |

---

## 3. Kernel Definition

### 3.1 CuteDSL Pattern

CuteDSL uses `@cute.jit` decorator on methods of a kernel class:

```python
import cutlass.cute as cute
import cutlass

class MyKernel:
    def __init__(self, acc_dtype, mma_tiler_mn, ...):
        self.acc_dtype = acc_dtype
        self.mma_tiler = (*mma_tiler_mn, 1)
        ...

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        C: cute.Tensor,
        M: cutlass.Constexpr[int],
    ):
        # Kernel body
        ...
```

### 3.2 FlyDSL Pattern

FlyDSL uses `MlirModule` subclass with `@kernel` and `@jit` decorators:

```python
from flydsl.lang.ir.module import MlirModule, kernel, jit
from flydsl.dialects.ext import flir, arith, gpu
import _mlir.extras.types as T

class MyKernel(MlirModule):
    GPU_MODULE_NAME = "my_kernel"
    GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942">']

    @kernel
    def my_kernel(self,
                  A: T.memref(T.dynamic(), T.f16()),
                  B: T.memref(T.dynamic(), T.f16()),
                  C: T.memref(T.dynamic(), T.f16()),
                  M: T.index()):
        # Kernel body
        ...

    @jit
    def __call__(self, A, B, C, M, stream_ptr: T.i64()):
        # Host-side launch wrapper
        ...
```

**Key differences:**
| CuteDSL | FlyDSL |
|---|---|
| `@cute.jit` on any method | `@kernel` for GPU function, `@jit` for host function |
| Class is a plain Python class | Class extends `MlirModule` |
| Parameters typed with `cute.Tensor`, `cutlass.Constexpr[int]` | Parameters typed with `T.memref(...)`, `T.index()` |
| Compilation via CUTLASS JIT pipeline | Compilation via `flir.compile()` |
| No explicit `GPU_MODULE_NAME` | Must set `GPU_MODULE_NAME` class attribute |

---

## 4. Thread and Block Hierarchy

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.arch.thread_idx()` | `flir.thread_idx("x")` | Thread index within block |
| `cute.arch.block_idx()` | `flir.block_idx("x")` | Block index |
| `cute.arch.block_dim()` | `flir.block_dim("x")` | Block dimension |
| `cute.arch.cluster_idx()` | *(not applicable)* | Cluster index (SM90+) |

**CuteDSL:**
```python
tid = cute.arch.thread_idx()
bid = cute.arch.block_idx()
```

**FlyDSL:**
```python
tid = flir.thread_idx("x")
bid = flir.block_idx("x")
```

---

## 5. Tensor Creation and Memory

### 5.1 Tensor Construction

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.make_tensor(ptr, layout)` | `flir.make_tensor(memref, shape, strides, ...)` | Create a tensor view |
| `cute.make_rmem_tensor(shape, dtype)` | `flir.make_fragment_like(template)` | Register fragment |
| `cute.Tensor` (SSA-based) | `flir.TensorView` (Python wrapper) | Tensor abstraction |

**CuteDSL:**
```python
# Global memory tensor
gmem_A = cute.make_tensor(global_A_ptr, cute.make_layout((M, K), (1, M)))

# Register fragment
rmem_frag = cute.make_rmem_tensor((16, 16), cute.float32)
```

**FlyDSL:**
```python
# Global memory tensor view
gmem_A = flir.TensorView(
    arg_a,                  # memref argument
    (tile_m, tile_k),       # shape
    strides=(1, M),         # strides
    base_indices=(base_m,), # base offset
    element_type=T.f16(),
)

# Register fragment
rmem_frag = flir.make_fragment_like(template_tensor)
```

### 5.2 Memory Hierarchy

| CuteDSL (NVIDIA) | FlyDSL (AMD) | Notes |
|---|---|---|
| Global Memory (GMEM) | Global Memory (GMEM) | Same concept |
| Shared Memory (SMEM, 48-228 KB) | LDS (Local Data Share, 64-160 KB) | Same purpose, different name |
| Register File (RMEM) | VGPR (Vector General Purpose Registers) | Same purpose |
| Tensor Memory (TMEM, Blackwell) | *(not applicable)* | Blackwell-specific |
| L2 Cache | L2 Cache | Similar |

**Shared memory allocation:**

**CuteDSL:**
```python
# Shared memory is allocated within the kernel
smem_layout = cute.make_layout((128, 64), (1, 128))
smem_tensor = cute.make_tensor(smem_ptr, smem_layout)
```

**FlyDSL:**
```python
from flydsl.utils import SmemAllocator

allocator = SmemAllocator(ctx, arch="gfx942")
lds_gen = allocator.allocate_array(T.f16(), num_elems=128*64)
allocator.finalize()  # Inside gpu.module body

# Inside kernel:
base = allocator.get_base()
lds_ptr = lds_gen(base)  # SmemPtr for typed access
```

### 5.3 Swizzling (Bank Conflict Avoidance)

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.Swizzle(M, B, S)` | `flir.swizzle_xor16(row, col, k_blocks)` | LDS bank conflict avoidance |

CuteDSL uses parameterized `Swizzle<M,B,S>` templates. FlyDSL uses XOR16 byte-level swizzle at 16-byte granularity.

**FlyDSL:**
```python
col_swizzled = flir.swizzle_xor16(row, col_bytes, k_blocks16)
```

---

## 6. Data Movement (Copy Operations)

### 6.1 Copy Atoms and Tiled Copies

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.Copy[cute.SM80_TMA[...]]()` | `flir.make_copy_atom(dtype, vector_size)` | Copy descriptor |
| `cute.copy(atom, src, dst)` | `flir.copy(atom, src, dst)` | Execute copy |
| *(implicit in tiled copy)* | `flir.make_tiled_copy_tv(atom, thr, val)` | Distributed copy |
| *(via copy API)* | `tiled.get_slice(tid)` → `.partition_S(src)` / `.partition_D(dst)` | Thread-level slicing |

**CuteDSL:**
```python
# TMA copy (Hopper/Blackwell)
tma_copy = cute.TMA(...)
cute.copy(tma_copy, gmem_src, smem_dst)

# CpAsync copy (Ampere)
cute.cpasync(gmem_src, smem_dst, size)
```

**FlyDSL:**
```python
# Create copy atom
atom = flir.make_copy_atom(T.f16(), vector_size=8)

# Create tiled copy (distributed across threads)
thr_layout = flir.make_ordered_layout(256, 8)  # 256 threads, 8 elements each
val_layout = flir.make_layout(flir.make_shape(1, 8), flir.make_stride(0, 1))
tiled_copy = flir.make_tiled_copy_tv(atom, thr_layout, val_layout)

# Get thread slice
thr_copy = tiled_copy.get_slice(tid)
src_partition = thr_copy.partition_S(src_tensor)
dst_partition = thr_copy.partition_D(dst_tensor)

# Execute copy
flir.copy(tiled_copy, src_tensor, dst_tensor)
```

### 6.2 Buffer Loads (AMD-specific)

FlyDSL provides AMD-specific buffer load intrinsics not present in CuteDSL:

```python
from flydsl.dialects.ext import buffer_ops

# Buffer resource for global memory (AMD buffer load dwordx4)
rsrc = buffer_ops.create_buffer_resource(arg_a, num_records)

# Buffer load 16 bytes
from kernels.mfma_preshuffle_pipeline import buffer_copy_gmem16_dwordx4
vec = buffer_copy_gmem16_dwordx4(flir, arg=arg_a, elem_type=T.i8(),
                                  idx_i32=offset, atom_g2r16=atom, rsrc=rsrc)
```

---

## 7. Compute Operations (MMA / MFMA)

### 7.1 Matrix Multiply

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.TiledMMA(...)` | MFMA intrinsics (see below) | Tiled matrix multiply |
| `cute.gemm(tiled_mma, a, b, c)` | Manual MFMA loop | Matrix multiply-accumulate |
| SM80 Tensor Cores | MFMA (Matrix Fused Multiply-Add) | Hardware MMA unit |
| Warp-level (32 threads) | Wavefront-level (64 threads) | Cooperative MMA |

**CuteDSL (SM90/SM100):**
```python
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a_dtype=cute.float16,
    a_major_mode=cute.LayoutRight,
    b_major_mode=cute.LayoutRight,
    acc_dtype=cute.float32,
    cta_group=tcgen05.CtaGroup.ONE,
    mma_shape=(128, 64),
)
cute.gemm(tiled_mma, a_frag, b_frag, c_accumulator)
```

**FlyDSL (GFX942/GFX950):**
```python
from flydsl.dialects.ext import rocdl

# Direct MFMA intrinsic (16x16xK)
# FP8: mfma_f32_16x16x32_fp8_fp8 (K=32)
# INT8: mfma_i32_16x16x32_i8 (K=32)
# FP16: mfma_f32_16x16x16f16 (K=16)
c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_pack, b_pack, c_acc)

# K64-byte micro-step pattern (2x K32 per step)
for ku in range(tile_k_bytes // 64):
    a_val = lds_load_pack_k32(...)   # Load A from LDS
    b_val = load_b_pack_k32(...)     # Load B from GMEM
    c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_val, b_val, c_acc)
    # second half
    a_val2 = lds_load_pack_k32(...)
    b_val2 = load_b_pack_k32(...)
    c_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(a_val2, b_val2, c_acc)
```

### 7.2 MMA Instruction Mapping

| CuteDSL (NVIDIA) | FlyDSL (AMD) | Sizes |
|---|---|---|
| `mma.sync.m16n8k16.f16` (SM80) | `mfma_f32_16x16x16f16` | M16 N16 K16 |
| `mma.sync.m16n8k32.s8` (SM80) | `mfma_i32_16x16x32_i8` | M16 N16 K32 |
| `tcgen05.mma` (SM100) | `mfma_f32_16x16x32_fp8_fp8` | M16 N16 K32 |
| *(N/A)* | `mfma_scale_x128` (GFX950) | MXFP4 scaled MMA |

---

## 8. Synchronization

| CuteDSL | FlyDSL | Description |
|---|---|---|
| `cute.arch.syncthreads()` | `gpu.barrier()` | Block-level barrier |
| `cute.arch.fence_view_async_shared()` | `gpu.barrier()` | Shared memory fence |
| `pipeline.NamedBarrier(...)` | *(manual barrier)* | Named barriers |
| `cute.arch.cluster_sync()` | *(not applicable)* | Cluster sync (SM90+) |

**CuteDSL:**
```python
cute.arch.syncthreads()
```

**FlyDSL:**
```python
from flydsl.dialects.ext import gpu
gpu.barrier()
```

---

## 9. Compilation and Execution

### 9.1 Compilation Pipeline

**CuteDSL:**
```python
# JIT compilation is implicit via @cute.jit
kernel = MyKernel(acc_dtype=cutlass.Float32, ...)
kernel(A_tensor, B_tensor, C_tensor, M=M)  # First call triggers compilation
```

**FlyDSL:**
```python
import flydsl

# Option 1: Using compile() (preferred for production)
mod = MyKernel()
executor = flydsl.compile(mod)
executor(A_torch, B_torch, C_torch, M)

# Option 2: Using compile_to_hsaco() (for tests)
from tests.utils import compile_to_hsaco
hsaco_bytes = compile_to_hsaco(mod)
```

### 9.2 Environment Variables

| CuteDSL | FlyDSL | Description |
|---|---|---|
| *(CUDA_VISIBLE_DEVICES)* | `FLIR_CHIP` / `FLIR_GPU_ARCH` | Target architecture |
| *(N/A)* | `FLIR_DUMP_IR=1` | Dump intermediate MLIR IR |
| *(N/A)* | `FLIR_DUMP_DIR=/path` | IR dump location |
| *(N/A)* | `COMPILE_ONLY=1` | Skip execution, compile only |
| *(N/A)* | `FLIR_NO_CACHE=1` | Disable compilation cache |
| *(N/A)* | `FLIR_TIME_COMPILE=1` | Print compilation timing |

---

## 10. Complete Porting Example: Vector Addition

### CuteDSL Version

```python
import cutlass.cute as cute
import cutlass

class VecAddKernel:
    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor, C: cute.Tensor,
                 N: cutlass.Constexpr[int]):
        tid = cute.arch.thread_idx()
        bid = cute.arch.block_idx()
        idx = bid * 256 + tid
        if idx < N:
            C[idx] = A[idx] + B[idx]
```

### FlyDSL Version

```python
from flydsl.lang.ir.module import MlirModule, kernel, jit
from flydsl.dialects.ext import flir, arith, gpu
import _mlir.extras.types as T

class VecAddKernel(MlirModule):
    GPU_MODULE_NAME = "vec_add"

    @kernel
    def vec_add(self,
                A: T.memref(T.dynamic(), T.f32()),
                B: T.memref(T.dynamic(), T.f32()),
                C: T.memref(T.dynamic(), T.f32()),
                N: T.index()):
        tid = flir.thread_idx("x")
        bid = flir.block_idx("x")
        block_size = arith.constant(256, index=True)
        idx = bid * block_size + tid

        # Load (via memref dialect, accessed through flir)
        a_val = flir.memref.load(A, [idx])
        b_val = flir.memref.load(B, [idx])

        # Compute
        c_val = arith.addf(a_val, b_val)

        # Store
        flir.memref.store(c_val, C, [idx])
```

---

## 11. GEMM Porting Checklist

When porting a CuteDSL GEMM to FlyDSL:

1. **Replace `@cute.jit`** with `MlirModule` + `@kernel` / `@jit`
2. **Replace `cute.Tensor`** parameters with `T.memref(...)` types
3. **Replace `cute.make_layout()`** with `flir.make_layout()` (identical API)
4. **Replace NVIDIA tensor cores** with AMD MFMA intrinsics:
   - `mfma_f32_16x16x32_fp8_fp8` for FP8
   - `mfma_i32_16x16x32_i8` for INT8
   - `mfma_f32_16x16x16f16` for FP16
5. **Replace SMEM allocation** with `SmemAllocator`
6. **Replace `cute.Swizzle`** with `flir.swizzle_xor16()`
7. **Replace TMA/CpAsync** with `buffer_ops` (AMD buffer loads)
8. **Adjust warp size**: 32 → 64 (AMD wavefront)
9. **Adjust shared memory limits**: Check `SMEM_CAPACITY_MAP` (gfx942=64KB, gfx950=160KB)
10. **Replace `cute.arch.syncthreads()`** with `gpu.barrier()`
11. **Use `flydsl.compile()`** instead of CuteDSL JIT compilation

---

## 12. Key Differences Summary

| Aspect | CuteDSL | FlyDSL |
|---|---|---|
| **Language** | Python + `@cute.jit` | Python + MLIR emission |
| **Hardware** | NVIDIA CUDA GPUs | AMD ROCm GPUs |
| **Warp/Wave size** | 32 (warp) | 64 (wavefront) |
| **Shared memory** | SMEM (48-228 KB) | LDS (64-160 KB) |
| **MMA unit** | Tensor Core (HMMA/GMMA/TCGEN05) | MFMA |
| **Global loads** | `ld.global` / TMA / CpAsync | `buffer_load_dwordx4` |
| **Swizzle** | `Swizzle<M,B,S>` | `swizzle_xor16` |
| **Compilation** | Python JIT → CUDA binary | Python → MLIR → HSACO |
| **Caching** | *(managed by CUTLASS)* | `FileCache` (SHA-256 keyed) |
| **Layout algebra** | Identical semantics | Identical semantics |
| **IR dump** | *(via CUTLASS debug flags)* | `FLIR_DUMP_IR=1` |

---

## 13. Source Files

### CuteDSL (reference)
- CUTLASS GitHub: `python/cutlass/cute/` — Core CuteDSL module
- CUTLASS GitHub: `examples/blackwell/` — Example kernels (dense GEMM, grouped GEMM, flash attention)
- PyTorch Inductor: `torch/_inductor/codegen/cutedsl/` — CuteDSL integration in PyTorch
- PyCute: `torch/distributed/_pycute/layout.py` — Pure-Python layout algebra reference

### FlyDSL
- `flydsl/src/flydsl/dialects/ext/flir.py` — Layout algebra Python API
- `flydsl/src/flydsl/lang/ir/module.py` — MlirModule, @kernel, @jit
- `flydsl/src/flydsl/compiler/compiler.py` — `flir.compile()` pipeline
- `flydsl/src/flydsl/utils/smem_allocator.py` — SmemAllocator
- `flydsl/src/flydsl/dialects/ext/rocm.py` — ROCm dialect helpers
- `kernels/preshuffle_gemm.py` — GEMM implementation example
- `tests/kernels/test_vec_add.py` — VecAdd example with full layout algebra
