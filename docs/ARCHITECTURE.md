# CuTe IR Architecture

## Overview

CuTe IR is a multi-level MLIR dialect system that progressively lowers high-level layout algebra to executable GPU kernels.

## Dialect Hierarchy

### Level 1: cute_ir (Hardware-Agnostic)

**Purpose**: Express layout algebra without hardware specifics

**Key Abstractions**:
- **Shape**: Multi-dimensional sizes `!cute.shape<N>`
- **Stride**: Memory access patterns `!cute.stride<N>`
- **Layout**: Complete memory mapping `!cute.layout<N>`
- **Tensor**: Data containers `!cute.tensor<T, Layout>`

**Operations**:
```mlir
// Layout construction
%layout = cute.make_layout %shape, %stride

// Layout queries
%size = cute.size %layout
%rank = cute.rank %layout

// Layout transformations
%flat = cute.flatten %layout
%comp = cute.composition %layout1, %layout2

// Coordinate mapping
%idx = cute.crd2idx %layout, %coord
```

**Design Principles**:
- Zero-cost abstractions
- Compile-time layout analysis
- Hardware-independent semantics

### Level 2: cute_nvgpu_ir (GPU Hardware-Aware)

**Purpose**: Express GPU-specific operations while maintaining layout semantics

**Key Abstractions**:
- **MmaAtom**: Single MMA instruction descriptor
- **TiledMma**: Multi-warp MMA pattern
- **CopyAtom**: Memory copy instruction
- **TMA**: Tensor Memory Accelerator (Hopper+)

**Operations**:
```mlir
// Warp-level MMA (SM80)
%d = cute_nvgpu.warp_mma_f16bf16 %a, %b, %c : 
    !cute_nvgpu.mma_atom<16,8,16,f16>

// Warpgroup MMA (SM90)
%d = cute_nvgpu.warpgroup_mma %a, %b, %c, %barrier :
    !cute_nvgpu.tiled_mma<128,128,64>

// TMA Load (SM90)
cute_nvgpu.tma_load_execute %tma_desc, %smem, %barrier
```

**Hardware Mapping**:
| Dialect Op | Hardware Instruction |
|-----------|---------------------|
| `warp_mma_f16bf16` | `mma.sync.aligned.m16n8k16.f16.f16` |
| `warpgroup_mma` | `wgmma.mma_async.sync.aligned` |
| `ldmatrix` | `ldmatrix.sync.aligned` |
| `tma_load_execute` | `cp.async.bulk.tensor` |

## Compilation Pipeline

### Stage 1: Layout Analysis

**Pass**: `cute-layout-analysis`

**Purpose**: Analyze layout properties at compile time

```mlir
// Before
%layout = cute.make_layout (!cute.shape<2>), (!cute.stride<2>)

// After analysis (attributes added)
%layout = cute.make_layout (!cute.shape<2>), (!cute.stride<2>)
  {size = 128, is_column_major = false, is_contiguous = true}
```

### Stage 2: Canonicalization

**Pass**: `cute-canonicalize`

**Purpose**: Simplify layout expressions

```mlir
// Before
%l1 = cute.make_layout %s1, %st1
%l2 = cute.composition %l1, %l1
%size = cute.size %l2

// After
%size = constant 16384 : !cute.int
```

### Stage 3: Layout → Standard Lowering

**Pass**: `cute-to-standard`

**Purpose**: Lower layout operations to standard MLIR

```mlir
// Before (CuTe IR)
%idx = cute.crd2idx %layout, %coord

// After (Standard MLIR)
%c0 = arith.constant 0
%idx = scf.for %i = %c0 to %rank step 1 iter_args(%acc = %c0) {
  %shape_i = cute.shape %layout, %i
  %stride_i = cute.stride %layout, %i
  %coord_i = cute.coord %coord, %i
  %prod = arith.muli %coord_i, %stride_i
  %new_acc = arith.addi %acc, %prod
  scf.yield %new_acc
}
```

### Stage 4: GPU Operation Lowering

**Pass**: `cute-nvgpu-to-nvgpu`

**Purpose**: Lower CuTe GPU ops to standard NVGPU dialect

```mlir
// Before (CuTe NVGPU)
%d = cute_nvgpu.warpgroup_mma %a, %b, %c, %barrier

// After (NVGPU)
%d = nvgpu.warpgroup_mma %a, %b, %c 
  {m = 128, n = 128, k = 64}
nvgpu.mbarrier.arrive %barrier
```

### Stage 5: NVVM Lowering

**Pass**: `convert-nvgpu-to-nvvm` (MLIR builtin)

```mlir
// Before (NVGPU)
%d = nvgpu.ldmatrix %ptr : !nvgpu.matrix<16x16xf16>

// After (NVVM)
%d = llvm.inline_asm "ldmatrix.sync.aligned.x4.m8n8.shared.b16 ..."
```

### Stage 6: Code Generation

**Tool**: `mlir-translate`, `llc`, `ptxas`

```
NVVM LLVM IR → PTX Assembly → CUBIN Binary
```

## Memory Model

### Layout Semantics

A layout maps coordinates to memory indices:

```
Layout: (Shape, Stride) → Index
Index(coord) = Σ(coord[i] * stride[i])
```

**Example**:
```mlir
Shape:  (8, 16)
Stride: (1, 8)

Coord (3, 5) → Index = 3*1 + 5*8 = 43
```

### Tiling Strategy

**Hierarchical Tiling**:
```
Global Memory (M, N, K)
    ↓ Tile to CTAs
CTA Tile (TileM, TileN, TileK)
    ↓ Partition to Warps
Warp Tile (WarpM, WarpN, WarpK)
    ↓ Map to MMA
MMA Shape (16, 8, 16)
```

**Implementation**:
```mlir
%global_layout = cute.make_layout %M, %N, %K
%cta_tile = cute.tile %global_layout, %TileM, %TileN, %TileK
%warp_partition = cute.local_partition %cta_tile, %warp_id
```

## Optimization Strategies

### 1. Layout Fusion

**Before**:
```mlir
%l1 = cute.partition %layout, %tile
%l2 = cute.flatten %l1
%l3 = cute.composition %l2, %another
```

**After**:
```mlir
%l3 = cute.fused_partition_flatten_compose %layout, %tile, %another
```

### 2. Memory Coalescing

**Analysis**: Detect non-coalesced access patterns

```mlir
%stride = cute.stride %layout, 0
%is_coalesced = arith.cmpi eq, %stride, 1
```

**Transformation**: Transpose layouts when beneficial

### 3. Swizzle Optimization

**Purpose**: Reduce bank conflicts in shared memory

```mlir
// Before (bank conflicts)
%smem_layout = cute.make_layout (128, 64), (1, 128)

// After (swizzled)
%smem_layout = cute.swizzle %smem_layout, 
  !cute.swizzle<mode=128B, bits=3>
```

### 4. Async Pipeline

**Pattern**: Overlap computation and memory transfer

```mlir
// Software pipeline
scf.for %i = 0 to %tiles {
  // Stage 0: Prefetch
  cute_nvgpu.tma_load_execute %tma, %smem_next, %barrier
  
  // Stage 1: Compute
  %result = cute_nvgpu.warpgroup_mma %a, %b, %c, %barrier_prev
  
  // Stage 2: Store
  cute.copy %result, %gmem
  
  // Barrier sync
  cute_nvgpu.mbarrier_wait %barrier
}
```

## Type System

### Type Hierarchy

```
!cute.int                    # Platform-dependent integer
!cute.shape<N>               # N-dimensional shape
!cute.stride<N>              # N-dimensional stride
!cute.layout<N>              # Complete layout (shape + stride)
!cute.coord<N>               # N-dimensional coordinate
!cute.tile<M,N,K>            # Tile descriptor
!cute.tensor<T, Layout>      # Typed tensor with layout

!cute_nvgpu.mma_atom<M,N,K,T>       # MMA instruction
!cute_nvgpu.tiled_mma<M,N,K>        # Multi-warp MMA
!cute_nvgpu.copy_atom<Size>         # Copy instruction
!cute_nvgpu.tma_load<Shape>         # TMA load descriptor
```

### Type Constraints

**Layout Rank Consistency**:
```mlir
// Valid: matching ranks
%layout = cute.make_layout %shape : !cute.shape<2>, 
                           %stride : !cute.stride<2>

// Invalid: mismatched ranks
%layout = cute.make_layout %shape : !cute.shape<2>,
                           %stride : !cute.stride<3>  // Error!
```

**MMA Shape Constraints**:
```mlir
// Valid: SM80 MMA shape
!cute_nvgpu.mma_atom<16,8,16,f16>

// Invalid: unsupported shape
!cute_nvgpu.mma_atom<17,9,15,f16>  // Error!
```

## Runtime Architecture

### Compilation Flow

```
┌──────────────┐
│  Python API  │ cute.Gemm(M, N, K)
└──────┬───────┘
       │
┌──────▼───────┐
│ CuteCompiler │ MLIR → PTX → CUBIN
└──────┬───────┘
       │
┌──────▼────────┐
│ KernelExecutor│ Load module, launch kernel
└──────┬────────┘
       │
┌──────▼────────┐
│  CUDA Driver  │ cuLaunchKernel
└───────────────┘
```

### Memory Management

**Device Buffers** (RAII):
```cpp
DeviceBuffer<float> d_A(M * K);  // Allocate
d_A.copy_from_host(h_A, M * K);  // Transfer
// ... use d_A ...
// Automatic deallocation on scope exit
```

**TMA Descriptors** (SM90+):
```cpp
TMADescriptor tma;
tma.initialize_2d(
    d_A, CUDA_R_16F,
    global_M, global_K,
    tile_M, tile_K,
    SwizzleMode::Swizzle128B
);
```

## Performance Characteristics

### Layout Operations

| Operation | Complexity | Optimized |
|-----------|-----------|-----------|
| `make_layout` | O(1) | Compile-time |
| `size` | O(1) | Compile-time constant folding |
| `crd2idx` | O(rank) | Loop unrolling |
| `flatten` | O(1) | Compile-time |
| `composition` | O(1) | Compile-time |



---

This architecture enables:
- ✅ Hardware-agnostic layout programming
- ✅ Progressive lowering with optimization opportunities
- ✅ Type-safe GPU kernel generation
- ✅ Zero-cost abstractions
