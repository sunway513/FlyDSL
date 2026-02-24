# Skill: FlyDSL MFMA Kernel Development

## Description
Patterns, pitfalls, and reference implementations for writing high-performance MFMA GPU kernels using FlyDSL (Python DSL + MLIR compiler) on AMD GPUs. Distilled from developing Flash Attention V4.0-V4.3 and Preshuffle GEMM kernels.

## When to Use
- Writing a new MFMA-based kernel (GEMM, attention, reduction) in FlyDSL
- Debugging MLIR/LLVM issues (barrier reordering, register pressure, type mismatches)
- Optimizing an existing FlyDSL kernel (LDS, occupancy, cooperative loads)
- Porting a CK/ASM kernel pattern to FlyDSL

---

## Build & Test

```bash
# Build MLIR backend
scripts/build_llvm.sh

# Build FLIR C++ + Python bindings
flir/build.sh

# Run all tests (55 tests)
scripts/run_tests.sh

# Run specific kernel test
python tests/kernels/test_flash_attention_v3.py
```

---

## 1. ArithValue Wrapping (Most Common Pitfall)

FlyDSL wraps MLIR Values in `ArithValue` to support Python operators (+, -, *, //). This is convenient but causes silent failures when raw MLIR Values are expected.

### Rule: Always `arith.as_value()` before MLIR op constructors

```python
# BAD — passes ArithValue wrapper, may crash or produce wrong IR
result = _arith.AddIOp(loop.induction_variable, step)

# GOOD — unwraps to raw MLIR Value
iv = arith.as_value(loop.induction_variable)
step_val = arith.as_value(step)
result = _arith.AddIOp(iv, step_val)

# ALSO GOOD — Python operators work directly on ArithValue
result = loop.induction_variable + step  # generates AddIOp automatically
```

### Where ArithValue appears:
| Source | Returns ArithValue? | Unwrap? |
|--------|:---:|:---:|
| `scf.for_` induction_variable | Yes | `arith.as_value()` |
| `scf.for_` inner_iter_args | Yes | `arith.as_value()` |
| `vec_ext.load_op()` | Yes | `arith.as_value()` |
| `vec_ext.broadcast()` | Yes | `arith.as_value()` |
| `arith.constant()` | Yes | `arith.as_value()` |
| Kernel parameters (scalars) | Yes | `arith.as_value()` |
| `_memref.LoadOp().result` | No | Not needed |
| `scf.for_` results | No (raw Value) | Not needed |

### Python operators on ArithValue:
- Works on **index** and **f32** types
- `+` → `AddIOp`/`AddFOp`, `-` → `SubIOp`/`SubFOp`, `*` → `MulIOp`/`MulFOp`, `//` → `DivSIOp`/`DivFOp`
- Use these when possible — cleaner than explicit op constructors

---

## 2. `scf.for_` Loop Pattern

```python
from flydsl.dialects.ext import scf

# Pattern: loop with iter_args
lb = arith.constant(0, index=True)
ub = arith.constant(N, index=True)
step = arith.constant(1, index=True)
init_acc = arith.constant(0.0)

with scf.for_(lb, ub, step, [init_acc]) as loop:
    # 1. Unwrap induction variable and iter_args
    iv = arith.as_value(loop.induction_variable)
    acc = arith.as_value(loop.inner_iter_args[0])

    # 2. Compute
    val = _memref.LoadOp(memref, [iv]).result
    new_acc = acc + val  # ArithValue operators work here

    # 3. Yield (MUST unwrap before yielding)
    scf.yield_([arith.as_value(new_acc)])

# 4. Get results (raw MLIR Values, no unwrap needed)
final_acc = loop.results[0]
```

### Common mistakes:
- Forgetting to unwrap `inner_iter_args` → silent wrong computation
- Forgetting to unwrap before `scf.yield_()` → crash
- Using `range()` instead of `range_constexpr()` for compile-time unrolled loops inside kernels

### `range_constexpr()` for compile-time loops:
```python
from flydsl.dialects.ext.python_control_flow import range_constexpr

# This unrolls at compile time (no scf.for_ in IR)
for i in range_constexpr(4):
    acc[i] = acc[i] + val[i]
```

---

## 3. MFMA Intrinsics

### mfma_f32_16x16x16f16
```python
# Signature: (result_type, [a, b, acc, cbsz, abid, blgp])
result = rocdl.mfma_f32_16x16x16f16(v4f32_type, [a_v4f16, b_v4f16, acc_v4f32, 0, 0, 0])
```

**Thread-data mapping** (64-lane wave, lane = b*16 + n, b=0..3, n=0..15):
- **Output C**: lane owns `C[b*4+ii, n]` for ii=0..3, stored as v4f32
- **A operand**: lane provides `A[n, b*4:b*4+4]` as v4f16 (4 consecutive K elements)
- **B operand**: lane provides `B[b*4:b*4+4, n]` as v4f16 (4 elements from consecutive rows)
- K=16 per instruction

**Loading A from row-major LDS** (Q in flash attention):
```python
a_idx = row_q * K_STRIDE + ks * 16 + b * 4  # v4f16 from row
a_pack = vec_ext.load_op(v4f16_type, lds_q, [a_idx])
```

**Loading B from row-major LDS** (K in flash attention):
```python
b_idx = row_k * K_STRIDE + ks * 16 + b * 4  # same layout as A
b_pack = vec_ext.load_op(v4f16_type, lds_k, [b_idx])
```

### mfma_f32_32x32x16_fp8 (gfx950)
- For FP8 GEMM on MI355X
- K=16 per instruction, inputs = v8i8 (packed FP8)

### mfma_scale_f32_16x16x128_f8f6f4 (gfx950 fast path)
- K=128 per instruction for FP8 — reduces instruction count dramatically
- See `preshuffle_gemm.py` for usage

---

## 4. LDS Management

### Bank-conflict-free padding
AMD LDS has 32 banks. Add padding to make stride coprime with 32:
```python
# stride = size + 2 makes stride/2 odd → coprime with 32 banks
K_STRIDE = HEAD_DIM + 2    # for row-major Q/K in LDS
VT_STRIDE = BLOCK_N + 2    # for transposed V in LDS
```

### SmemAllocator
```python
from flydsl.utils import SmemAllocator

allocator = SmemAllocator(None, arch=gpu_arch)
_state["lds_q"] = allocator.allocate_array(elem_type, BLOCK_M * K_STRIDE)
_state["lds_kv"] = allocator.allocate_array(elem_type, BLOCK_N * K_STRIDE)
allocator.finalize()

# In kernel body:
base_ptr = allocator.get_base()
lds_q = _state["lds_q"](base_ptr).get()
```

### Ping-pong double buffering (separate allocators)
```python
allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
_state["lds_a_pong"] = allocator_pong.allocate_array(elem_type, size)
_state["lds_a_ping"] = allocator_ping.allocate_array(elem_type, size)
allocator_pong.finalize()
allocator_ping.finalize()

# In kernel: alternate between pong and ping each iteration
```

---

## 5. Cooperative Vectorized Tile Load

Critical for performance — use vectorized loads with full wave cooperation:

```python
VEC_WIDTH = 8  # v8f16 = 16 bytes per load
THREADS_PER_ROW = HEAD_DIM // VEC_WIDTH   # e.g., 128/8 = 16
ROWS_PER_BATCH = WARP_SIZE // THREADS_PER_ROW  # 64/16 = 4
NUM_BATCHES = BLOCK_M // ROWS_PER_BATCH

# Each thread computes its lane assignment once
load_row_in_batch = tid // THREADS_PER_ROW   # which row within batch
load_col_base = (tid % THREADS_PER_ROW) * VEC_WIDTH  # which column chunk

def coop_load_to_lds(src_memref, lds_memref, tile_start):
    for batch in range_constexpr(NUM_BATCHES):
        row = tile_start + batch * ROWS_PER_BATCH + load_row_in_batch
        g_idx = row * HEAD_DIM + load_col_base
        vec = vec_ext.load_op(v8f16_type, src_memref, [g_idx])
        lds_idx = (batch * ROWS_PER_BATCH + load_row_in_batch) * K_STRIDE + load_col_base
        vec_ext.store(vec, lds_memref, [lds_idx])
```

**Key insight**: Each thread loads one v8f16 (16B) per batch → 4-8x fewer memory ops than scalar.

---

## 6. Warp Shuffle Reductions

### Softmax row-wise reduction (16-wide within MFMA b-groups)
```python
# XOR shuffle offsets [8, 4, 2, 1] stay within 16-lane b-groups
for offset in [8, 4, 2, 1]:
    other = rocdl.ds_swizzle(val, offset)  # XOR shuffle
    val = max(val, other)  # or add for sum
```

### Block-wide reduction (from reduce.py)
```python
from kernels.reduce import make_block_reduce

block_reduce = make_block_reduce(tid, BLOCK_SIZE, compute_type, ...)
max_val = block_reduce(val, "max")
sum_val = block_reduce(val, "add")
# Uses: intra-wave XOR shuffle → inter-wave LDS partial sums → wave0 final reduce
```

---

## 7. Vector Operations

```python
from flydsl.dialects.ext import vector as vec_ext

# Load vector from memory
v = vec_ext.load_op(v8f16_type, memref, [idx])  # returns ArithValue!
v_raw = arith.as_value(v)

# Store vector
vec_ext.store(v, memref, [idx])  # auto-unwraps ArithValue

# Broadcast scalar to vector
v = vec_ext.broadcast(v4f32_type, scalar)  # returns ArithValue!

# Vector reduction (sum)
from mlir.dialects import vector as _vector
total = _vector.reduction(f32_type, "add", v_raw, fastmath=fm_fast)

# Extract element
elem = vec_ext.extract(v, static_position=[i], dynamic_position=[])

# Scalar load/store
val = _memref.LoadOp(memref, [idx]).result
_memref.StoreOp(val, memref, [idx])
```

---

## 8. Kernel Module Structure

```python
import flydsl as flir

class MyKernel(flir.MlirModule):
    GPU_MODULE_NAME = "my_kernel_f16"
    GPU_MODULE_TARGETS = ['#rocdl.target<chip = "gfx942", abi = "500">']

    def init_gpu_module(self):
        # Allocate LDS, set up _state dict
        allocator = SmemAllocator(None, arch="gfx942")
        self._state["lds"] = allocator.allocate_array(f16_type, LDS_SIZE)
        allocator.finalize()

    @flir.kernel
    def compute(self, A, B, C, M, N, K):
        # GPU kernel body — runs on device
        tid = gpu.thread_id("x")
        bid = gpu.block_id("x")
        # ... MFMA computation ...

    @flir.jit
    def __call__(self, A, B, C, M, N, K):
        # Host dispatch — launches kernel
        grid = (num_blocks,)
        block = (256,)
        flir.gpu_ext.LaunchFuncOp(self.compute, grid, block, [A, B, C, M, N, K])

# Compile and run
m = MyKernel()
exe = flydsl.compile(m)
exe(A_tensor, B_tensor, C_tensor, M, N, K)
```

---

## 9. Critical Pitfalls

### LLVM barrier reordering (SHOWSTOPPER)
- **Problem**: MLIR IR has correct order `barrier → LDS reads → barrier` but LLVM AMDGPU scheduler reorders LDS reads past barriers in final ISA
- **`rocdl.s_waitcnt(0)` has NO effect** — LLVM ignores it as a scheduling barrier
- **Cannot rely on LDS overlay with separate read/write phases**
- **Workaround**: Load from global memory instead of LDS when possible (Q-from-global pattern in V4.3). Or use ping-pong double buffering (no read/write overlap needed).

### AGPRs not accessible from MLIR
- MI300X has 256 VGPRs + 256 AGPRs but LLVM won't spill MFMA accumulators to AGPRs from MLIR
- `--greedy-reverse-local-assignment` flag has no effect
- Need inline ASM (HipKittens-style) for AGPR access
- **Impact**: Higher VGPR pressure → lower occupancy → can't use large tile sizes

### Don't nest MLIR op constructors
```python
# BAD — inner result not unwrapped
result = _arith.MulIOp(_arith.AddIOp(a, b).result, c)

# GOOD — unwrap intermediate results
sum_val = _arith.AddIOp(a, b).result
result = _arith.MulIOp(sum_val, c)
```

### INT4 unpacking
- No `v_perm` instruction available from MLIR
- Requires 7-op unpack sequence for INT4 (W4A8 packed)
- See `preshuffle_gemm.py` for the pattern

---

## 10. Flash Attention Optimization Progression

Each version added one optimization. Use as a roadmap for new kernels:

| Version | Change | Speedup | Key Technique |
|---------|--------|---------|--------------|
| V4.0 | Baseline | 26 TFLOPS | MFMA 16x16x16f16, 4 barriers/iter |
| V4.1 | Q in registers + V transposed LDS + bank-conflict padding | 2.3x | Eliminate redundant LDS traffic, fix bank conflicts |
| V4.2 | BLOCK_N=32 (was 16) | 1.1x | Halve K-loop iterations and barriers |
| V4.2+exit | Causal early-exit | 1.6x | Skip blocks where all Q positions < all K positions |
| V4.3 | Q from global (bypass LDS) | ~1x | Save LDS for K/V, works around barrier reordering |

**Remaining gap to ASM v3 (1012 TFLOPS, 4.7x):**
- Tile size: ASM uses BLOCK_M=256 vs FlyDSL BLOCK_M=64
- MFMA size: ASM uses mfma_32x32x8 vs FlyDSL mfma_16x16x16
- Double buffering: ASM ping-pongs K tiles in LDS
- XDL || softmax: ASM interleaves softmax ALU with MFMA execution
- AGPRs: ASM uses accumulator GPRs for higher occupancy (8 waves vs 4)

---

## 11. Reference Kernels

| Kernel | File | Patterns Demonstrated |
|--------|------|----------------------|
| Flash Attention V3 | `kernels/flash_attention_v3.py` | MFMA FA, cooperative loads, online softmax, causal masking, warp shuffle reductions |
| Preshuffle GEMM | `kernels/preshuffle_gemm.py` | MFMA+LDS pipeline, ping-pong, XOR16 swizzle, FP8/INT4/BF16, scale epilogue, scheduler hints |
| Reduce | `kernels/reduce.py` | `warp_reduce_sum`, `warp_reduce_max`, `block_reduce` |

## 12. Next Optimizations (MI350 Design Doc)

- K/V double-buffering in LDS (ping-pong: load next while computing current)
- S/P ping-pong in VGPRs (eliminate P→LDS roundtrip between GEMM1 and GEMM2)
- XDL || softmax co-execution (interleave softmax ALU with MFMA)
- Staggered wave groups (2 groups offset by half iteration for LDS BW sharing)
- V_PERMLANE16/32_SWAP for reductions (instead of XOR shuffle)
