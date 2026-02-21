---
name: mfma-kernel-patterns
description: Technical reference for AMD MFMA instruction patterns in FlyDSL kernels. Thread mapping, LDS layouts, cooperative loads, softmax, and P@V GEMM patterns.
argument-hint: "<topic, e.g. 'mfma thread mapping', 'bank-conflict-free LDS', 'cooperative load', 'flash attention softmax'>"
---

# MFMA Kernel Patterns for FlyDSL

Reference for writing high-performance MFMA-based GPU kernels in FlyDSL on AMD MI300X (GFX942). Distilled from Flash Attention V4.x development.

Based on `$ARGUMENTS`, find the relevant section below and provide the pattern with FlyDSL code.

---

## 1. MFMA Thread Mapping — `mfma_f32_16x16x16f16`

### Instruction semantics
- **C[16,16] += A[16,16] × B[16,16]** in f16→f32
- K=16 elements consumed per instruction
- Inputs: A, B as `v4f16`, output/accumulator C as `v4f32`

### Thread-to-element mapping (64 threads per wavefront)
```
lane = thread_id % 64
b = lane // 16       # b ∈ {0,1,2,3} — "batch" within the 16x16 tile
n = lane % 16        # n ∈ {0,1,...,15} — column index

C output:  lane owns C[b*4+ii, n] for ii=0,1,2,3  → v4f32
A operand: lane provides A[n, b*4 : b*4+4]        → v4f16 (4 consecutive K elements)
B operand: lane provides B[b*4 : b*4+4, n]        → v4f16 (4 elements from consecutive rows)
```

### FlyDSL invocation
```python
from flydsl.dialects.ext import rocdl, arith, vec_ext

v4f16_type = VectorType.get([4], F16Type.get())
v4f32_type = VectorType.get([4], F32Type.get())

# acc = A @ B + acc
acc = arith.as_value(
    rocdl.mfma_f32_16x16x16f16(
        v4f32_type, [a_pack, b_pack, acc, 0, 0, 0]
    )
)
# Last 3 args: cbsz=0, abid=0, blgp=0 (no broadcast)
```

### Loading A-operand from row-major LDS
A[n, b*4:b*4+4] — thread needs 4 consecutive elements from row `n`, starting at column `b*4`:
```python
# Q stored row-major in LDS: Q[row, col] at lds[row * STRIDE + col]
a_lds_idx = (
    arith.ArithValue(lane_mod_16) * STRIDE   # row = n = lane%16
    + ks * 16                                  # K-step offset
    + arith.ArithValue(lane_div_16) * 4        # col = b*4
).value
a_pack = arith.as_value(vec_ext.load_op(v4f16_type, lds, [a_lds_idx]))
```

### Loading B-operand from row-major LDS
B[b*4:b*4+4, n] — thread needs 4 elements from 4 consecutive rows at column `n`:
```python
# K stored row-major: K[row, col] at lds[row * K_STRIDE + col]
# Need rows b*4, b*4+1, b*4+2, b*4+3 at column n
# These are NOT contiguous in memory → 4 scalar loads (slow!)
# Solution: transpose in LDS (see Section 4)
```

### Loading B-operand from TRANSPOSED LDS (preferred)
V stored as V^T: V[row,col] at lds[col * VT_STRIDE + row]:
```python
# B-operand for P@V: need V^T[b*4:b*4+4, n] which is contiguous
v_lds_idx = (
    (ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE  # col = ds*16+n
    + arith.ArithValue(lane_div_16) * 4                      # row = b*4
).value
v_pack = arith.as_value(vec_ext.load_op(v4f16_type, lds_kv, [v_lds_idx]))
```

---

## 2. Bank-Conflict-Free LDS Padding

### Problem
LDS has 32 banks, each 4 bytes wide. For f16 (2 bytes), 2 elements per bank.
When stride is a multiple of 64 elements (e.g., HD=128), all threads in a wavefront
accessing the same column hit the same bank → 32-way conflict.

### Solution: Pad stride so stride/2 is odd
```python
PAD = 2  # minimum padding for f16
K_STRIDE = HEAD_DIM + PAD   # e.g., 128+2=130, 130/2=65 (odd, coprime with 32)
VT_STRIDE = BLOCK_N + PAD   # e.g., 32+2=34, 34/2=17 (odd, coprime with 32)
```

### Why it works
- Bank index = (byte_offset / 4) % 32
- For f16 at lds[row * stride + col]: byte_offset = (row * stride + col) * 2
- Bank = (row * stride + col) / 2 % 32
- If stride/2 is odd, consecutive rows map to different banks (stride/2 mod 32 ≠ 0)
- 65 mod 32 = 1 → perfect stride for HD=128

### LDS allocation
```python
LDS_Q_SIZE = BLOCK_M * K_STRIDE           # Q tile with padding
LDS_KV_SIZE = max(
    BLOCK_N * K_STRIDE,                    # K tile (row-major, padded)
    HEAD_DIM * VT_STRIDE                   # V^T tile (transposed, padded)
)
# Total LDS = (LDS_Q_SIZE + LDS_KV_SIZE) * sizeof(f16)
```

---

## 3. Cooperative Tile Load (Global → LDS)

### Pattern: 256 threads loading a [ROWS, COLS] tile with vectorized loads

```python
VEC_WIDTH = 8                                    # v8f16 = 16 bytes per load
THREADS_PER_ROW = HEAD_DIM // VEC_WIDTH          # e.g., 128/8 = 16
ROWS_PER_BATCH = NUM_THREADS // THREADS_PER_ROW  # e.g., 256/16 = 16
NUM_BATCHES = BLOCK_M // ROWS_PER_BATCH          # e.g., 64/16 = 4

# Thread assignment
load_col_base = (thread_in_block % THREADS_PER_ROW) * VEC_WIDTH
load_row_in_batch = thread_in_block // THREADS_PER_ROW

# Load loop
for batch in range_constexpr(NUM_BATCHES):
    row = tile_start + load_row_in_batch + batch * ROWS_PER_BATCH
    g_idx = row * total_cols + load_col_base
    vec = arith.as_value(vec_ext.load_op(v8f16_type, global_mem, [g_idx]))
    lds_idx = (load_row_in_batch + batch * ROWS_PER_BATCH) * STRIDE + load_col_base
    vec_ext.store(vec, lds, [lds_idx])
```

### Bounds checking (when tile may exceed data bounds)
```python
from flydsl.dialects.ext.scf import IfOp
row_ok = arith.as_value(
    flir.arith.CmpIOp(flir.arith.CmpIPredicate.ult, row, num_rows).result
)
if_op = IfOp(row_ok)
with if_op:
    # load and store only if row is valid
```

### Cooperative transposed store (for V^T in LDS)
```python
# Load V[row, col] from global, store as V^T[col, row] in LDS
for batch in range_constexpr(NUM_BATCHES_V):
    row = tile_start + load_row_in_batch + batch * ROWS_PER_BATCH
    g_idx = row * HEAD_DIM + load_col_base
    vec = arith.as_value(vec_ext.load_op(v8f16_type, V_global, [g_idx]))
    # Scatter-store: extract each element, store transposed
    for e in range_constexpr(VEC_WIDTH):
        elem = arith.as_value(vec_ext.extract(vec, static_position=[e], dynamic_position=[]))
        col_e = load_col_base + e
        lds_idx = col_e * VT_STRIDE + (load_row_in_batch + batch * ROWS_PER_BATCH)
        _memref.StoreOp(elem, lds_kv, [lds_idx])
```

---

## 4. Online Softmax with Warp Shuffle

### Per-row max and sum across BLOCK_N positions (16 or 32 wide)

For `mfma_16x16x16f16`, each output row is owned by threads within the same b-group
(lanes b*16..b*16+15). So 16-wide reduction stays within 16 consecutive lanes.

### XOR shuffle pattern for 16-wide reduction
```python
SHUFFLE_OFFSETS = [8, 4, 2, 1]  # log2(16) = 4 steps

def warp_reduce_max_16(val):
    """Reduce max across 16 lanes within b-group."""
    result = val
    for offset in SHUFFLE_OFFSETS:
        other = ds_swizzle(result, offset)  # XOR shuffle
        result = max(result, other)
    return result
```

### FlyDSL implementation
```python
from flydsl.dialects.ext.rocdl import ds_swizzle_val

def reduce_max_16(val_v):
    """val_v is v4f32 — reduce max across 4 elements, then across 16 lanes."""
    # First reduce within v4f32 (4 row elements per thread)
    m = vec_ext.extract(val_v, static_position=[0], ...)
    for i in [1, 2, 3]:
        ei = vec_ext.extract(val_v, static_position=[i], ...)
        m = arith.maximumf(m, ei)
    # Then reduce across 16 lanes
    for offset in [8, 4, 2, 1]:
        swizzle_pattern = offset | (0x1F << 5) | (0x1F << 10)
        other = ds_swizzle_val(m, swizzle_pattern)
        m = arith.maximumf(m, other)
    return m
```

### Softmax update with running max (online softmax)
```python
# For each KV tile iteration:
# 1. Compute S = Q @ K^T (get s_acc as v4f32)
# 2. Row max of S tile
m_new = reduce_max_16(s_acc)
m_cur = arith.maximumf(m_old, m_new)

# 3. Rescale previous: correction = exp(m_old - m_cur)
correction = exp2f((m_old - m_cur) * LOG2E)
l_new = l_old * correction

# 4. Compute P = exp(S - m_cur), accumulate l_new += sum(P)
# 5. Rescale output: o_acc *= correction, then o_acc += P @ V
```

### BLOCK_N=32: Two-part softmax
When BLOCK_N=32, Q@K^T produces two v4f32 accumulators (lo: cols 0-15, hi: cols 16-31):
```python
# Separate lo/hi max
m_lo = reduce_max_16(s_accs[0])  # max over cols 0-15
m_hi = reduce_max_16(s_accs[1])  # max over cols 16-31
m_new = arith.maximumf(m_lo, m_hi)  # combined max
# Then apply softmax with combined max to both halves
```

---

## 5. P@V GEMM Pattern

### Intermediate result P (softmax output) as A-operand for P@V

P is [BLOCK_M_PER_WAVE, BLOCK_N] in f16. Must be stored in LDS in A-operand format.

### Store P to LDS (from v4f32 softmax output)
```python
P_STRIDE = BLOCK_N  # or BLOCK_N + PAD for bank-conflict-free

# P is in s_acc (v4f32), need to convert to f16 and store row-major
# Thread lane owns rows [b*4+0..b*4+3] at column n
for ii in range(4):
    p_val_f32 = vec_ext.extract(p_v4f16_or_f32, ...)
    p_val_f16 = arith.truncf(p_val_f32, F16Type)
    row = wave_p_offset + b * 4 + ii
    col = n
    lds_idx = row * P_STRIDE + col
    _memref.StoreOp(p_val_f16, lds_p, [lds_idx])

# For BLOCK_N=32: store lo half at cols 0..15, hi half at cols 16..31
```

### Reload P as A-operand for P@V MFMA
```python
# A-operand: need P[n, b*4:b*4+4] — row n, 4 consecutive cols
p_lds_idx = (
    arith.ArithValue(lane_mod_16) * P_STRIDE   # row = n
    + ds * 16                                    # K-step within P
    + arith.ArithValue(lane_div_16) * 4          # col = b*4
).value
p_pack = arith.as_value(vec_ext.load_op(v4f16_type, lds_p, [p_lds_idx]))
```

### BLOCK_N=32 P@V: Split K dimension
P is [16, 32], V^T is [32, HD]. Split K=32 into two K=16 steps:
```python
for ds in range_constexpr(K_STEPS):  # K_STEPS = HD/16
    # P_lo[16,16] @ V_top[16, ds*16:ds*16+16]
    p_lo_pack = load_p_a_operand(col_offset=0, ds_k=0)
    v_top_pack = load_v_b_operand(ds=ds, row_offset=0)
    o_accs[ds] = mfma(p_lo_pack, v_top_pack, o_accs[ds])

    # P_hi[16,16] @ V_bot[16, ds*16:ds*16+16]
    p_hi_pack = load_p_a_operand(col_offset=16, ds_k=1)
    v_bot_pack = load_v_b_operand(ds=ds, row_offset=16)
    o_accs[ds] = mfma(p_hi_pack, v_bot_pack, o_accs[ds])
```

---

## 6. Causal Masking Patterns

### Causal early-exit (tile-level)
Skip entire KV tiles beyond the causal boundary:
```python
# q_start = tile_idx * BLOCK_M
# Last Q row in this tile = q_start + BLOCK_M - 1
# Only need KV positions 0 .. q_start + BLOCK_M - 1
if CAUSAL:
    kv_upper = (arith.ArithValue(q_start) + BLOCK_M).value
else:
    kv_upper = seq_len

with scf.for_(0, kv_upper, BLOCK_N, iter_args=init_args) as loop:
```

### Within-tile causal mask (for the boundary tile)
When `kv_start + j > q_start + i`, mask S[i,j] = -inf:
```python
if CAUSAL:
    # For each row ii owned by this thread:
    q_row = q_start + wave_offset + b * 4 + ii
    # For each col n:
    kv_col = kv_start + n  # (or kv_start + nm*16 + n for BLOCK_N=32)
    is_masked = kv_col > q_row
    if is_masked:
        s_val = -infinity
```

---

## 7. Output Write-Back Pattern

### Rescale and write output from accumulators

After the KV loop completes, output accumulators need final rescaling by `1/l`:
```python
# o_accs[ds] is v4f32 for ds in range(K_STEPS)
# l_final is the softmax denominator (f32, broadcast across 16 lanes)

for ds in range_constexpr(K_STEPS):
    # Rescale: o_accs[ds] /= l_final
    inv_l = arith.divf(c_one_f, l_final)
    inv_l_vec = vec_ext.broadcast(v4f32_type, inv_l)
    o_scaled = arith.mulf(o_accs[ds], inv_l_vec)

    # Convert f32 → f16
    o_f16 = arith.truncf(o_scaled, v4f16_type)

    # Write to global: thread owns O[b*4+ii, ds*16 + ...] for ii=0..3
    # Store as v4f16 (4 consecutive output elements per thread)
    for ii in range(4):
        o_row = q_start + wave_offset + b * 4 + ii
        o_col = ds * 16 + n  # but need to handle 4-element packing
        g_idx = (batch * S * H * D) + (o_row * H * D) + (head * D) + (ds * 16 + b * 4)
        # Store individual f16 elements or use vectorized store
```

---

## 8. Q-from-Global Memory Pattern (V4.3)

### Problem: LDS overlay for Q is unreliable
LLVM AMDGPU scheduler reorders `ds_read` past `s_barrier`, breaking read/write phase separation.
**Solution**: Load Q directly from global memory into MFMA registers, bypassing LDS entirely.

### Thread-to-Q mapping for `mfma_f32_16x16x16f16`
Each lane needs Q[n, b*4:b*4+4] as A-operand (v4f16):
```python
# q_row = which Q row this lane reads from
q_row = (
    arith.ArithValue(q_start)      # tile start
    + arith.ArithValue(wave_q_offset)  # wave's row offset within tile
    + arith.ArithValue(lane_mod_16)    # n = lane % 16
).value

# Pre-load all K_STEPS Q packs before KV loop
q_a_packs = []
for ks in range_constexpr(K_STEPS):
    q_col = flir.const_index(ks * 16 + 0)
    q_col = (arith.ArithValue(q_col) + arith.ArithValue(lane_div_16) * 4).value
    g_idx = global_idx(q_row, q_col)  # q_row * HEAD_DIM + q_col
    q_a_packs.append(arith.as_value(
        vec_ext.load_op(v4f16_type, Q, [g_idx])
    ))
```

### Memory access pattern
- 4 threads with same `lane_mod_16` (different `b` values) access same row, adjacent 8-byte chunks
- Partially coalesced: 4×8=32 bytes per group of 4 threads
- Each wave issues K_STEPS × 64 loads (e.g., 8 × 64 = 512 loads for HD=128)

### LDS savings
- V4.2: lds_q (BLOCK_M × K_STRIDE × 2 bytes) + lds_kv + lds_p = ~23KB
- V4.3: lds_kv + lds_p = ~12.5KB (Q eliminated entirely)

---

## 9. K/V Double-Buffering Pattern (from MI350 FMHA Design)

### Concept: Ping-pong K/V in LDS
Allocate 2× LDS for K/V tiles. While computing GEMM on buffer A, prefetch next K/V into buffer B.

```
Init: Load K[0]→buf_A, Load V[0]→buf_A, Load K[1]→buf_B
Loop iteration i:
  Compute: GEMM1(Q, K[buf_i%2]) → S → softmax → P
  Prefetch: Load V[i]→buf_{(i+1)%2}, Load K[i+1]→buf_{(i+1)%2}
  Compute: GEMM2(P, V[buf_i%2]) → O
```

### LDS budget
```python
# Each buffer: K[BLOCK_N, HD] + V^T[HD, BLOCK_N] (or combined KV region)
LDS_PER_BUFFER = BLOCK_N * K_STRIDE + HEAD_DIM * VT_STRIDE  # in f16 elements
LDS_TOTAL = 2 * LDS_PER_BUFFER * 2  # 2 buffers × 2 bytes/f16
# For BN=32, HD=128, K_STRIDE=130, VT_STRIDE=34:
# = 2 * (32*130 + 128*34) * 2 = 2 * (4160+4352) * 2 = 34048 bytes (~33KB)
```

### Key: No barrier between compute and prefetch
The prefetch into buffer B doesn't conflict with reads from buffer A.
Only need barrier when SWITCHING buffers (before next iteration's compute phase).

---

## 10. S/P Ping-Pong in VGPRs (from MI350 FMHA Design)

### Problem: P→LDS→VGPR roundtrip
Current V4.x stores P to LDS after softmax, then reloads as A-operand for GEMM2. Costs ~2 barriers + LDS traffic.

### Solution: Keep P in VGPRs with transposed GEMM ordering
Use **K^T × Q^T** (not Q × K^T) for GEMM1 so the result S is already in A-operand format for GEMM2.

```
GEMM1: S = K^T[HD, BN] × Q^T[BM, HD] → S[BN, BM]  (S in accumulator = A-operand layout)
Softmax: S → P  (in-place, stays in VGPRs)
GEMM2: O += V^T[HD, BN] × P[BN, BM] → O[HD, BM]   (P directly used as B-operand)
```

### VGPR cost for S/P ping-pong
Need 2× S/P storage: current iteration's S (being computed) + previous iteration's P (being consumed by GEMM2).
```
SP_VGPRs = 2 * (BLOCK_N/16) * 4 * (BLOCK_M_PER_WAVE/16) = 2 * 2 * 4 * 1 = 16 VGPRs (BN=32, BM_wave=16)
```

---

## 11. XDL || Softmax Co-Execution Pipeline (from MI350 FMHA Design)

### Concept
On GFX942+, MFMA (XDL unit) and VALU can execute simultaneously. Interleave:
- GEMM2 MFMA instructions (XDL) with softmax VALU from current iteration
- GEMM1 MFMA instructions (XDL) with output rescaling VALU from previous iteration

### Pipeline structure (per iteration)
```
Phase 1: XDL=GEMM1(K^T×Q^T→S)   || VALU=rescale_output(prev correction)
Phase 2: XDL=GEMM2(V^T×P_prev)  || VALU=softmax(S→P_curr)
```

### Requirements
- S/P ping-pong (Section 10) — so current softmax doesn't conflict with GEMM2 input
- Careful instruction interleaving in generated code (alternate MFMA and VALU instructions)

---

## 12. LLVM Scheduler Pitfalls

### LLVM reorders LDS reads past barriers
**Confirmed behavior**: The LLVM AMDGPU instruction scheduler can move `ds_read` (LDS read) instructions past `s_barrier` synchronization points. This was confirmed by comparing:
- MLIR IR stage 07 (gpu-kernel-outlining): barrier1 → Q reads → barrier2 → scf.for (correct)
- Final ISA: barrier1 → barrier2 → Q reads → K writes (broken — no barrier between Q reads and K writes)

### Workarounds that DON'T work
- `rocdl.s_waitcnt(0)` before barrier — LLVM ignores it as scheduling fence
- Multiple barriers in sequence — LLVM coalesces/reorders around them

### What DOES work
- Eliminate the need for ordering: don't overlay LDS regions that need read-before-write protection
- Use separate LDS allocations (no overlap) + standard barriers
- Q-from-global (bypass LDS entirely for read-once data)

---

## 13. Common FlyDSL Pitfalls

### ArithValue wrapping
- `scf.for_` inner_iter_args and induction_variable return ArithValue wrappers
- Always unwrap: `arith.as_value(loop.inner_iter_args[i])`, `arith.as_value(loop.induction_variable)`
- `vec_ext.load_op()` returns ArithValue → unwrap before passing to `rocdl.mfma_*`
- `vec_ext.broadcast()` returns ArithValue → unwrap before `arith.mulf`

### range vs range_constexpr
- Inside `@flir.kernel` bodies, `range_constexpr()` unrolls at compile time (required for static indexing)
- Plain `range()` generates `scf.for` IR loops (dynamic, has overhead)
- Use `range_constexpr()` for MFMA loops, K_STEPS, VEC_WIDTH iterations

### scf.for_ iter_args pattern
```python
init_args = [val1, val2, ...]
with scf.for_(lb, ub, step, iter_args=init_args) as loop:
    # Unwrap
    a = arith.as_value(loop.inner_iter_args[0])
    b = arith.as_value(loop.inner_iter_args[1])
    # ... compute new_a, new_b ...
    scf.yield_([new_a, new_b])
# After loop:
result_a = arith.as_value(loop.results[0])
result_b = arith.as_value(loop.results[1])
```

### Don't nest MLIR op constructors
```python
# BAD: nested constructors may fail
result = arith.mulf(vec_ext.broadcast(...), some_val)

# GOOD: unwrap intermediate results
broadcast_val = arith.as_value(vec_ext.broadcast(...))
result = arith.mulf(broadcast_val, some_val)
```

---

## 14. Resource Budget (MI300X / GFX942)

| Resource | Per CU | Per SIMD (4/CU) | For 2 waves | For 4 waves |
|----------|--------|-----------------|-------------|-------------|
| LDS | 64 KB | 16 KB | 32 KB/wg | 16 KB/wg |
| Arch VGPRs | 512 | 512 | 256/wave | 128/wave |
| Accum VGPRs | 512 | 512 | 256/wave | 128/wave |
| SGPRs | 800 | — | 400/wave | 200/wave |

### Flash Attention V4.x resource usage
| Version | arch_vgpr | accum_vgpr | sgpr | LDS | Waves/SIMD |
|---------|-----------|------------|------|-----|------------|
| V4.2 | 120 | 56 | 112 | 23040 B | 2 (LDS-limited) |
| V4.3 | 124 | 0 | ~112 | 12800 B | 4 (VGPR-limited) |

- V4.3 reduced LDS below 16KB threshold but VGPR count (124) still limits to 4 waves
- No AGPRs used (LLVM allocates all accumulators in arch VGPRs)
- To use AGPRs: need HipKittens-style inline assembly (not available from MLIR path)
