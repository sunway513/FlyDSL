---
name: gpu-kernel-optimize
description: Systematic GPU kernel optimization workflow for FlyDSL kernels on AMD MI300X. Profile, identify bottleneck, apply targeted fix, measure, iterate.
argument-hint: "<kernel-name and target config, e.g. 'flash_attention B=1 H=64 D=128 S=8192'>"
---

# GPU Kernel Optimization Workflow

Systematic process for optimizing FlyDSL GPU kernels on AMD MI300X (GFX942). This skill encodes the methodology proven during Flash Attention optimization (V4.0→V4.2+exit: 26→108 TFLOPS, 4.1x improvement).

## Core Principle: Profile-Driven Optimization

**NEVER guess what to optimize.** Always measure first, identify the dominant bottleneck, fix it, then re-measure. Each iteration should target a single bottleneck category.

## Step 1: Establish Baseline

Run the kernel benchmark at the target configuration from `$ARGUMENTS`:

```bash
cd /home/pensun/FlyDSL
python3 tests/kernels/test_<kernel>.py --batch B --num_heads H --head_dim D --seq_len S --warmup 10 --iters 50
```

Record: time (us), TFLOPS, correctness (max_err, cos_sim).

If a reference implementation exists (ASM, CK, PyTorch), also measure it. **Ensure FLOP formulas match** — a common pitfall is comparing causal TFLOPS (÷2) against non-causal TFLOPS.

## Step 2: Profile with rocprof

Create counter config files. Use **two passes** (rocprof v2 limits ~8 counters per pass):

**Pass 1** — Instruction mix:
```
pmc: SQ_WAVES SQ_INSTS_VALU SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR SQ_INSTS_LDS SQ_INSTS_VALU_MFMA_MOPS_F16 SQ_BUSY_CU_CYCLES GRBM_GUI_ACTIVE
```

**Pass 2** — Latency/stalls:
```
pmc: SQ_WAVES SQ_INSTS_SALU SQ_INSTS_SMEM SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_LEVEL_WAVES SQ_WAIT_INST_LDS
```

Run profiling:
```bash
rocprof --pmc-file /tmp/counters1.txt -o /tmp/prof1.csv python3 tests/kernels/test_<kernel>.py <args>
rocprof --pmc-file /tmp/counters2.txt -o /tmp/prof2.csv python3 tests/kernels/test_<kernel>.py <args>
```

**Important**: Use supported GFX942 counters only. Avoid TCC_HIT and other unsupported counters.

## Step 3: Analyze Profile — Identify Bottleneck Category

Compute per-wave metrics by dividing raw counters by SQ_WAVES. Key ratios:

### Bottleneck identification decision tree:

```
SQ_WAIT_INST_LDS / SQ_ACTIVE_INST_VALU > 1.5?
├── YES → LDS-BOUND (latency or bandwidth)
│   ├── High SQ_INSTS_LDS per wave? → Reduce LDS traffic (see Optimization Menu)
│   └── Low SQ_INSTS_LDS but high wait? → Bank conflicts (see Optimization Menu)
│
└── NO → Check MFMA utilization
    ├── SQ_INSTS_VALU_MFMA_MOPS / SQ_ACTIVE_INST_VALU < 0.5? → Instruction overhead
    │   └── Too many non-MFMA instructions between MFMAs (barriers, address calc)
    │
    └── SQ_LEVEL_WAVES < 2? → Occupancy-bound
        └── Check: LDS per workgroup (need <16KB for 4 waves/SIMD on MI300X)
            VGPRs per wave (need <128 arch_vgpr for 4 waves/SIMD)
```

### Key thresholds (MI300X):
- **LDS**: 64KB per CU, 4 SIMDs per CU → 16KB per SIMD for 4 waves
- **VGPRs**: 512 per SIMD → 128 per wave for 4 waves
- **Accum VGPRs**: 512 per SIMD → 128 per wave for 4 waves
- **Barriers**: Each `s_barrier` costs ~100 cycles. Minimize barriers per iteration.

## Step 4: Apply Targeted Optimization

See the **Optimization Menu** below. Pick the optimization that addresses the identified bottleneck.

## Step 5: Verify and Iterate

1. **Correctness first**: Run full test suite. Must pass (max_err < 1e-2, cos_sim > 0.99)
2. **Benchmark**: Compare old vs new at same config
3. **Re-profile**: Verify the bottleneck shifted (e.g., LDS wait ratio dropped)
4. **Update memory**: Record what changed and the measured impact
5. **Go to Step 3**: Identify the new dominant bottleneck

---

## Optimization Menu

### A. Reduce LDS Traffic

**Problem**: High SQ_INSTS_LDS and SQ_WAIT_INST_LDS per wave.

| Technique | When to Use | Expected Impact |
|-----------|-------------|-----------------|
| **Q-in-registers** | Q is re-read from LDS every KV iteration | Eliminates N×K_STEPS LDS reads from inner loop. FA V4.0→V4.1: LDS ops 40K→32K/wave |
| **Increase BLOCK_N** | Many small iterations with barrier overhead | Halves iterations and barriers. FA V4.1→V4.2: BLOCK_N=16→32, 1.12x at S=8192 |
| **Vectorized LDS loads** | Scalar LDS loads for B-operand of MFMA | Store data transposed so B-operand is contiguous → v4f16 load. FA V4.0→V4.1: key enabler |

**Implementation pattern for Q-in-registers**:
```python
# Before KV loop: preload Q A-operand packs from LDS
q_a_packs = []
for ks in range_constexpr(K_STEPS):
    q_lds_idx = <compute from wave offset, lane, ks>
    q_a_packs.append(arith.as_value(
        vec_ext.load_op(v4f16_type, lds_q, [q_lds_idx])
    ))
# Inside KV loop: use q_a_packs[ks] directly, no LDS reads for Q
```

**Implementation pattern for transposed V in LDS**:
```python
# V[row, col] stored at lds[col * VT_STRIDE + row] (transposed)
# Cooperative store: extract each element, scatter-store
for e in range_constexpr(VEC_WIDTH):
    elem = arith.as_value(vec_ext.extract(vec, static_position=[e], ...))
    col_e = (arith.ArithValue(load_col_base) + e).value
    lds_idx = (arith.ArithValue(col_e) * VT_STRIDE + arith.ArithValue(load_row)).value
    _memref.StoreOp(elem, lds_kv, [lds_idx])

# B-operand load: now contiguous v4f16
v_lds_idx = ((ds * 16 + arith.ArithValue(lane_mod_16)) * VT_STRIDE
             + arith.ArithValue(lane_div_16) * 4).value
v_pack = arith.as_value(vec_ext.load_op(v4f16_type, lds_kv, [v_lds_idx]))
```

### B. Fix Bank Conflicts

**Problem**: Low SQ_INSTS_LDS but high SQ_WAIT_INST_LDS — bank conflicts.

| Technique | Formula | Example |
|-----------|---------|---------|
| **Padding** | stride = size + PAD, where (stride/element_size) is odd | K_STRIDE = HD+2 = 130 (130/2=65, odd, coprime with 32 banks) |
| **Swizzle** | XOR-based address permutation | `swizzle_xor16` for 16-element rotations |

**Why PAD=2 works**: LDS has 32 banks, each 4 bytes. For f16 (2 bytes), two elements per bank. stride/2 must be odd (coprime with 32) so consecutive rows map to different banks. HD=128 → 128/2=64 (even, conflicts!) → (128+2)/2=65 (odd, no conflicts).

### C. Reduce Barrier Overhead

**Problem**: Many barriers per iteration eating into useful compute.

| Technique | When | Impact |
|-----------|------|--------|
| **Increase BLOCK_N** | Barriers are per-iteration | 2x BLOCK_N → half iterations → half barriers |
| **Fuse loads** | Separate K and V loads with barrier between | Single load phase + single barrier |
| **Double-buffering (ping-pong)** | Load next tile while computing current | Eliminates load→compute barrier chain |

### D. Reduce Wasted Compute

**Problem**: Kernel computes results that are discarded (e.g., causal masking).

| Technique | When | Impact |
|-----------|------|--------|
| **Causal early-exit** | Causal attention iterates past mask boundary | Loop to `q_start + BLOCK_M` instead of `seq_len`. ~1.6x at S=8192 (theoretical 2x) |
| **Tile-level skip** | Entire tiles produce zero output | Skip tiles where `kv_start > q_end` |

**Implementation**:
```python
if CAUSAL:
    kv_upper = (arith.ArithValue(q_start) + BLOCK_M).value
else:
    kv_upper = seq_len_v
with scf.for_(0, kv_upper, BLOCK_N, iter_args=init_args) as loop:
```

### E. Increase Occupancy

**Problem**: SQ_LEVEL_WAVES < 2, low SIMD utilization.

| Resource | Limit for 4 waves/SIMD | How to Reduce |
|----------|----------------------|---------------|
| LDS | < 16KB per workgroup | Reduce tile sizes, recompute instead of store |
| Arch VGPRs | < 128 per wave | Reduce live register pressure, spill to LDS |
| Accum VGPRs | < 128 per wave | Smaller MFMA accumulator tiles |

### F. Scale Up Tile Size

**Problem**: Per-tile overhead (launch, address calc) dominates small tiles.

| Technique | Tradeoff |
|-----------|----------|
| **BLOCK_M=128** | 2x Q rows per tile → halves tile count, better Q amortization. Needs 8 waves. |
| **BLOCK_M=256** | 4x Q rows (ASM uses this). Needs careful register/LDS management. |

### G. Use Larger MFMA

| Current | Target | Benefit |
|---------|--------|---------|
| `mfma_16x16x16f16` | `mfma_32x32x8f16` | 4x more output elements per instruction, fewer loop iterations |

**Tradeoff**: 32x32 MFMA uses more accum VGPRs (16×v4f32 = 64 VGPRs vs 4×v4f32 = 16 VGPRs).

### H. Q-from-Global Memory (bypass LDS for Q)

**Problem**: Q in LDS wastes capacity when Q is only read once per KV loop.

Load Q directly from global memory into MFMA registers. Each lane loads its v4f16 A-operand:
```python
# Each MFMA lane loads Q[q_row, ks*16 + b*4 : +4] directly from global
q_row = q_start + wave_q_offset + lane_mod_16
for ks in range_constexpr(K_STEPS):
    q_col = ks * 16 + lane_div_16 * 4
    g_idx = q_row * HEAD_DIM + q_col
    q_a_packs[ks] = vec_ext.load_op(v4f16_type, Q_global, [g_idx])
```

**Impact**: Eliminates Q LDS (e.g., 16KB → 0). Partially coalesced (4 threads share 8-byte chunks).
**Caveat**: No perf gain if VGPRs are the occupancy bottleneck (V4.3 lesson).

### I. K/V Double-Buffering in LDS (from MI350 FMHA design)

**Problem**: Stall waiting for global→LDS load to complete before computing.

Allocate 2x LDS for K/V. Ping-pong between buffers:
```
Iteration i: Compute GEMM on buffer[i%2], Load next K/V into buffer[(i+1)%2]
```
**Requires**: 2x LDS for KV tiles. On MI300X with 64KB LDS, feasible for small tiles.

### J. S/P Ping-Pong in VGPRs (from MI350 FMHA design)

**Problem**: P→LDS→VGPR roundtrip between GEMM1 (Q@K^T→S→P) and GEMM2 (P@V).

Keep P in VGPRs. GEMM1 produces S in accumulators → softmax → P stays in registers.
Use GEMM ordering **K^T × Q^T** (not Q × K^T) so P is in A-operand format for GEMM2.
**Key insight**: Transposed GEMM ordering matches MFMA data layout, avoiding P LDS roundtrip.

### K. XDL || Softmax Co-Execution (from MI350 FMHA design)

**Problem**: Softmax ALU runs sequentially after GEMM1, wasting MFMA execution units.

On GFX942+, MFMA (XDL) and VALU can co-execute. Interleave softmax ALU with GEMM2 MFMA:
```
Step 1: GEMM1(K^T × Q^T) → S in accumulators
Step 2: Softmax ALU on S [overlapped with] GEMM2(V^T × P) from PREVIOUS iteration
```
**Requires**: S/P ping-pong (technique J) to overlap current softmax with previous GEMM2.

---

## Pitfalls and Lessons Learned

1. **FLOP formula mismatch**: ASM benchmarks often use non-causal formula `4*H*S*S*D` even for causal kernels. Our benchmarks use causal formula `4*H*S*(S/2)*D`. Always normalize before comparing. The "gap" can appear 2x worse than reality.

2. **rocprof counter support**: Not all counters work on all architectures. `TCC_HIT` is unsupported on GFX942. If profiling fails, check counter names against `rocprof --list-counters`.

3. **Two-pass profiling**: rocprof v2 limits counters per pass. Split into instruction-mix pass and latency/stall pass. Always include SQ_WAVES in both for per-wave normalization.

4. **ArithValue wrapping**: FlyDSL's `scf.for_` returns ArithValue-wrapped iter_args. Always unwrap with `arith.as_value()` before passing to MLIR op constructors. The `induction_variable` also needs unwrapping.

5. **Barrier count matters**: Each `s_barrier`/`gpu.barrier()` costs ~100 cycles. With 512 iterations and 4 barriers/iter, that's 200K wasted cycles. Halving iterations (larger BLOCK_N) directly halves this.

6. **Causal early-exit gets < 2x**: Theoretical is 2x (skip half the KV positions on average). Practical is ~1.6x due to launch overhead and the smallest tiles (near diagonal) doing minimal work.

7. **Profile the right kernel**: rocprof CSV may contain multiple kernel entries (compilation helpers, etc.). Find the actual kernel by name (e.g., `flash_attention_v4_1_kernel.kd`) and use its row.

8. **LLVM AMDGPU scheduler reorders across barriers**: The LLVM backend can move `ds_read` (LDS read) instructions past `s_barrier` synchronization points, breaking intended LDS overlay patterns. Confirmed by comparing MLIR IR (correct order) vs final ISA (reordered). `rocdl.s_waitcnt(0)` does NOT act as a scheduling barrier — LLVM ignores it. **Never rely on LDS overlay with separate read/write phases unless you can prove LLVM won't reorder.**

9. **AGPRs inaccessible from MLIR path**: MI300X has 256 arch VGPRs + 256 accum VGPRs (AGPRs). The LLVM flag `--greedy-reverse-local-assignment` has no effect on MFMA accumulator allocation. To use AGPRs, you need HipKittens-style inline assembly with explicit register pinning. ~20% improvement possible from eliminating AGPR↔VGPR copies.

10. **Q-from-global doesn't improve perf if VGPR-limited**: V4.3 reduced LDS from 23KB→12.5KB but performance stayed at ~108 TFLOPS because 124 VGPRs still limit occupancy to 4 waves/SIMD. LDS reduction only helps if it's the occupancy bottleneck.

---

## Reference: Flash Attention Optimization History

| Version | Change | S=8192 TFLOPS | Speedup | Key Metric Change |
|---------|--------|---------------|---------|-------------------|
| V4.0 | Baseline (BM=64,BN=16) | 26.3 | 1.0x | LDS wait/VALU = 3.05x |
| V4.1 | +Q-regs, +V-transpose, +padding | 60.1 | 2.3x | LDS wait/VALU = 0.53x |
| V4.2 | +BLOCK_N=32 | 67.6 | 2.6x | Half iterations/barriers |
| V4.2+exit | +causal early-exit | 108.2 | 4.1x | Skip ~50% KV work |
| V4.3 | +Q-from-global (LDS 23→12.5KB) | ~108 | 4.1x | LDS freed but VGPR-limited |
| ASM v3 | Hand-tuned assembly | 1012 | — | BM=256,mfma_32x32 |

Gap: 4.7x (using consistent FLOP convention).

### V4.3 Notes
- LDS overlay approach (reusing Q LDS for KV+P) FAILED due to LLVM scheduler reordering reads past barriers
- Q-from-global approach works: each MFMA lane loads v4f16 directly from global memory
- No perf improvement because 124 VGPRs (not LDS) limit occupancy to 4 waves/SIMD
- AGPRs not usable from MLIR — would need inline ASM (HipKittens approach)

### Remaining 4.7x Gap Analysis
| Factor | Our V4.3 | ASM v3 | Impact |
|--------|---------|--------|--------|
| Tile size | BM=64 | BM=256 | 4x Q amortization |
| MFMA size | 16x16x16 | 32x32x8 | 4x output/instr |
| K/V buffering | Single | Double (ping-pong) | Hides global→LDS latency |
| P path | P→LDS→VGPR | P in VGPRs (S/P ping-pong) | Eliminates roundtrip |
| Softmax | Sequential | XDL \|\| ALU co-exec | Hides softmax latency |
| Wave groups | 1 group | 2 staggered groups | LDS BW sharing |
| Reductions | XOR shuffle | V_PERMLANE16/32_SWAP | Faster reduction |
| AGPRs | None | Full AGPR usage | More register space |
