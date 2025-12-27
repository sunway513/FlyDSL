#!/usr/bin/env python3
"""
MoE GEMM stage1 unit test (flir, MFMA FP8, B preshuffle).

Reference: `../aiter/op_tests/test_moe_2stage.py` (stage1 semantics)
- Per-token topk routing: `topk_ids` and `topk_weights`
- Stage1: y2 = x @ W1[expert]^T -> split into gate/up (2*inter_dim)
         out1 = activation(gate) * up
         (optional) apply topk_weights (doweight_stage1)

This test constructs CK-style routing buffers:
- `sorted_token_ids`: packed (token_id | (topk_slot << 24)), padded to tile_m
- `expert_ids`: expert id per M tile (tile_m rows)

Kernel writes output as E[t, topk_slot, inter_dim] (fp16).
"""

import logging
import os
from typing import Tuple, Optional

import pytest
import torch
import torch.nn.functional as F

import pyflir
from pyflir.dialects.ext import flir
from pyflir.dialects.ext.python_control_flow import range_constexpr
from pyflir.runtime.device import get_rocm_arch as get_hip_arch
from pyflir.utils import SmemAllocator
from tests.utils import pertoken_quant, shuffle_weight
from tests.test_common import verify_output, run_perftest

from _mlir import ir
from _mlir.dialects import vector
from pyflir.dialects.ext import arith, gpu, buffer_ops, math as mlir_math
from _mlir.dialects import arith as _arith_mlir
import _mlir.dialects.rocdl as rocdl
import _mlir.extras.types as T
from pyflir.lang.ir.types import T as I

# Optional: use aiter's exact routing/sorting implementation (matches `aiter/op_tests/test_moe_2stage.py`).
# Some environments ship aiter python but miss required JIT .so dependencies; we fall back gracefully.
try:
    import aiter
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting

    HAS_AITER = True
except Exception:
    HAS_AITER = False

from tests.python.gpu.mfma_fp8_preshuffle_pipeline import (
    make_preshuffle_b_layout,
)

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def unwrap(v):
    if isinstance(v, int):
        return arith.constant(v, index=True).value
    while hasattr(v, "value") or hasattr(v, "_value"):
        v = getattr(v, "_value", getattr(v, "value", v))
    return v


def build_sorted_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    tile_m: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build CK-style routing buffers:
    - `sorted_token_ids[int32]` (fused token_idx[23:0] + topk_slot[31:24])
    - `sorted_weights[fp32]` aligned with sorted_token_ids
    - `sorted_expert_ids[int32]` per M-block

    Prefer aiter's `moe_sorting` (same path as `aiter/op_tests/test_moe_2stage.py`),
    then pad to `len(sorted_expert_ids) * tile_m` so our kernel never OOB-loads.
    """
    assert topk_ids.is_cuda and topk_weights.is_cuda
    tokens, topk = topk_ids.shape

    if HAS_AITER:
        try:
            topk_ids_i32 = topk_ids.to(torch.int32)
            topk_w_f32 = topk_weights.to(torch.float32)
            sorted_ids, sorted_w, sorted_expert_ids, _num_valid_ids, _moe_buf = aiter_moe_sorting(
                topk_ids_i32,
                topk_w_f32,
                num_experts,
                model_dim,
                torch.float16,
                tile_m,  # block_size must match our kernel tile_m
            )

            # Pad to full blocks so the kernel can safely index [bx*tile_m + lane_row]
            mblocks = int(sorted_expert_ids.numel())
            padded_len = mblocks * tile_m
            if int(sorted_ids.numel()) < padded_len:
                pad_ids = torch.empty((padded_len,), device="cuda", dtype=torch.int32)
                pad_w = torch.empty((padded_len,), device="cuda", dtype=torch.float32)
                pad_ids.fill_(tokens)  # sentinel fused token (token=tokens, slot=0)
                pad_w.zero_()
                pad_ids[: sorted_ids.numel()] = sorted_ids
                pad_w[: sorted_w.numel()] = sorted_w
                sorted_ids = pad_ids
                sorted_w = pad_w
            else:
                sorted_ids = sorted_ids[:padded_len]
                sorted_w = sorted_w[:padded_len]

            return sorted_ids, sorted_w, sorted_expert_ids
        except Exception:
            # Fall back below.
            pass

    # Fallback (no working aiter): build the same CK-style buffers using torch,
    # with a deterministic ordering: sort by (expert, token, slot).
    topk_ids_i64 = topk_ids.to(torch.int64)
    topk_w_f32 = topk_weights.to(torch.float32)

    token_idx = torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1).expand(tokens, topk)
    slot_idx = torch.arange(topk, device="cuda", dtype=torch.int64).unsqueeze(0).expand(tokens, topk)

    expert_flat = topk_ids_i64.reshape(-1)
    token_flat = token_idx.reshape(-1)
    slot_flat = slot_idx.reshape(-1)
    fused_flat = (token_flat & 0xFFFFFF) | ((slot_flat & 0xFF) << 24)
    weight_flat = topk_w_f32.reshape(-1)

    # Sort by (expert, token, slot) to make ordering stable/deterministic.
    linear = token_flat * topk + slot_flat
    key = expert_flat * (tokens * topk) + linear
    order = torch.argsort(key)

    expert_sorted = expert_flat[order]
    fused_sorted = fused_flat[order].to(torch.int32)
    w_sorted = weight_flat[order]

    counts = torch.bincount(expert_flat.clamp(min=0, max=num_experts - 1), minlength=num_experts).to("cpu").tolist()
    out_ids = []
    out_w = []
    out_expert_ids: list[int] = []
    off = 0
    sentinel = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    sentinel_w = torch.tensor([0.0], device="cuda", dtype=torch.float32)
    for e in range(num_experts):
        cnt = int(counts[e])
        seg_ids = fused_sorted[off : off + cnt]
        seg_w = w_sorted[off : off + cnt]
        off += cnt

        # pad to tile_m
        pad = (-cnt) % tile_m
        if pad:
            seg_ids = torch.cat([seg_ids, sentinel.expand(pad)])
            seg_w = torch.cat([seg_w, sentinel_w.expand(pad)])

        out_ids.append(seg_ids)
        out_w.append(seg_w)
        for _ in range(int(seg_ids.numel() // tile_m)):
            out_expert_ids.append(e)

    sorted_token_ids = torch.cat(out_ids, dim=0)
    sorted_weights = torch.cat(out_w, dim=0)
    sorted_expert_ids = torch.tensor(out_expert_ids, device="cuda", dtype=torch.int32)
    return sorted_token_ids, sorted_weights, sorted_expert_ids


def torch_stage1_ref(
    x_fp8: torch.Tensor,
    w1_fp8_flat: torch.Tensor,
    scale_x: torch.Tensor,
    scale_w1_flat: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inter_dim: int,
    doweight_stage1: bool,
) -> torch.Tensor:
    """Return [tokens, topk, inter_dim] fp32."""
    tokens, model_dim = x_fp8.shape
    topk = topk_ids.shape[1]
    experts = int(topk_ids.max().item()) + 1

    x = x_fp8.to(torch.float32) * scale_x  # [tokens, model_dim] (scale_x [tokens,1])
    w1 = w1_fp8_flat.to(torch.float32) * scale_w1_flat  # [experts*2*inter_dim, model_dim] (scale [rows,1])
    w1 = w1.view(experts, 2 * inter_dim, model_dim)

    out = torch.zeros((tokens, topk, inter_dim), device="cuda", dtype=torch.float32)
    for e in range(experts):
        # routes assigned to expert e
        mask = topk_ids == e
        idx = mask.nonzero(as_tuple=False)  # [num, 2] (t, slot)
        if idx.numel() == 0:
            continue
        t_idx = idx[:, 0]
        s_idx = idx[:, 1]
        y2 = F.linear(x[t_idx, :], w1[e, :, :])  # [num, 2*inter_dim]
        gate = y2[:, :inter_dim]
        up = y2[:, inter_dim:]
        y = F.silu(gate) * up
        if doweight_stage1:
            y = y * topk_weights[t_idx, s_idx].unsqueeze(-1)
        out[t_idx, s_idx, :] = y
    return out


@pytest.mark.parametrize(
    "tokens,model_dim,inter_dim,experts,topk,doweight_stage1",
    [
        (256, 1024, 256, 4, 2, False),
        (256, 1024, 256, 4, 2, True),
    ],
)
def _maybe_aiter_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    block_m: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids) or None."""
    if not HAS_AITER:
        return None
    try:
        # aiter expects i32 ids and fp32 weights
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_w_f32 = topk_weights.to(torch.float32)
        sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = aiter_moe_sorting(
            topk_ids_i32,
            topk_w_f32,
            num_experts,
            model_dim,
            torch.float16,
            block_m,
        )
        # `num_valid_ids` is documented as [1]; some builds allocate [2]. Keep the first element.
        if num_valid_ids.numel() > 1:
            num_valid_ids = num_valid_ids[:1].contiguous()
        return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids
    except Exception:
        return None


def _pad_sorted_buffers_to_full_blocks(
    *,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    tokens: int,
    block_m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad (sorted_ids, sorted_weights) to len(sorted_expert_ids)*block_m for safe OOB-free indexing."""
    assert sorted_ids.dtype == torch.int32
    assert sorted_weights.dtype == torch.float32
    blocks = int(sorted_expert_ids.numel())
    padded_len = blocks * block_m
    if int(sorted_ids.numel()) >= padded_len:
        return sorted_ids[:padded_len], sorted_weights[:padded_len]
    pad_ids = torch.empty((padded_len,), device="cuda", dtype=torch.int32)
    pad_w = torch.empty((padded_len,), device="cuda", dtype=torch.float32)
    pad_ids.fill_(tokens)  # sentinel fused token (token=tokens, slot=0)
    pad_w.zero_()
    pad_ids[: sorted_ids.numel()] = sorted_ids
    pad_w[: sorted_weights.numel()] = sorted_weights
    return pad_ids, pad_w


@pytest.mark.parametrize("tokens, model_dim, inter_dim, experts, topk, tile_m, tile_n, tile_k, doweight_stage1", [
    (256, 4096, 2048, 17, 9, 64, 128, 128, False),
])
def test_moe_stage1(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
):
    assert model_dim % 64 == 0
    assert model_dim % tile_k == 0
    assert inter_dim % tile_n == 0

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Data: input and weights (aiter shapes)
    x_fp32 = torch.randn((tokens, model_dim), device=device, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=device, dtype=torch.float32)
    # w2 is required by aiter CK API even for stage1; keep it allocated to avoid null ptr.
    # Stage1 kernels should not touch it, but we allocate a correct-shape tensor for safety.
    w2_fp32 = torch.randn((experts, model_dim, inter_dim), device=device, dtype=torch.float32)

    # Routing: aiter uses fused_topk; we use torch topk+softmax for portability/determinism.
    score = torch.randn((tokens, experts), device=device, dtype=torch.float32)
    topk_vals, topk_ids = torch.topk(score, k=topk, dim=1)
    topk_weights = torch.softmax(topk_vals, dim=1).to(torch.float32)

    # Prefer aiter moe_sorting buffers (exact CK routing format), fall back to torch sorting.
    sort_mode = os.environ.get("pyflir_MOE_SORT_MODE", "aiter" if HAS_AITER else "torch").lower()
    sorted_token_ids = sorted_weights = sorted_expert_ids = num_valid_ids = None
    if sort_mode == "aiter":
        res = _maybe_aiter_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts=experts,
            model_dim=model_dim,
            block_m=tile_m,
        )
        if res is None:
            pytest.skip("aiter moe_sorting unavailable in this environment; set pyflir_MOE_SORT_MODE=torch to run.", allow_module_level=False)
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res
    else:
        sorted_token_ids, sorted_weights, sorted_expert_ids = build_sorted_routing(
            topk_ids,
            topk_weights,
            num_experts=experts,
            model_dim=model_dim,
            tile_m=tile_m,
        )
        # num_valid_ids is total_tokens_post_pad; for our builder that's the full buffer length.
        num_valid_ids = torch.tensor([int(sorted_token_ids.numel())], device=device, dtype=torch.int32)

    # Pad to full blocks for safe indexing in our flir kernel (which uses grid_x = num_blocks).
    sorted_token_ids, sorted_weights = _pad_sorted_buffers_to_full_blocks(
        sorted_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        tokens=tokens,
        block_m=tile_m,
    )
    sorted_size = int(sorted_token_ids.numel())

    # FP8 per-token quantize (aiter config)
    x_fp8, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.float8_e4m3fnuz)  # [tokens, K], [tokens,1]
    w1_fp8, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.float8_e4m3fnuz)  # [E, 2*inter, K], [E,2*inter,1]
    # w2 is not used by our kernel, but required by CK stage1 API
    w2_fp8, _scale_w2_unused = pertoken_quant(w2_fp32, quant_dtype=torch.float8_e4m3fnuz)

    # Preshuffle weights (aiter/CK layout)
    w1_shuffled = shuffle_weight(w1_fp8)
    w2_shuffled = shuffle_weight(w2_fp8)

    # Flatten W1 for our flir kernel (treat expert dim as part of N)
    w1_shuffled_flat = w1_shuffled.view(experts * (2 * inter_dim), model_dim)
    w1_fp8_flat = w1_fp8.view(experts * (2 * inter_dim), model_dim)
    scale_w1_flat = scale_w1.view(experts * (2 * inter_dim), 1)

    # Pad storage for forced global dwordx4 loads (same trick as existing GEMM tests)
    PAD_ELEMS = 256
    x_flat = x_fp8.contiguous().view(-1)
    x_storage = torch.empty(x_flat.numel() + PAD_ELEMS, device=device, dtype=x_fp8.dtype)
    x_storage[: x_flat.numel()] = x_flat
    x_fp8 = x_storage[: x_flat.numel()].view(tokens, model_dim)

    w_flat = w1_shuffled_flat.contiguous().view(-1)
    w_storage = torch.empty(w_flat.numel() + PAD_ELEMS, device=device, dtype=w1_shuffled.dtype)
    w_storage[: w_flat.numel()] = w_flat
    w1_shuffled_flat = w_storage[: w_flat.numel()].view(experts * (2 * inter_dim), model_dim)

    # Flatten scales to 1D memrefs
    scale_x_1d = scale_x.view(-1).contiguous()  # [tokens]
    scale_w1_1d = scale_w1_flat.view(-1).contiguous()  # [rows]
    sorted_weights_1d = sorted_weights.contiguous().view(-1)  # [sorted_size]

    # Output: [tokens, topk, inter_dim] fp16
    out = torch.zeros((tokens, topk, inter_dim), device=device, dtype=torch.float16)

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    size_out = tokens * topk * inter_dim
    size_x = tokens * model_dim
    size_w = experts * (2 * inter_dim) * model_dim
    size_sorted = sorted_size
    size_expert_ids = int(sorted_expert_ids.numel())

    total_threads = 256
    elems_x_per_tile = tile_m * tile_k
    elems_per_thread_x = elems_x_per_tile // total_threads
    bytes_per_thread_x = elems_per_thread_x  # fp8
    # Keep MoE stage1 X gmem->LDS pipeline consistent with the optimized GEMM kernel:
    # split into <=16B pieces and use `flir.copy(load-only)` for buffer_load_dwordx4.
    # (Compute the split lens inside the kernel so the code matches GEMM structure.)

    # CK-style LDS128 mode (same idea as test_preshuffle_gemm.py):
    # - LDS stride == tile_k (no extra padding) + XOR16 swizzle
    # - Use ds_{read,write}_b128 (16B) and extract 8B halves for MFMA steps
    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in ("1", "true", "True", "YES", "yes")
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k

    class _MOE1(flir.MlirModule):
        GPU_MODULE_NAME = "mfma_moe1"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_x_decl"] = allocator.allocate_array(I.f8, tile_m * lds_stride)
            allocator.finalize()

        @flir.kernel
        def kernel(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, I.f8),
            arg_w: lambda: T.memref(size_w, I.f8),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            # Compile-time experiment knobs (read once in Python, used to gate IR emission).
            _no_epilogue = os.environ.get("FLIR_MOE_NO_EPILOGUE", "0") in ("1", "true", "True", "YES", "yes")

            f8 = I.f8
            f32 = I.f32
            i32 = I.i32
            i64 = I.i64
            vec4_f32 = I.vec(4, f32)
            vec8_f8 = I.vec(8, f8)
            vec16_f8 = I.vec(16, f8)
            vec1_i64 = I.vec(1, i64)
            vec2_i64 = I.vec(2, i64)

            c0 = arith.constant(0, index=True)
            c4 = arith.constant(4, index=True)
            c1 = arith.constant(1, index=True)
            c16 = arith.constant(16, index=True)
            c64 = arith.constant(64, index=True)
            c256 = arith.constant(256, index=True)
            c1024 = arith.constant(1024, index=True)
            c_tile_k = arith.constant(tile_k, index=True)

            c0f = arith.constant(0.0, type=f32).value
            c1f = arith.constant(1.0, type=f32).value
            c_log2e = arith.constant(1.4426950408889634, type=f32).value  # log2(e)

            def silu(x):
                # sigmoid(x) = 1 / (1 + exp(-x)) using exp2
                neg = flir.arith.SubFOp(unwrap(c0f), unwrap(x)).result
                scaled = flir.arith.MulFOp(unwrap(neg), unwrap(c_log2e)).result
                exp_neg = mlir_math.exp2(unwrap(scaled))
                den = flir.arith.AddFOp(unwrap(c1f), unwrap(exp_neg)).result
                sig = flir.arith.DivFOp(unwrap(c1f), unwrap(den)).result
                return flir.arith.MulFOp(unwrap(x), unwrap(sig)).result

            zero_attr = ir.DenseElementsAttr.get_splat(vec4_f32, ir.FloatAttr.get(f32, 0.0))
            acc_init = _arith_mlir.ConstantOp(vec4_f32, zero_attr).result

            # Layouts
            layout_x = flir.make_layout((tokens_in, k_in), stride=(k_in, 1))

            # B preshuffle layout: match GEMM test helper exactly.
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            b_layout = make_preshuffle_b_layout(flir, _arith_mlir, c_n=unwrap(c_n_total), c_k=unwrap(k_in))
            layout_b = b_layout.layout_b
            c_k0 = _arith_mlir.DivUIOp(unwrap(k_in), unwrap(c64)).result

            shape_lds = flir.make_shape(tile_m, tile_k)
            stride_lds = flir.make_stride(lds_stride, 1)
            layout_lds = flir.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")  # tile along sorted M
            by = gpu.block_id("y")  # tile along inter_dim

            # Common constants/atoms (hoisted): keep IR small like GEMM.
            # CK-style XOR16 swizzle parameter (constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k // 16, index=True)
            atom_x_s16 = flir.make_copy_atom(f8, vector_size=16)
            atom_x_s8 = flir.make_copy_atom(f8, vector_size=8)
            atom_x_g2r16 = flir.make_copy_atom(f8, vector_size=16)
            atom_x_g2r8 = flir.make_copy_atom(f8, vector_size=8)
            layout_tx_wave_lane = flir.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = flir.make_layout((4, 16), stride=(16, 1))
            layout_lin_rowcol = flir.make_layout((tile_m, tile_k), stride=(tile_k, 1))

            base_ptr = allocator.get_base()
            lds_x = _state["lds_x_decl"](base_ptr).get()

            x_rsrc = buffer_ops.create_buffer_resource(arg_x)
            w_rsrc = buffer_ops.create_buffer_resource(arg_w)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out)
            sx_rsrc = buffer_ops.create_buffer_resource(arg_scale_x)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w)
            sorted_rsrc = buffer_ops.create_buffer_resource(arg_sorted_token_ids)
            expert_rsrc = buffer_ops.create_buffer_resource(arg_expert_ids)
            sorted_w_rsrc = buffer_ops.create_buffer_resource(arg_sorted_weights)

            # Expert id for this M tile (keep address math in `index`)
            expert_i32 = buffer_ops.buffer_load(expert_rsrc, bx, vec_width=1, dtype=i32)
            expert_idx = _arith_mlir.IndexCastOp(ir.IndexType.get(), unwrap(expert_i32)).result
            inter2_idx = arith.constant(2 * inter_dim, index=True)
            expert_off_idx = _arith_mlir.MulIOp(unwrap(expert_idx), unwrap(inter2_idx)).result  # index

            # Thread -> (row_a_local, col_a_local) via layout algebra (GEMM-style).
            vec_len_val = arith.constant(bytes_per_thread_x, index=True)
            linear_id = _arith_mlir.MulIOp(unwrap(tx), unwrap(vec_len_val)).result
            coord_rc = flir.idx2crd(unwrap(linear_id), layout_lin_rowcol)
            row_a_local = flir.get(coord_rc, 0)
            col_a_local = flir.get(coord_rc, 1)

            bx_m = _arith_mlir.MulIOp(unwrap(bx), unwrap(arith.constant(tile_m, index=True))).result
            sorted_row = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_a_local)).result

            # Load fused token id and decode token (t) and topk slot (s)
            fused = buffer_ops.buffer_load(sorted_rsrc, sorted_row, vec_width=1, dtype=i32)
            mask24 = arith.i32(0xFFFFFF)._value
            t_i32 = _arith_mlir.AndIOp(unwrap(fused), unwrap(mask24)).result
            s_i32 = _arith_mlir.ShRUIOp(unwrap(fused), unwrap(arith.i32(24)._value)).result

            # valid token?
            tokens_i32 = arith.i32(tokens)._value
            topk_i32 = arith.i32(topk)._value
            valid_t = _arith_mlir.CmpIOp(_arith_mlir.CmpIPredicate.ult, unwrap(t_i32), unwrap(tokens_i32)).result
            t_idx = _arith_mlir.IndexCastOp(ir.IndexType.get(), unwrap(t_i32)).result

            # X base index (token-major)
            coord_x = flir.make_coord(unwrap(t_idx), unwrap(col_a_local))
            idx_x = flir.crd2idx(coord_x, layout_x)
            idx_x_div4 = _arith_mlir.DivUIOp(unwrap(idx_x), unwrap(c4)).result

            # ---- X gmem->reg prefetch (GEMM-style) ----
            max_bytes_per_load = 16
            num_x_loads = (bytes_per_thread_x + max_bytes_per_load - 1) // max_bytes_per_load
            vec_x_parts_lens = []
            remaining_bytes = bytes_per_thread_x
            for _ in range_constexpr(num_x_loads):
                curr_bytes = min(remaining_bytes, max_bytes_per_load)
                vec_x_parts_lens.append(curr_bytes)
                remaining_bytes -= curr_bytes

            vec2_i32 = I.vec(2, i32)
            vec4_i32 = I.vec(4, i32)

            def load_x_part(idx_i32, extent: int):
                """Load `extent` fp8 bytes from X (gmem) into a vector via buffer_load backend.

                `idx_i32` is an i32-element (dword) offset (not bytes).
                """
                x_view = flir.TensorView(
                    arg_x,
                    (extent,),
                    strides=(1,),
                    base_indices=(unwrap(idx_i32),),
                    element_type=f8,
                )
                atom = atom_x_g2r16 if extent == 16 else atom_x_g2r8
                return flir.copy(
                    atom,
                    x_view,
                    None,
                    pred=valid_t,
                    alignment=extent,
                    return_vector=True,
                    src_buffer_resource=x_rsrc,
                    src_buffer_offset_in_bytes=False,
                )

            vec_x_inits = []
            curr_load_off = 0
            for i in range_constexpr(num_x_loads):
                curr_bytes = vec_x_parts_lens[i]
                off_i32 = arith.constant(curr_load_off // 4, index=True)
                idx_i32 = idx_x_div4 + off_i32
                x_f8 = load_x_part(idx_i32, curr_bytes)
                if curr_bytes == 16:
                    vec_x_inits.append(vector.BitCastOp(vec4_i32, x_f8).result)
                elif curr_bytes == 8:
                    vec_x_inits.append(vector.BitCastOp(vec2_i32, x_f8).result)
                else:
                    vec_x_inits.append(x_f8)
                curr_load_off += curr_bytes

            vec_x_parts = vec_x_inits

            # tx -> wave/lane (GEMM-style decomposition).
            coord_wl = flir.idx2crd(unwrap(tx), layout_tx_wave_lane)
            wave_id = flir.get(coord_wl, 0)
            lane_id = flir.get(coord_wl, 1)
            coord_l16 = flir.idx2crd(lane_id, layout_lane16)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            # Match GEMM naming/pattern: row in LDS is lane_mod_16, and col base is lane_div_16*16.
            row_a_lds = lane_mod_16
            col_offset_base = flir.crd2idx(flir.make_coord(lane_div_16, 0), layout_lane16)

            # Dynamic N tiling within block (same as existing kernels)
            by_n = _arith_mlir.MulIOp(unwrap(by), unwrap(arith.constant(tile_n, index=True))).result
            num_waves = 4
            n_per_wave = tile_n // num_waves
            num_acc_n = n_per_wave // 16
            c_n_per_wave = arith.constant(n_per_wave, index=True)
            wave_mod_4 = _arith_mlir.RemUIOp(unwrap(wave_id), unwrap(c4)).result
            n_tile_base = _arith_mlir.MulIOp(unwrap(wave_mod_4), unwrap(c_n_per_wave)).result

            # Precompute n_blk/n_intra for gate and up rows (GEMM-style: idx2crd/get)
            n_intra_gate = []
            n_blk_gate = []
            n_intra_up = []
            n_blk_up = []
            col_g_list = []
            valid_col_list = []
            inter_idx = arith.constant(inter_dim, index=True)
            # layout for (row -> (blk,intra)) where intra is 0..15
            c_n0 = _arith_mlir.DivUIOp(unwrap(c_n_total), unwrap(c16)).result
            layout_n_blk_intra = flir.make_layout((c_n0, 16), stride=(16, 1))
            for ni in range_constexpr(num_acc_n):
                offset = arith.constant(ni * 16, index=True)
                col_g = _arith_mlir.AddIOp(unwrap(by_n), unwrap(n_tile_base)).result
                col_g = _arith_mlir.AddIOp(unwrap(col_g), unwrap(offset)).result
                col_g = _arith_mlir.AddIOp(unwrap(col_g), unwrap(lane_mod_16)).result
                col_g_list.append(col_g)

                row_gate = _arith_mlir.AddIOp(unwrap(expert_off_idx), unwrap(col_g)).result
                row_up = _arith_mlir.AddIOp(unwrap(row_gate), unwrap(inter_idx)).result

                coord_gate = flir.idx2crd(unwrap(row_gate), layout_n_blk_intra)
                n_blk_gate.append(flir.get(coord_gate, 0))
                n_intra_gate.append(flir.get(coord_gate, 1))

                coord_up = flir.idx2crd(unwrap(row_up), layout_n_blk_intra)
                n_blk_up.append(flir.get(coord_up, 0))
                n_intra_up.append(flir.get(coord_up, 1))

                valid_col_list.append(
                    _arith_mlir.CmpIOp(
                        _arith_mlir.CmpIPredicate.ult, unwrap(col_g), unwrap(inter_idx)
                    ).result
                )

            m_repeat = tile_m // 16
            k_unroll = tile_k // 32

            # --- B Load Logic (K32) - match GEMM style exactly ---
            layout_k0_kpack64 = flir.make_layout((c_k0, 64), stride=(64, 1))
            layout_half8 = flir.make_layout((2, 8), stride=(8, 1))
            atom_b_g2r = flir.make_copy_atom(f8, vector_size=8)

            def load_b_pack(base_k, ki_step, ni, blk_list, intra_list):
                coord_k = flir.idx2crd(unwrap(base_k), layout_k0_kpack64)
                k0_base = flir.get(coord_k, 0)
                k0 = k0_base + (ki_step // 2)
                k1 = lane_div_16  # 0..3
                half = ki_step % 2
                half_val = arith.constant(half, index=True)
                k2_base = flir.crd2idx(flir.make_coord(unwrap(half_val), 0), layout_half8)

                n_intra = intra_list[ni]
                n_blk = blk_list[ni]
                coord_b = flir.make_coord(n_blk, k0, k1, n_intra, k2_base)
                idx_bytes = flir.crd2idx(coord_b, layout_b)

                b_view = flir.TensorView(
                    arg_w,
                    (8,),
                    strides=(1,),
                    base_indices=(unwrap(idx_bytes),),
                    element_type=f8,
                )
                b8_f8 = flir.copy(
                    atom_b_g2r,
                    b_view,
                    None,
                    alignment=8,
                    return_vector=True,
                    src_buffer_resource=w_rsrc,
                )
                b_vec64 = vector.BitCastOp(vec1_i64, b8_f8).result
                return vector.ExtractOp(b_vec64, static_position=[0], dynamic_position=[]).result

            acc_gate = [acc_init] * (num_acc_n * m_repeat)
            acc_up = [acc_init] * (num_acc_n * m_repeat)

            def emit_tile(k_iv, acc_gate_in, acc_up_in, vec_x_in_parts, is_last=False):
                # Store X to LDS
                store_off = 0
                for pi in range_constexpr(num_x_loads):
                    val_vec = vec_x_in_parts[pi]
                    curr_bytes = vec_x_parts_lens[pi]
                    # Match GEMM address pattern exactly.
                    col_0 = col_a_local + store_off
                    col_swz = flir.swizzle_xor16(row_a_local, col_0, k_blocks16)
                    coord_store_0 = flir.make_coord(row_a_local, col_swz)
                    idx_0 = flir.crd2idx(coord_store_0, layout_lds)

                    if curr_bytes == 16:
                        v16 = vector.BitCastOp(vec16_f8, val_vec).result
                        s_view = flir.TensorView(
                            lds_x, (16,), strides=(1,), base_indices=(idx_0,), element_type=f8
                        )
                        flir.copy(atom_x_s16, v16, s_view, alignment=16 if _ck_lds128 else None)
                    elif curr_bytes == 8:
                        v8 = vector.BitCastOp(vec8_f8, val_vec).result
                        s_view = flir.TensorView(
                            lds_x, (8,), strides=(1,), base_indices=(idx_0,), element_type=f8
                        )
                        flir.copy(atom_x_s8, v8, s_view, alignment=8 if _ck_lds128 else None)
                    else:
                        # Rare fallback: keep as fp8 vector and rely on scalar stores.
                        s_view = flir.TensorView(
                            lds_x, (curr_bytes,), strides=(1,), base_indices=(idx_0,), element_type=f8
                        )
                        flir.copy(atom_x_s8, val_vec, s_view)
                    store_off += curr_bytes

                gpu.barrier()

                vec_x_next = vec_x_in_parts
                if not is_last:
                    next_k = _arith_mlir.AddIOp(unwrap(k_iv), unwrap(c_tile_k)).result
                    next_k_div4 = _arith_mlir.DivUIOp(unwrap(next_k), unwrap(c4)).result
                    vec_x_next = []
                    curr_load_off2 = 0
                    for i in range_constexpr(num_x_loads):
                        curr_bytes = vec_x_parts_lens[i]
                        off_i32 = arith.constant(curr_load_off2 // 4, index=True)
                        idx_i32 = idx_x_div4 + next_k_div4 + off_i32
                        x_f8 = load_x_part(idx_i32, curr_bytes)
                        if curr_bytes == 16:
                            vec_x_next.append(vector.BitCastOp(vec4_i32, x_f8).result)
                        elif curr_bytes == 8:
                            vec_x_next.append(vector.BitCastOp(vec2_i32, x_f8).result)
                        else:
                            vec_x_next.append(x_f8)
                        curr_load_off2 += curr_bytes

                gate_list = list(acc_gate_in)
                up_list = list(acc_up_in)

                # Inline MFMA hot loop (match fast GEMM K32 pattern: ki_step outer -> mi inner).
                c0_i32 = arith.i32(0)._value

                for ki_step in range_constexpr(k_unroll):
                    # Load B packs (gate+up) once per ki_step, reuse across MI.
                    b_gate_packs = []
                    b_up_packs = []
                    for ni in range_constexpr(num_acc_n):
                        b_gate_packs.append(load_b_pack(k_iv, ki_step, ni, n_blk_gate, n_intra_gate))
                        b_up_packs.append(load_b_pack(k_iv, ki_step, ni, n_blk_up, n_intra_up))

                    half = ki_step % 2
                    ki64 = (ki_step // 2) * 64
                    col_base = col_offset_base + ki64

                    for mi in range_constexpr(m_repeat):
                        mi_val = arith.constant(mi * 16, index=True)
                        curr_row_a_lds = row_a_lds + mi_val

                        # Read X from LDS using the same (row,col)->(row,col') xor swizzle as the store.
                        col_base_swizzled = flir.swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
                        if _ck_lds128:
                            # Base is 16B-aligned; read 16B then pick low/high 8B for this MFMA step.
                            coord_a16 = flir.make_coord(curr_row_a_lds, col_base_swizzled)
                            idx_a16 = flir.crd2idx(coord_a16, layout_lds)
                            loaded_a16 = vector.LoadOp(
                                vec16_f8, lds_x, [arith.ArithValue(idx_a16).value]
                            ).result
                            a_vec128 = vector.BitCastOp(vec2_i64, loaded_a16).result
                            a_pack = vector.ExtractOp(
                                a_vec128, static_position=[half], dynamic_position=[]
                            ).result
                        else:
                            # Legacy: read only the needed 8B half.
                            col_swizzled = col_base_swizzled + half * 8
                            coord_a = flir.make_coord(curr_row_a_lds, col_swizzled)
                            idx_a = flir.crd2idx(coord_a, layout_lds)
                            loaded_a8 = vector.LoadOp(
                                vec8_f8, lds_x, [arith.ArithValue(idx_a).value]
                            ).result
                            a_vec64 = vector.BitCastOp(vec1_i64, loaded_a8).result
                            a_pack = vector.ExtractOp(
                                a_vec64, static_position=[0], dynamic_position=[]
                            ).result

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            gate_list[acc_idx] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a_pack),
                                    unwrap(b_gate_packs[ni]),
                                    unwrap(gate_list[acc_idx]),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result
                            up_list[acc_idx] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                vec4_f32,
                                [
                                    unwrap(a_pack),
                                    unwrap(b_up_packs[ni]),
                                    unwrap(up_list[acc_idx]),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                    unwrap(c0_i32),
                                ],
                            ).result

                gpu.barrier()
                return gate_list, up_list, vec_x_next

            c_k_main = _arith_mlir.SubIOp(unwrap(k_in), unwrap(c_tile_k)).result
            for k_iv in range(c0, c_k_main, c_tile_k):
                acc_gate, acc_up, vec_x_parts = emit_tile(k_iv, acc_gate, acc_up, vec_x_parts, is_last=False)
            acc_gate, acc_up, _ = emit_tile(c_k_main, acc_gate, acc_up, vec_x_parts, is_last=True)

            if _no_epilogue:
                # Minimal "no-epilogue" sink to prevent DCE of the MFMA core:
                # write one fp16 value per accumulator vector (gate+up) into a compact prefix of `out`.
                # This avoids all routing/scale/activation work while keeping the main loop intact.
                tx_i32 = _arith_mlir.IndexCastOp(i32, unwrap(tx)).result
                c256_i32 = arith.i32(256)._value
                c_up_base = arith.i32(256 * (m_repeat * num_acc_n))._value

                for mi in range_constexpr(m_repeat):
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        slot_i32 = arith.i32(acc_idx * 256)._value

                        # Gate: store element 0
                        vg = vector.ExtractOp(acc_gate[acc_idx], [], [0]).result
                        vg_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(vg)).result
                        idx_gate = _arith_mlir.AddIOp(unwrap(tx_i32), unwrap(slot_i32)).result
                        buffer_ops.buffer_store(vg_f16, out_rsrc, idx_gate)

                        # Up: store element 0 at an offset
                        vu = vector.ExtractOp(acc_up[acc_idx], [], [0]).result
                        vu_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(vu)).result
                        idx_up = _arith_mlir.AddIOp(unwrap(idx_gate), unwrap(c_up_base)).result
                        buffer_ops.buffer_store(vu_f16, out_rsrc, idx_up)

                return

            # Store epilogue to out[t, slot, inter]
            # Recompute token/slot for each output row this lane writes.
            #
            # GEMM-style prefetch: scale_w depends only on (expert,row) and `col_g` (ni),
            # not on token/slot. Hoist it out of the MI/II loops.
            sw_gate_vals = []
            sw_up_vals = []
            for ni in range_constexpr(num_acc_n):
                col_g = col_g_list[ni]
                valid_col = valid_col_list[ni]
                row_gate_idx = _arith_mlir.AddIOp(unwrap(expert_off_idx), unwrap(col_g)).result
                row_up_idx = _arith_mlir.AddIOp(unwrap(row_gate_idx), unwrap(inter_idx)).result
                sw_gate_vals.append(
                    buffer_ops.buffer_load(sw_rsrc, row_gate_idx, vec_width=1, dtype=f32, mask=valid_col)
                )
                sw_up_vals.append(
                    buffer_ops.buffer_load(sw_rsrc, row_up_idx, vec_width=1, dtype=f32, mask=valid_col)
                )

            for mi in range_constexpr(m_repeat):
                mi_base = arith.constant(mi * 16, index=True)
                for ii in range_constexpr(4):
                    row_off = _arith_mlir.AddIOp(unwrap(_arith_mlir.MulIOp(unwrap(lane_div_16), unwrap(c4)).result), unwrap(arith.constant(ii, index=True))).result
                    row_in_tile = _arith_mlir.AddIOp(unwrap(mi_base), unwrap(row_off)).result
                    sorted_row2 = _arith_mlir.AddIOp(unwrap(bx_m), unwrap(row_in_tile)).result

                    fused2 = buffer_ops.buffer_load(sorted_rsrc, sorted_row2, vec_width=1, dtype=i32)
                    t2 = _arith_mlir.AndIOp(unwrap(fused2), unwrap(mask24)).result
                    s2 = _arith_mlir.ShRUIOp(unwrap(fused2), unwrap(arith.i32(24)._value)).result
                    valid2 = _arith_mlir.CmpIOp(_arith_mlir.CmpIPredicate.ult, unwrap(t2), unwrap(tokens_i32)).result

                    t2_idx = _arith_mlir.IndexCastOp(ir.IndexType.get(), unwrap(t2)).result
                    sx = buffer_ops.buffer_load(sx_rsrc, t2_idx, vec_width=1, dtype=f32, mask=valid2)

                    # Sorted weight aligned with `sorted_row2` (matches aiter moe_sorting output)
                    tw = buffer_ops.buffer_load(sorted_w_rsrc, sorted_row2, vec_width=1, dtype=f32, mask=valid2)

                    for ni in range_constexpr(num_acc_n):
                        col_g = col_g_list[ni]
                        valid_col = valid_col_list[ni]
                        valid_store = _arith_mlir.AndIOp(unwrap(valid2), unwrap(valid_col)).result

                        col_i32 = _arith_mlir.IndexCastOp(i32, unwrap(col_g)).result
                        sw_gate = sw_gate_vals[ni]
                        sw_up = sw_up_vals[ni]

                        acc_idx = mi * num_acc_n + ni
                        vg = vector.ExtractOp(acc_gate[acc_idx], [], [ii]).result
                        vu = vector.ExtractOp(acc_up[acc_idx], [], [ii]).result

                        vg = flir.arith.MulFOp(unwrap(vg), unwrap(sx)).result
                        vg = flir.arith.MulFOp(unwrap(vg), unwrap(sw_gate)).result
                        vu = flir.arith.MulFOp(unwrap(vu), unwrap(sx)).result
                        vu = flir.arith.MulFOp(unwrap(vu), unwrap(sw_up)).result

                        y = flir.arith.MulFOp(unwrap(silu(vg)), unwrap(vu)).result
                        if doweight_stage1:
                            y = flir.arith.MulFOp(unwrap(y), unwrap(tw)).result

                        y_f16 = _arith_mlir.TruncFOp(T.f16(), unwrap(y)).result

                        # out linear index = ((t*topk + s)*inter_dim + col)
                        inter_i32_local = arith.i32(inter_dim)._value
                        idx0 = _arith_mlir.MulIOp(unwrap(t2), unwrap(topk_i32)).result
                        idx0 = _arith_mlir.AddIOp(unwrap(idx0), unwrap(s2)).result
                        idx0 = _arith_mlir.MulIOp(unwrap(idx0), unwrap(inter_i32_local)).result
                        idx_out = _arith_mlir.AddIOp(unwrap(idx0), unwrap(col_i32)).result

                        buffer_ops.buffer_store(y_f16, out_rsrc, idx_out, mask=valid_store)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_out: lambda: T.memref(size_out, T.f16()),
            arg_x: lambda: T.memref(size_x, I.f8),
            arg_w: lambda: T.memref(size_w, I.f8),
            arg_scale_x: lambda: T.memref(tokens, T.f32()),
            arg_scale_w: lambda: T.memref(experts * (2 * inter_dim), T.f32()),
            arg_sorted_token_ids: lambda: T.memref(size_sorted, T.i32()),
            arg_expert_ids: lambda: T.memref(size_expert_ids, T.i32()),
            arg_sorted_weights: lambda: T.memref(size_sorted, T.f32()),
            tokens_in: lambda: T.index(),
            inter_in: lambda: T.index(),
            k_in: lambda: T.index(),
        ):
            bdx = 256
            gx = size_expert_ids
            gy = inter_dim // tile_n
            flir.gpu_ext.LaunchFuncOp(
                ["mfma_moe1", "kernel"],
                grid_size=(gx, gy, 1),
                block_size=(bdx, 1, 1),
                kernel_operands=[
                    unwrap(arg_out),
                    unwrap(arg_x),
                    unwrap(arg_w),
                    unwrap(arg_scale_x),
                    unwrap(arg_scale_w),
                    unwrap(arg_sorted_token_ids),
                    unwrap(arg_expert_ids),
                    unwrap(arg_sorted_weights),
                    unwrap(tokens_in),
                    unwrap(inter_in),
                    unwrap(k_in),
                ],
            )

    m = _MOE1()
    exe = pyflir.compile(m)

    def launch(o, x, w, sx, sw, st, eids, sw_sorted):
        exe(o, x, w, sx, sw, st, eids, sw_sorted, tokens, inter_dim, model_dim)

    _, us = run_perftest(
        launch,
        out,
        x_fp8,
        w1_shuffled_flat,
        scale_x_1d,
        scale_w1_1d,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights_1d,
        num_iters=5,
        num_warmup=2,
    )
    torch.cuda.synchronize()

    if os.environ.get("FLIR_MOE_NO_EPILOGUE", "0") in ("1", "true", "True", "YES", "yes"):
        # No-epilogue mode writes synthetic values; skip correctness/CK comparison.
        flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
        tflops = flops / (us / 1e6) / 1e12
        print(f"[no-epilogue] MoE stage1: {us:.1f} us, {tflops:.2f} TFLOPS")
        return

    ref = torch_stage1_ref(
        x_fp8,
        w1_fp8_flat,
        scale_x,
        scale_w1_flat,
        topk_ids.to(torch.int64),
        topk_weights,
        inter_dim=inter_dim,
        doweight_stage1=doweight_stage1,
    )

    assert verify_output(out.to(torch.float32), ref, rtol=0.25, atol=0.25)

    # Compare + benchmark vs aiter CK stage1 (optional; enabled by default when aiter is runnable).
    compare_ck = os.environ.get("COMPARE_AITER_CK", "1" if HAS_AITER else "0") == "1"
    if compare_ck:
        if not HAS_AITER:
            pytest.skip("aiter not available; cannot compare to CK moe stage1.", allow_module_level=False)
        try:
            from aiter.ops.moe_op import ck_moe_stage1_fwd
            from aiter.ops.enum import QuantType, ActivationType

            out_ck = torch.empty((tokens, topk, inter_dim), device=device, dtype=torch.float16)

            # aiter CK expects w1/w2 with expert dimension preserved.
            w1_ck = w1_shuffled
            w2_ck = w2_shuffled
            w1_scale_ck = scale_w1.contiguous()

            def launch_ck(o, x, w1_, w2_, sorted_ids_, sorted_eids_, num_valid_, w1_scale_, a1_scale_, sorted_w_):
                ck_moe_stage1_fwd(
                    hidden_states=x,
                    w1=w1_,
                    w2=w2_,
                    sorted_token_ids=sorted_ids_,
                    sorted_expert_ids=sorted_eids_,
                    num_valid_ids=num_valid_,
                    out=o,
                    topk=topk,
                    kernelName="",
                    w1_scale=w1_scale_,
                    a1_scale=a1_scale_,
                    block_m=tile_m,
                    sorted_weights=sorted_w_ if doweight_stage1 else None,
                    quant_type=QuantType.per_Token,
                    activation=ActivationType.Silu,
                    splitk=1,
                    dst_type=torch.float16,
                )

            # Benchmark CK stage1
            _, us_ck = run_perftest(
                launch_ck,
                out_ck,
                x_fp8,
                w1_ck,
                w2_ck,
                sorted_token_ids,
                sorted_expert_ids,
                num_valid_ids,
                w1_scale_ck,
                scale_x,
                sorted_weights,
                num_iters=5,
                num_warmup=2,
            )

            # Correctness: flir vs CK
            assert verify_output(out.to(torch.float32), out_ck.to(torch.float32), rtol=0.25, atol=0.25)

            # Perf print: use the same flop model for both
            flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
            tflops_ck = flops / (us_ck / 1e6) / 1e12
            print(f"[aiter CK] stage1: {us_ck:.1f} us, {tflops_ck:.2f} TFLOPS")
        except Exception as e:
            pytest.skip(f"aiter CK moe stage1 not runnable in this environment: {e}", allow_module_level=False)
    flops = 2 * tokens * topk * (2 * inter_dim) * model_dim
    tflops = flops / (us / 1e6) / 1e12

    # Rough bytes-moved accounting (same spirit as GEMM tests: count each tensor once).
    bytes_moved = 0
    bytes_moved += tokens * model_dim * 1  # x fp8
    bytes_moved += experts * (2 * inter_dim) * model_dim * 1  # w fp8
    bytes_moved += tokens * topk * inter_dim * 2  # out fp16
    bytes_moved += tokens * 4  # scale_x f32 (1D)
    bytes_moved += experts * (2 * inter_dim) * 4  # scale_w f32 (1D)
    bytes_moved += int(sorted_weights.numel()) * 4  # sorted_weights f32
    bytes_moved += int(sorted_token_ids.numel()) * 4  # sorted_token_ids i32
    bytes_moved += int(sorted_expert_ids.numel()) * 4  # sorted_expert_ids i32
    tbps = bytes_moved / 1e12 / (us / 1e6)

    print(
        f"MoE stage1: {us:.1f} us, {tflops:.2f} TFLOPS, {tbps:.3f} TB/s (doweight_stage1={doweight_stage1})"
    )


if __name__ == "__main__":
    torch.set_default_device("cuda")
    # Run the aiter-aligned FP8 per-token stage1 test.
    test_moe_stage1(tokens=2048, model_dim=4096, inter_dim=2560, experts=17, topk=9, tile_m=64, tile_n=128, tile_k=128, doweight_stage1=False)



