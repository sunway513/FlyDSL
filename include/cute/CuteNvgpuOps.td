//===- CuteNvgpuOps.td - CuTe NVGPU Operations -------------*- tablegen -*-===//
//
// GPU Hardware-Specific Operations
// Architecture: SM80 (Ampere), SM90 (Hopper), SM100 (Blackwell)
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_NVGPU_OPS
#define CUTE_NVGPU_OPS

include "cute/CuteNvgpuDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Atom Construction Operations
//===----------------------------------------------------------------------===//

def CuteNvgpu_MakeMmaAtomOp : CuteNvgpu_Op<"make_mma_atom", [Pure]> {
  let summary = "Construct MMA atom descriptor";
  let description = [{
    Creates compile-time MMA atom with specified shape and types.
    
    Example:
    ```mlir
    %atom = cute_nvgpu.make_mma_atom !f16, !f16, !f32, [16, 8, 16], "SM80"
      : !cute_nvgpu.mma_atom<!f16, !f16, !f32, [16, 8, 16], SM80>
    ```
  }];
  
  let arguments = (ins
    TypeAttr:$elem_type_a,
    TypeAttr:$elem_type_b,
    TypeAttr:$elem_type_c,
    I64ArrayAttr:$shape,
    StrAttr:$arch
  );
  let results = (outs CuteNvgpu_MmaAtomType:$result);
}

def CuteNvgpu_MakeCopyAtomOp : CuteNvgpu_Op<"make_copy_atom", [Pure]> {
  let summary = "Construct copy atom descriptor";
  let arguments = (ins
    TypeAttr:$elem_type,
    I64Attr:$vector_width,
    StrAttr:$instruction,
    StrAttr:$src_space,
    StrAttr:$dst_space
  );
  let results = (outs CuteNvgpu_CopyAtomType:$result);
}

//===----------------------------------------------------------------------===//
// Non-Executable TMA Operations (SM90+)
//===----------------------------------------------------------------------===//

def CuteNvgpu_MakeTmaLoadOp : CuteNvgpu_Op<"make_tma_load", [Pure]> {
  let summary = "Create non-executable TMA load descriptor";
  let description = [{
    Creates TMA load descriptor for Hopper+ GPUs.
    Requires atom_set_tma_descriptor before execution.
  }];
  
  let arguments = (ins
    TypeAttr:$elem_type,
    I64ArrayAttr:$tile_shape,
    StrAttr:$swizzle
  );
  let results = (outs CuteNvgpu_TmaLoadType:$result);
}

def CuteNvgpu_MakeTmaStoreOp : CuteNvgpu_Op<"make_tma_store", [Pure]> {
  let summary = "Create non-executable TMA store descriptor";
  let arguments = (ins
    TypeAttr:$elem_type,
    I64ArrayAttr:$tile_shape,
    StrAttr:$swizzle
  );
  let results = (outs CuteNvgpu_TmaStoreType:$result);
}

//===----------------------------------------------------------------------===//
// Executable TMA Operations (SM90+)
//===----------------------------------------------------------------------===//

def CuteNvgpu_MakeExecTmaLoadOp : CuteNvgpu_Op<"make_exec_tma_load"> {
  let summary = "Create executable TMA load with barrier";
  let description = [{
    Wraps non-exec TMA load with barrier for immediate execution.
    Used in kernel body after descriptor initialization.
  }];
  
  let arguments = (ins
    CuteNvgpu_TmaLoadType:$descriptor,
    AnyType:$barrier
  );
  let results = (outs CuteNvgpu_TmaLoadExecType:$result);
}

def CuteNvgpu_MakeExecTmaStoreOp : CuteNvgpu_Op<"make_exec_tma_store"> {
  let summary = "Create executable TMA store with barrier";
  let arguments = (ins
    CuteNvgpu_TmaStoreType:$descriptor,
    AnyType:$barrier
  );
  let results = (outs CuteNvgpu_TmaStoreExecType:$result);
}

//===----------------------------------------------------------------------===//
// Runtime Atom Value Operations
//===----------------------------------------------------------------------===//

def CuteNvgpu_AtomGetValueOp : CuteNvgpu_Op<"atom_get_value"> {
  let summary = "Get runtime value from atom field";
  let description = [{
    Retrieves runtime-computed values like TMA descriptor pointers.
    
    Example:
    ```mlir
    %tma_desc = cute_nvgpu.atom_get_value %tma_atom, "tma_descriptor"
      : (!cute_nvgpu.tma_load<...>) -> !llvm.ptr
    ```
  }];
  
  let arguments = (ins
    AnyType:$atom,
    StrAttr:$field_name
  );
  let results = (outs AnyType:$result);
}

def CuteNvgpu_AtomSetValueOp : CuteNvgpu_Op<"atom_set_value"> {
  let summary = "Set runtime value for atom field";
  let description = [{
    Stores runtime values (e.g., TMA descriptor from kernel parameters).
    Must be called before using exec TMA operations.
  }];
  
  let arguments = (ins
    AnyType:$atom,
    StrAttr:$field_name,
    AnyType:$value
  );
}

//===----------------------------------------------------------------------===//
// SM80 (Ampere) MMA Operations
//===----------------------------------------------------------------------===//

def CuteNvgpu_WarpMmaF16BF16Op : CuteNvgpu_Op<"warp_mma_f16bf16"> {
  let summary = "Warp-level MMA for FP16/BF16 (SM80+)";
  let description = [{
    Executes warp-level matrix multiply-accumulate.
    Maps to mma.sync.aligned.m16n8k16 PTX instruction.
    
    Shapes: [16,8,16] or [16,8,8] depending on input types.
  }];
  
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    CuteNvgpu_MmaAtomType:$atom
  );
  let results = (outs AnyType:$fragD);
}

def CuteNvgpu_WarpMmaTF32Op : CuteNvgpu_Op<"warp_mma_tf32"> {
  let summary = "Warp-level MMA for TF32 (SM80+)";
  let description = [{
    TensorFloat32 (19-bit precision) MMA operation.
    Shape: [16, 8, 8]
  }];
  
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    CuteNvgpu_MmaAtomType:$atom
  );
  let results = (outs AnyType:$fragD);
}

def CuteNvgpu_WarpMmaSparseOp : CuteNvgpu_Op<"warp_mma_sparse"> {
  let summary = "Sparse warp MMA with 2:4 sparsity (SM80+)";
  let description = [{
    Structured sparsity MMA (50% zeros).
    Requires sparse metadata input.
  }];
  
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    AnyType:$metadata,
    CuteNvgpu_MmaAtomType:$atom,
    SparseMetadataFormat:$sparsity_format
  );
  let results = (outs AnyType:$fragD);
}

//===----------------------------------------------------------------------===//
// SM90 (Hopper) Warpgroup MMA Operations
//===----------------------------------------------------------------------===//

def CuteNvgpu_WarpgroupMmaOp : CuteNvgpu_Op<"warpgroup_mma"> {
  let summary = "Warpgroup-level MMA (SM90+)";
  let description = [{
    4-warp collective MMA operation (warpgroup = 128 threads).
    Shapes: [64, 64, 16], [64, 128, 16], [128, 64, 16], etc.
    
    Supports:
    - FP16, BF16, FP8, INT8
    - TMA async pipeline
    - Producer-consumer barriers
  }];
  
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    CuteNvgpu_MmaAtomType:$atom,
    OptionalAttr<I32Attr>:$warpgroup_id
  );
  let results = (outs AnyType:$fragD);
}

def CuteNvgpu_WarpgroupMmaSparseOp : CuteNvgpu_Op<"warpgroup_mma_sparse"> {
  let summary = "Sparse warpgroup MMA (SM90+)";
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    AnyType:$metadata,
    CuteNvgpu_MmaAtomType:$atom,
    SparseMetadataFormat:$sparsity_format
  );
  let results = (outs AnyType:$fragD);
}

//===----------------------------------------------------------------------===//
// SM100 (Blackwell) TCGEN05 MMA Operations
//===----------------------------------------------------------------------===//

def CuteNvgpu_Tcgen05MmaOp : CuteNvgpu_Op<"tcgen05_mma"> {
  let summary = "TCGEN05 MMA operation (SM100+)";
  let description = [{
    5th generation Tensor Core operations for Blackwell.
    Enhanced FP4/FP6 support and larger tile sizes.
  }];
  
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    CuteNvgpu_MmaAtomType:$atom
  );
  let results = (outs AnyType:$fragD);
}

def CuteNvgpu_Tcgen05BlockScaledMmaOp : CuteNvgpu_Op<"tcgen05_block_scaled_mma"> {
  let summary = "Block-scaled MMA for FP4/FP6 (SM100+)";
  let description = [{
    MMA with block-wise scaling factors for quantized datatypes.
  }];
  
  let arguments = (ins
    AnyType:$fragA,
    AnyType:$fragB,
    AnyType:$fragC,
    AnyType:$scaleA,
    AnyType:$scaleB,
    CuteNvgpu_MmaAtomType:$atom
  );
  let results = (outs AnyType:$fragD);
}

//===----------------------------------------------------------------------===//
// Copy Operations
//===----------------------------------------------------------------------===//

def CuteNvgpu_CopyUniversalOp : CuteNvgpu_Op<"copy_universal"> {
  let summary = "Universal async copy operation";
  let description = [{
    Selects optimal copy instruction based on:
    - Address spaces (global/shared/register)
    - Data alignment
    - GPU architecture
    
    Automatically chooses: TMA, cpasync, ldmatrix, or vector load/store.
  }];
  
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    CuteNvgpu_CopyAtomType:$atom
  );
}

def CuteNvgpu_LdmatrixOp : CuteNvgpu_Op<"ldmatrix"> {
  let summary = "Load matrix from shared memory (SM75+)";
  let description = [{
    ldmatrix.sync.aligned instruction.
    Loads 8x8 or 16x16 matrix from SMEM to registers.
  }];
  
  let arguments = (ins
    AnyType:$src_smem,
    I32Attr:$num_rows,
    BoolAttr:$transpose
  );
  let results = (outs AnyType:$result);
}

def CuteNvgpu_CpasyncG2SOp : CuteNvgpu_Op<"cpasync_g2s"> {
  let summary = "Async copy global→shared (SM80+)";
  let description = [{
    cp.async instruction family.
    Supports pipeline depth up to 8 stages.
  }];
  
  let arguments = (ins
    AnyType:$src_global,
    AnyType:$dst_smem,
    I32Attr:$bytes,
    OptionalAttr<BoolAttr>:$predicate
  );
}

def CuteNvgpu_CpasyncCommitGroupOp : CuteNvgpu_Op<"cpasync_commit_group"> {
  let summary = "Commit async copy group";
  let arguments = (ins);
}

def CuteNvgpu_CpasyncWaitGroupOp : CuteNvgpu_Op<"cpasync_wait_group"> {
  let summary = "Wait for async copy group completion";
  let arguments = (ins I32Attr:$n);
}

//===----------------------------------------------------------------------===//
// TMA Execution Operations (SM90+)
//===----------------------------------------------------------------------===//

def CuteNvgpu_TmaLoadExecuteOp : CuteNvgpu_Op<"tma_load_execute"> {
  let summary = "Execute TMA load operation";
  let description = [{
    Performs async TMA load from global to shared memory.
    Requires prior barrier arrival.
  }];
  
  let arguments = (ins
    CuteNvgpu_TmaLoadExecType:$tma,
    AnyType:$src_global,
    AnyType:$dst_smem,
    Variadic<Index>:$coords
  );
}

def CuteNvgpu_TmaStoreExecuteOp : CuteNvgpu_Op<"tma_store_execute"> {
  let summary = "Execute TMA store operation";
  let arguments = (ins
    CuteNvgpu_TmaStoreExecType:$tma,
    AnyType:$src_smem,
    AnyType:$dst_global,
    Variadic<Index>:$coords
  );
}

//===----------------------------------------------------------------------===//
// Barrier Operations (SM90+)
//===----------------------------------------------------------------------===//

def CuteNvgpu_MBarrierInitOp : CuteNvgpu_Op<"mbarrier_init"> {
  let summary = "Initialize Hopper mbarrier";
  let arguments = (ins
    AnyType:$barrier,
    I32Attr:$count
  );
}

def CuteNvgpu_MBarrierArriveOp : CuteNvgpu_Op<"mbarrier_arrive"> {
  let summary = "Signal arrival at barrier";
  let arguments = (ins AnyType:$barrier);
}

def CuteNvgpu_MBarrierWaitOp : CuteNvgpu_Op<"mbarrier_wait"> {
  let summary = "Wait at barrier until phase completion";
  let arguments = (ins 
    AnyType:$barrier,
    I32Attr:$phase
  );
}

#endif // CUTE_NVGPU_OPS
