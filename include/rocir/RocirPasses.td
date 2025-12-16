//===- RocirPasses.td - Layout IR Lowering Passes ----------*- tablegen -*-===//
//
// Lowering passes for Rocir IR dialects
// Transformation chain: rocdsl → gpu/scf → rocdl → LLVM IR → AMDGCN
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_PASSES
#define ROCIR_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// rocdsl → Standard Dialects (scf, arith, memref)
//===----------------------------------------------------------------------===//

// Run at module scope so we can lower Rocir ops inside GPU kernels (`gpu.func`)
// as well as regular `func.func`.
def RocirToStandardPass : Pass<"rocir-to-standard", "::mlir::ModuleOp"> {
  let summary = "Lower Rocir IR to standard MLIR dialects";
  let description = [{
    Converts rocdsl operations to combinations of scf, arith, and memref.
  }];
  
  let constructor = "mlir::rocir::createRocirToStandardPass()";
  let dependentDialects = [
    "::mlir::scf::SCFDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Dead Code Elimination (standalone / general)
//===----------------------------------------------------------------------===//

// Upstream MLIR's generic 'dce/adce' passes are not registered in our embedded
// Python environment. Provide a standalone, general DCE pass with a distinct
// name to avoid conflicting with upstream if it becomes available later.
def RocirTrivialDCEPass : Pass<"trivial-dce", "::mlir::ModuleOp"> {
  let summary = "Erase trivially-dead operations (general DCE)";
  let description = [{
    Iteratively erases operations that are unused and have no side effects, as
    determined by MLIR's side-effect interfaces.
  }];
  let constructor = "mlir::rocir::createRocirTrivialDCEPass()";
}

//===----------------------------------------------------------------------===//
// Layout Canonicalization
//===----------------------------------------------------------------------===//

def RocirLayoutCanonicalizePass : Pass<"rocir-layout-canonicalize", "::mlir::func::FuncOp"> {
  let summary = "Canonicalize and simplify Layout operations";
  let description = [{
    Simplifies layout algebra operations through algebraic identities.
  }];
  
  let constructor = "mlir::rocir::createRocirLayoutCanonicalizePass()";
}

//===----------------------------------------------------------------------===//
// Rocir ROCm IR → gpu/rocdl dialect (AMD GPU specific)
//===----------------------------------------------------------------------===//

def RocirRocmToGPUPass : Pass<"rocir-rocm-to-gpu", "::mlir::gpu::GPUModuleOp"> {
  let summary = "Lower Rocir ROCm IR to MLIR GPU/ROCDL dialect";
  let description = [{
    Converts Rocir ROCm operations to GPU and ROCDL operations for AMD GPUs.
    
    Key Transformations:
    - rocir.rocm.mfma → rocdl.mfma
    - rocir.rocm.lds_* → gpu.barrier + rocdl.ds_*
    - rocir.rocm.async_copy → rocdl.buffer_load with async
    - rocir.rocm.barrier → gpu.barrier
  }];
  
  let constructor = "mlir::rocir::createRocirRocmToGPUPass()";
  let dependentDialects = [
    "::mlir::gpu::GPUDialect",
    "::mlir::ROCDL::ROCDLDialect",
    "::mlir::vector::VectorDialect"
  ];
  
  let options = [
    Option<"targetArch", "target-arch", "std::string",
           /*default=*/"\"gfx942\"",
           "Target AMD GPU architecture (gfx90a, gfx942, etc.)">
  ];
}

#endif // ROCIR_PASSES
