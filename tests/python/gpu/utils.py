"""Shared utilities for GPU testing and compilation."""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.compiler.rocir_opt_helper import apply_rocir_coord_lowering
from rocdsl.runtime.hip_util import get_hip_arch


def compile_to_hsaco(mlir_module):
    """
    Compile MLIR module to HSACO binary for AMD GPUs.
    
    Pipeline:
    1. Apply rocir coordinate lowering (rocir ops -> arithmetic)
    2. Canonicalize and CSE
    3. Attach ROCDL target for current GPU architecture
    4. Convert GPU dialect to ROCDL
    5. Lower to LLVM
    6. Generate binary
    
    Args:
        mlir_module: MLIR module containing GPU kernels
        
    Returns:
        bytes: HSACO binary object
    """
    # Apply rocir coordinate lowering first
    lowered_module = apply_rocir_coord_lowering(mlir_module)
    
    # Get the current GPU architecture
    gpu_arch = get_hip_arch()
    
    # Then run the main GPU compilation pipeline
    lowered = run_pipeline(
        lowered_module,
        Pipeline()
        .canonicalize()
        .cse()
        .rocdl_attach_target(chip=gpu_arch)
        .Gpu(Pipeline()
             .convert_scf_to_cf()  # Lower SCF loops first
             .convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP")
             .reconcile_unrealized_casts())  # Clean up inside GPU module
        .gpu_to_llvm()
        .reconcile_unrealized_casts()  # Clean up type conversions
        .lower_to_llvm()
        .reconcile_unrealized_casts()  # Clean up again after lowering
        .gpu_module_to_binary(format="bin")
    )
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)
