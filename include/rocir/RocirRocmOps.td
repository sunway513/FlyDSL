//===- RocirRocmOps.td - CuTe ROCm Operations Definition ----*- tablegen -*-===//
//
// CuTe AMD GPU operations for GFX942
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_ROCM_OPS
#define ROCIR_ROCM_OPS

include "cute/RocirRocmDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// MFMA Operations
//===----------------------------------------------------------------------===//

def RocirRocm_MfmaOp : RocirRocm_OpBase<"mfma", [Pure]> {
  let summary = "MFMA (Matrix Fused Multiply-Add) operation for GFX942";
  let description = [{
    Performs matrix multiplication using AMD MFMA instructions on GFX942.
    
    Operation: D = A * B + C
    
    Supports various shapes and data types:
    - mfma_f32_32x32x8_f16: 32x32x8 with FP16 inputs, FP32 output
    - mfma_f32_16x16x16_f16: 16x16x16 with FP16 inputs, FP32 output
    - mfma_f64_16x16x4_f64: 16x16x4 with FP64 inputs/outputs
    - mfma_f32_32x32x16_bf16: 32x32x16 with BF16 inputs, FP32 output
    
    Example:
    ```mlir
    %d = cute_rocm.mfma %a, %b, %c {
      shape = [32, 32, 8],
      arch = "gfx942"
    } : (!cute.tensor<f16, layout_a>, !cute.tensor<f16, layout_b>, 
         !cute.tensor<f32, layout_c>) -> !cute.tensor<f32, layout_d>
    ```
  }];
  
  let arguments = (ins
    AnyType:$a,                          // A operand (LHS)
    AnyType:$b,                          // B operand (RHS)
    AnyType:$c,                          // C operand (accumulator)
    ArrayAttr:$shape,                    // [M, N, K] shape
    GfxArchEnum:$arch                    // Target architecture
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` 
    `(` type($a) `,` type($b) `,` type($c) `)` `->` type($result)
  }];
}

def RocirRocm_TiledMfmaOp : RocirRocm_OpBase<"tiled_mfma", [Pure]> {
  let summary = "Tiled MFMA operation across multiple wavefronts";
  let description = [{
    Performs a tiled matrix multiplication using MFMA instructions.
    Distributes computation across multiple wavefronts based on tiling pattern.
    
    Example:
    ```mlir
    %d = cute_rocm.tiled_mfma %a, %b, %c {
      atom = #cute_rocm.mfma_atom<[32, 32, 8], f16, f16, f32, gfx942>,
      tile_shape = [128, 128, 32]
    } : (!cute.tensor<...>, !cute.tensor<...>, !cute.tensor<...>) 
        -> !cute.tensor<...>
    ```
  }];
  
  let arguments = (ins
    AnyType:$a,
    AnyType:$b,
    AnyType:$c,
    TypeAttr:$atom,                      // MFMA atom descriptor
    ArrayAttr:$tile_shape                // Tiling pattern
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` 
    `(` type($a) `,` type($b) `,` type($c) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// LDS Operations
//===----------------------------------------------------------------------===//

def RocirRocm_LdsAllocOp : RocirRocm_OpBase<"lds_alloc"> {
  let summary = "Allocate LDS (Local Data Share) buffer";
  let description = [{
    Allocates a buffer in LDS memory with specified layout.
    LDS is AMD's equivalent to NVIDIA's shared memory.
    
    GFX942 provides 64KB LDS per compute unit.
    
    Example:
    ```mlir
    %lds = cute_rocm.lds_alloc {
      element_type = f16,
      layout = !cute.layout<...>,
      size = 32768
    } : !cute_rocm.lds_buffer<f16, ..., 32768>
    ```
  }];
  
  let arguments = (ins
    TypeAttr:$element_type,
    TypeAttr:$layout,
    I64Attr:$size
  );
  
  let results = (outs RocirRocm_LdsBufferType:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def RocirRocm_LdsLoadOp : RocirRocm_OpBase<"lds_load", [Pure]> {
  let summary = "Load data from LDS to registers";
  let description = [{
    Loads data from LDS buffer to register file.
    
    Example:
    ```mlir
    %data = cute_rocm.lds_load %lds_buffer, %coord {
      vector_size = 4
    } : (!cute_rocm.lds_buffer<f16, ...>, !cute.coord<2>) -> !cute.tensor<f16, ...>
    ```
  }];
  
  let arguments = (ins
    RocirRocm_LdsBufferType:$buffer,
    AnyType:$coord,
    I64Attr:$vector_size
  );
  
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $buffer `,` $coord attr-dict `:` 
    `(` type($buffer) `,` type($coord) `)` `->` type($result)
  }];
}

def RocirRocm_LdsStoreOp : RocirRocm_OpBase<"lds_store"> {
  let summary = "Store data from registers to LDS";
  let description = [{
    Stores data from register file to LDS buffer.
    
    Example:
    ```mlir
    cute_rocm.lds_store %data, %lds_buffer, %coord {
      vector_size = 4
    } : (!cute.tensor<f16, ...>, !cute_rocm.lds_buffer<f16, ...>, !cute.coord<2>)
    ```
  }];
  
  let arguments = (ins
    AnyType:$data,
    RocirRocm_LdsBufferType:$buffer,
    AnyType:$coord,
    I64Attr:$vector_size
  );
  
  let assemblyFormat = [{
    $data `,` $buffer `,` $coord attr-dict `:` 
    `(` type($data) `,` type($buffer) `,` type($coord) `)`
  }];
}

//===----------------------------------------------------------------------===//
// Copy Operations
//===----------------------------------------------------------------------===//

def RocirRocm_CopyOp : RocirRocm_OpBase<"copy"> {
  let summary = "Copy data with layout transformation";
  let description = [{
    Copies data from source to destination with optional layout transformation.
    Optimized for GFX942 memory hierarchy.
    
    Supports:
    - Global to LDS (buffer_load)
    - LDS to Register (ds_read)
    - Register to Global (buffer_store)
    - Register to LDS (ds_write)
    
    Example:
    ```mlir
    cute_rocm.copy %src, %dst {
      vector_size = 8,
      coalesced = true
    } : (!cute.tensor<f16, layout_src>, !cute.tensor<f16, layout_dst>)
    ```
  }];
  
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    I64Attr:$vector_size,
    BoolAttr:$coalesced
  );
  
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` `(` type($src) `,` type($dst) `)`
  }];
}

def RocirRocm_AsyncCopyOp : RocirRocm_OpBase<"async_copy"> {
  let summary = "Asynchronous copy operation";
  let description = [{
    Initiates an asynchronous copy from global memory to LDS.
    Uses buffer_load instructions with asynchronous semantics.
    
    Example:
    ```mlir
    %token = cute_rocm.async_copy %src, %dst {
      vector_size = 16
    } : (!cute.tensor<f16, ...>, !cute_rocm.lds_buffer<f16, ...>) 
        -> !cute_rocm.async_token
    ```
  }];
  
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    I64Attr:$vector_size
  );
  
  let results = (outs AnyType:$token);
  
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` 
    `(` type($src) `,` type($dst) `)` `->` type($token)
  }];
}

def RocirRocm_CopyWaitOp : RocirRocm_OpBase<"copy_wait"> {
  let summary = "Wait for async copy completion";
  let description = [{
    Waits for async copy operations to complete.
    Inserts s_waitcnt instruction.
    
    Example:
    ```mlir
    cute_rocm.copy_wait %token
    ```
  }];
  
  let arguments = (ins AnyType:$token);
  
  let assemblyFormat = "$token attr-dict `:` type($token)";
}

//===----------------------------------------------------------------------===//
// Synchronization Operations
//===----------------------------------------------------------------------===//

def RocirRocm_BarrierOp : RocirRocm_OpBase<"barrier"> {
  let summary = "Workgroup barrier synchronization";
  let description = [{
    Synchronizes all work items in a workgroup.
    Implements s_barrier instruction.
    
    Example:
    ```mlir
    cute_rocm.barrier
    ```
  }];
  
  let assemblyFormat = "attr-dict";
}

def RocirRocm_WavefrontBarrierOp : RocirRocm_OpBase<"wavefront_barrier"> {
  let summary = "Wavefront-level barrier";
  let description = [{
    Synchronizes all lanes within a wavefront.
    Uses s_waitcnt with appropriate flags.
    
    Example:
    ```mlir
    cute_rocm.wavefront_barrier
    ```
  }];
  
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def RocirRocm_ThreadIdOp : RocirRocm_OpBase<"thread_id", [Pure]> {
  let summary = "Get thread ID within workgroup";
  let description = [{
    Returns the thread ID in the specified dimension (x, y, z).
    
    Example:
    ```mlir
    %tid_x = cute_rocm.thread_id { dim = 0 } : i32
    %tid_y = cute_rocm.thread_id { dim = 1 } : i32
    ```
  }];
  
  let arguments = (ins I32Attr:$dim);
  let results = (outs I32:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def RocirRocm_WavefrontIdOp : RocirRocm_OpBase<"wavefront_id", [Pure]> {
  let summary = "Get wavefront ID within workgroup";
  let description = [{
    Returns the wavefront ID.
    
    Example:
    ```mlir
    %wid = cute_rocm.wavefront_id : i32
    ```
  }];
  
  let results = (outs I32:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

def RocirRocm_LaneIdOp : RocirRocm_OpBase<"lane_id", [Pure]> {
  let summary = "Get lane ID within wavefront";
  let description = [{
    Returns the lane ID (0-63 for GFX942).
    
    Example:
    ```mlir
    %lane = cute_rocm.lane_id : i32
    ```
  }];
  
  let results = (outs I32:$result);
  
  let assemblyFormat = "attr-dict `:` type($result)";
}

#endif // ROCIR_ROCM_OPS
