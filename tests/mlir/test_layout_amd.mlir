// RUN: mlir-opt %s -pass-pipeline='builtin.module(cute-to-rocm)' | FileCheck %s

// Test layout operations for AMD GFX942

module attributes {
  rocir.target_arch = "gfx942",
  rocir.target_vendor = "amd"
} {
  
  // Test basic layout creation for AMD
  func.func @test_make_layout_amd() -> !rocir.layout<2> {
    // Create shape and stride for MFMA-friendly layout
    // GFX942 MFMA prefers 32x32 or 16x16 tiles
    %shape = rocir.make_shape 32, 32 : !rocir.shape<2>
    %stride = rocir.make_stride 1, 32 : !rocir.stride<2>
    
    // Create layout from shape and stride
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<2>
    
    // CHECK: rocir.make_layout
    // CHECK-DAG: 32
    // CHECK-DAG: 32
    return %layout : !rocir.layout<2>
  }
  
  // Test layout size query
  func.func @test_layout_size_amd(%arg0: !rocir.layout<2>) -> !rocir.int {
    %size = rocir.size %arg0 : !rocir.layout<2> -> !rocir.int
    
    // CHECK: rocir.size
    return %size : !rocir.int
  }
  
  // Test coordinate to index conversion
  // This is fundamental for MFMA data layout
  func.func @test_coord_to_index_amd() {
    // 32x32 layout matching MFMA tile
    %shape = rocir.make_shape 32, 32 : !rocir.shape<2>
    %stride = rocir.make_stride 1, 32 : !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<2>
    
    %coord = rocir.make_coord 8, 16 : !rocir.coord<2>
    
    // Convert coordinate to linear index
    // Expected: 8 + 16*32 = 8 + 512 = 520
    %idx = rocir.crd2idx %layout, %coord : !rocir.layout<2>, !rocir.coord<2> -> !rocir.int
    
    // CHECK: rocir.crd2idx
    return
  }
  
  // Test MFMA-compatible layout (16x16x16 for FP16)
  func.func @test_mfma_16x16x16_layout() -> !rocir.layout<3> {
    // MFMA f32_16x16x16_f16 layout
    %shape = rocir.make_shape 16, 16, 16 : !rocir.shape<3>
    %stride = rocir.make_stride 1, 16, 256 : !rocir.stride<3>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<3>
    
    // CHECK: rocir.make_layout
    // CHECK-DAG: 16
    return %layout : !rocir.layout<3>
  }
  
  // Test MFMA-compatible layout (32x32x8 for FP16)
  func.func @test_mfma_32x32x8_layout() -> !rocir.layout<3> {
    // MFMA f32_32x32x8_f16 layout
    %shape = rocir.make_shape 32, 32, 8 : !rocir.shape<3>
    %stride = rocir.make_stride 1, 32, 1024 : !rocir.stride<3>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<3>
    
    // CHECK: rocir.make_layout
    // CHECK-DAG: 32
    // CHECK-DAG: 8
    return %layout : !rocir.layout<3>
  }
  
  // Test LDS-friendly layout (avoid bank conflicts)
  // GFX942 has 32 LDS banks, 4-byte bank width
  func.func @test_lds_layout() -> !rocir.layout<2> {
    // Padded layout to avoid bank conflicts
    // Shape: 64 x 64, Stride: 1 x 68 (68 = 64 + 4 padding)
    %shape = rocir.make_shape 64, 64 : !rocir.shape<2>
    %stride = rocir.make_stride 1, 68 : !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<2>
    
    // CHECK: rocir.make_layout
    // CHECK: 68
    return %layout : !rocir.layout<2>
  }
  
  // Test wavefront-partitioned layout
  // GFX942 has wavefront size = 64
  func.func @test_wavefront_partitioned_layout() -> !rocir.layout<2> {
    // Layout partitioned across 64 lanes
    // Each lane handles 4 elements
    %shape = rocir.make_shape 64, 4 : !rocir.shape<2>
    %stride = rocir.make_stride 1, 64 : !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<2>
    
    // CHECK: rocir.make_layout
    // CHECK-DAG: 64
    // CHECK-DAG: 4
    return %layout : !rocir.layout<2>
  }
  
  // Test hierarchical layout for tiled MFMA
  func.func @test_tiled_mfma_layout() -> !rocir.layout<2> {
    // Outer tile: 4x4 MFMA atoms
    // Inner tile: 32x32 MFMA shape
    // Total: 128x128
    %shape = rocir.make_shape 128, 128 : !rocir.shape<2>
    %stride = rocir.make_stride 1, 128 : !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<2>
    
    // CHECK: rocir.make_layout
    // CHECK-DAG: 128
    return %layout : !rocir.layout<2>
  }
  
  // Test coalesced global memory layout
  // GFX942 prefers 128-byte (32 x f32) aligned coalescing
  func.func @test_coalesced_global_layout() -> !rocir.layout<2> {
    // Layout ensuring 128-byte alignment
    // 32 threads × 4 elements (f32) = 128 bytes
    %shape = rocir.make_shape 32, 1024 : !rocir.shape<2>
    %stride = rocir.make_stride 1, 32 : !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<2>
    
    // CHECK: rocir.make_layout
    return %layout : !rocir.layout<2>
  }
  
  // Test layout composition for GEMM on GFX942
  func.func @test_gemm_layout_gfx942() {
    // A matrix layout (M x K)
    %shape_a = rocir.make_shape 128, 32 : !rocir.shape<2>
    %stride_a = rocir.make_stride 1, 128 : !rocir.stride<2>
    %layout_a = rocir.make_layout %shape_a, %stride_a : !rocir.layout<2>
    
    // B matrix layout (K x N)
    %shape_b = rocir.make_shape 32, 128 : !rocir.shape<2>
    %stride_b = rocir.make_stride 1, 32 : !rocir.stride<2>
    %layout_b = rocir.make_layout %shape_b, %stride_b : !rocir.layout<2>
    
    // C matrix layout (M x N)
    %shape_c = rocir.make_shape 128, 128 : !rocir.shape<2>
    %stride_c = rocir.make_stride 1, 128 : !rocir.stride<2>
    %layout_c = rocir.make_layout %shape_c, %stride_c : !rocir.layout<2>
    
    // CHECK: rocir.make_layout
    // CHECK: rocir.make_layout
    // CHECK: rocir.make_layout
    return
  }
  
  // Test layout for BF16 MFMA (32x32x16)
  func.func @test_mfma_bf16_layout() -> !rocir.layout<3> {
    // MFMA f32_32x32x16_bf16 layout
    %shape = rocir.make_shape 32, 32, 16 : !rocir.shape<3>
    %stride = rocir.make_stride 1, 32, 1024 : !rocir.stride<3>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<3>
    
    // CHECK: rocir.make_layout
    return %layout : !rocir.layout<3>
  }
  
  // Test layout for FP64 MFMA (16x16x4)
  func.func @test_mfma_f64_layout() -> !rocir.layout<3> {
    // MFMA f64_16x16x4_f64 layout
    %shape = rocir.make_shape 16, 16, 4 : !rocir.shape<3>
    %stride = rocir.make_stride 1, 16, 256 : !rocir.stride<3>
    %layout = rocir.make_layout %shape, %stride : !rocir.layout<3>
    
    // CHECK: rocir.make_layout
    return %layout : !rocir.layout<3>
  }
}
