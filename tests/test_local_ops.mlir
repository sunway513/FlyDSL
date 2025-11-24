// Test local_partition and local_tile operations

func.func @test_local_partition() -> index {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  
  // Create a 32x64 layout
  %shape = cute.make_shape %c32, %c64 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c32 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Create an 8x8 tile layout
  %tile_shape = cute.make_shape %c8, %c8 : (index, index) -> !cute.shape<2>
  %tile_stride = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Partition for thread 0
  %result = cute.local_partition %layout, %tile, %c0 : (!cute.layout<2>, !cute.layout<2>, index) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_local_tile() -> index {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  
  // Create a 128x256 layout (global tensor)
  %shape = cute.make_shape %c128, %c256 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c128 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Tile shape: 32x64 (CTA tile)
  %tile_shape = cute.make_shape %c32, %c64 : (index, index) -> !cute.shape<2>
  
  // Coordinate: (0, 0)
  %coord = cute.make_shape %c0, %c0 : (index, index) -> !cute.shape<2>
  
  // Extract tile at coordinate (0,0)
  %result = cute.local_tile %layout, %tile_shape, %coord : (!cute.layout<2>, !cute.shape<2>, !cute.shape<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_local_partition_thread() -> index {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  
  // Create a 16x16 layout
  %shape = cute.make_shape %c16, %c16 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Create a 2x2 thread tile
  %tile_shape = cute.make_shape %c2, %c2 : (index, index) -> !cute.shape<2>
  %tile_stride = cute.make_stride %c1, %c2 : (index, index) -> !cute.stride<2>
  %tile = cute.make_layout %tile_shape, %tile_stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Partition for thread 5
  %result = cute.local_partition %layout, %tile, %c5 : (!cute.layout<2>, !cute.layout<2>, index) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_local_tile_block() -> index {
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  // Create a 64x64 layout
  %shape = cute.make_shape %c64, %c64 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c64 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  // Tile shape: 16x16
  %tile_shape = cute.make_shape %c16, %c16 : (index, index) -> !cute.shape<2>
  
  // Coordinate: (2, 2) - extract tile at block (2,2)
  %coord = cute.make_shape %c2, %c2 : (index, index) -> !cute.shape<2>
  
  %result = cute.local_tile %layout, %tile_shape, %coord : (!cute.layout<2>, !cute.shape<2>, !cute.shape<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}
