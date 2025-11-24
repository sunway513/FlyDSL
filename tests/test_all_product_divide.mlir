// Test all product and divide operations

func.func @test_tiled_product() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_block = cute.make_shape %c8, %c16 : (index, index) -> !cute.shape<2>
  %stride_block = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %block = cute.make_layout %shape_block, %stride_block : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %shape_tiler = cute.make_shape %c2, %c2 : (index, index) -> !cute.shape<2>
  %stride_tiler = cute.make_stride %c1, %c2 : (index, index) -> !cute.stride<2>
  %tiler = cute.make_layout %shape_tiler, %stride_tiler : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %result = cute.tiled_product %block, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_flat_product() -> index {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  %shape_block = cute.make_shape %c8, %c4 : (index, index) -> !cute.shape<2>
  %stride_block = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %block = cute.make_layout %shape_block, %stride_block : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %shape_tiler = cute.make_shape %c4, %c4 : (index, index) -> !cute.shape<2>
  %stride_tiler = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
  %tiler = cute.make_layout %shape_tiler, %stride_tiler : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %result = cute.flat_product %block, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_raked_product() -> index {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_block = cute.make_shape %c16, %c8 : (index, index) -> !cute.shape<2>
  %stride_block = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %block = cute.make_layout %shape_block, %stride_block : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %shape_tiler = cute.make_shape %c2, %c2 : (index, index) -> !cute.shape<2>
  %stride_tiler = cute.make_stride %c1, %c2 : (index, index) -> !cute.stride<2>
  %tiler = cute.make_layout %shape_tiler, %stride_tiler : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %result = cute.raked_product %block, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_blocked_product() -> index {
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  
  %shape_block = cute.make_shape %c32, %c16 : (index, index) -> !cute.shape<2>
  %stride_block = cute.make_stride %c1, %c32 : (index, index) -> !cute.stride<2>
  %block = cute.make_layout %shape_block, %stride_block : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %shape_tiler = cute.make_shape %c4, %c4 : (index, index) -> !cute.shape<2>
  %stride_tiler = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
  %tiler = cute.make_layout %shape_tiler, %stride_tiler : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %result = cute.blocked_product %block, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_zipped_divide() -> index {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  
  %shape_layout = cute.make_shape %c32, %c64 : (index, index) -> !cute.shape<2>
  %stride_layout = cute.make_stride %c1, %c32 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape_layout, %stride_layout : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %shape_tiler = cute.make_shape %c8, %c8 : (index, index) -> !cute.shape<2>
  %stride_tiler = cute.make_stride %c1, %c8 : (index, index) -> !cute.stride<2>
  %tiler = cute.make_layout %shape_tiler, %stride_tiler : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %result = cute.zipped_divide %layout, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}

func.func @test_flat_divide() -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  
  %shape_layout = cute.make_shape %c16, %c32 : (index, index) -> !cute.shape<2>
  %stride_layout = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape_layout, %stride_layout : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %shape_tiler = cute.make_shape %c4, %c4 : (index, index) -> !cute.shape<2>
  %stride_tiler = cute.make_stride %c1, %c4 : (index, index) -> !cute.stride<2>
  %tiler = cute.make_layout %shape_tiler, %stride_tiler : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %result = cute.flat_divide %layout, %tiler : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  %size = cute.size %result : !cute.layout<2> -> index
  return %size : index
}
