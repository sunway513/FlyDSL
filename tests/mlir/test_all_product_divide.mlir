// Test all product and divide operations

func.func @test_tiled_product() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_block = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<2>
  %stride_block = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<2>
  %block = rocir.make_layout %shape_block, %stride_block : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %shape_tiler = rocir.make_shape %c2, %c2 : (index, index) -> !rocir.shape<2>
  %stride_tiler = rocir.make_stride %c1, %c2 : (index, index) -> !rocir.stride<2>
  %tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %result = rocir.tiled_product %block, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  %size = rocir.size %result : !rocir.layout<2> -> index
  return %size : index
}

func.func @test_flat_product() -> index {
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  %shape_block = rocir.make_shape %c8, %c4 : (index, index) -> !rocir.shape<2>
  %stride_block = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<2>
  %block = rocir.make_layout %shape_block, %stride_block : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %shape_tiler = rocir.make_shape %c4, %c4 : (index, index) -> !rocir.shape<2>
  %stride_tiler = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<2>
  %tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %result = rocir.flat_product %block, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  %size = rocir.size %result : !rocir.layout<2> -> index
  return %size : index
}

func.func @test_raked_product() -> index {
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  
  %shape_block = rocir.make_shape %c16, %c8 : (index, index) -> !rocir.shape<2>
  %stride_block = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<2>
  %block = rocir.make_layout %shape_block, %stride_block : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %shape_tiler = rocir.make_shape %c2, %c2 : (index, index) -> !rocir.shape<2>
  %stride_tiler = rocir.make_stride %c1, %c2 : (index, index) -> !rocir.stride<2>
  %tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %result = rocir.raked_product %block, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  %size = rocir.size %result : !rocir.layout<2> -> index
  return %size : index
}

func.func @test_blocked_product() -> index {
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  
  %shape_block = rocir.make_shape %c32, %c16 : (index, index) -> !rocir.shape<2>
  %stride_block = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<2>
  %block = rocir.make_layout %shape_block, %stride_block : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %shape_tiler = rocir.make_shape %c4, %c4 : (index, index) -> !rocir.shape<2>
  %stride_tiler = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<2>
  %tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %result = rocir.blocked_product %block, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  %size = rocir.size %result : !rocir.layout<2> -> index
  return %size : index
}

func.func @test_zipped_divide() -> index {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  
  %shape_layout = rocir.make_shape %c32, %c64 : (index, index) -> !rocir.shape<2>
  %stride_layout = rocir.make_stride %c1, %c32 : (index, index) -> !rocir.stride<2>
  %layout = rocir.make_layout %shape_layout, %stride_layout : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %shape_tiler = rocir.make_shape %c8, %c8 : (index, index) -> !rocir.shape<2>
  %stride_tiler = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<2>
  %tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %result = rocir.zipped_divide %layout, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  %size = rocir.size %result : !rocir.layout<2> -> index
  return %size : index
}

func.func @test_flat_divide() -> index {
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  
  %shape_layout = rocir.make_shape %c16, %c32 : (index, index) -> !rocir.shape<2>
  %stride_layout = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<2>
  %layout = rocir.make_layout %shape_layout, %stride_layout : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %shape_tiler = rocir.make_shape %c4, %c4 : (index, index) -> !rocir.shape<2>
  %stride_tiler = rocir.make_stride %c1, %c4 : (index, index) -> !rocir.stride<2>
  %tiler = rocir.make_layout %shape_tiler, %stride_tiler : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %result = rocir.flat_divide %layout, %tiler : (!rocir.layout<2>, !rocir.layout<2>) -> !rocir.layout<2>
  %size = rocir.size %result : !rocir.layout<2> -> index
  return %size : index
}
