// Test all cute operations
func.func @test_size() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = cute.make_shape %c8, %c16, %c32 : (index, index, index) -> !cute.shape<3>
  %size = cute.size %shape : !cute.shape<3> -> index
  
  return %size : index
}

func.func @test_get() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  
  %shape = cute.make_shape %c8, %c16, %c32 : (index, index, index) -> !cute.shape<3>
  %elem = cute.get %shape, %c1 : !cute.shape<3>, index -> index
  
  return %elem : index
}

func.func @test_rank() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  %shape = cute.make_shape %c8, %c16, %c32 : (index, index, index) -> !cute.shape<3>
  %rank = cute.rank %shape : !cute.shape<3> -> index
  
  return %rank : index
}

func.func @test_cosize() -> index {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = cute.make_shape %c8, %c128 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %cosize = cute.cosize %layout : !cute.layout<2> -> index
  
  return %cosize : index
}

func.func @test_crd2idx() -> index {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  
  %coord = cute.make_coord %c2, %c3 : (index, index) -> !cute.coord<2>
  %shape = cute.make_shape %c8, %c128 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %idx = cute.crd2idx %coord, %layout : (!cute.coord<2>, !cute.layout<2>) -> index
  
  return %idx : index
}

func.func @test_idx2crd() -> !cute.coord<2> {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  
  %shape = cute.make_shape %c8, %c128 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %coord = cute.idx2crd %c50, %layout : (index, !cute.layout<2>) -> !cute.coord<2>
  
  return %coord : !cute.coord<2>
}

func.func @test_get_shape() -> !cute.shape<2> {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = cute.make_shape %c8, %c128 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %extracted = cute.get_shape %layout : !cute.layout<2> -> !cute.shape<2>
  
  return %extracted : !cute.shape<2>
}

func.func @test_get_stride() -> !cute.stride<2> {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = cute.make_shape %c8, %c128 : (index, index) -> !cute.shape<2>
  %stride = cute.make_stride %c1, %c16 : (index, index) -> !cute.stride<2>
  %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  
  %extracted = cute.get_stride %layout : !cute.layout<2> -> !cute.stride<2>
  
  return %extracted : !cute.stride<2>
}
