func.func @test1() {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  
  %shape = rocir.make_shape %c8, %c16 : (index, index) -> !rocir.shape<2>
  %stride = rocir.make_stride %c1, %c8 : (index, index) -> !rocir.stride<2>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %extracted_shape = rocir.get_shape %layout : !rocir.layout<2> -> !rocir.shape<2>
  %size = rocir.size %extracted_shape : !rocir.shape<2> -> index
  
  return
}
