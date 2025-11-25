// Test rocir.cosize operation
func.func @test_cosize() -> index {
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  
  %shape = rocir.make_shape %c8, %c128 : (index, index) -> !rocir.shape<2>
  %stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<2>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  %cosize = rocir.cosize %layout : !rocir.layout<2> -> index
  
  return %cosize : index
}
