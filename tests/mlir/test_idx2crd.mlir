// Test rocir.idx2crd operation (inverse of crd2idx)
func.func @test_idx2crd() -> !rocir.coord<2> {
  %c50 = arith.constant 50 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c8 = arith.constant 8 : index
  %c128 = arith.constant 128 : index
  
  // Create layout with shape (8, 128) and stride (1, 16)
  %shape = rocir.make_shape %c8, %c128 : (index, index) -> !rocir.shape<2>
  %stride = rocir.make_stride %c1, %c16 : (index, index) -> !rocir.stride<2>
  %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
  
  // Convert linear index 50 back to coordinate
  // Expected: (2, 3) since 50 = 2*1 + 3*16
  %coord = rocir.idx2crd %c50, %layout : (index, !rocir.layout<2>) -> !rocir.coord<2>
  
  return %coord : !rocir.coord<2>
}
