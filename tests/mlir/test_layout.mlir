// Test basic layout creation and manipulation

module {
  func.func @test_make_layout(%i8: !rocir.int, %i16: !rocir.int, %i1: !rocir.int) -> !rocir.layout<2> {
    // Create shape and stride
    %shape = rocir.make_shape %i8, %i16 : (!rocir.int, !rocir.int) -> !rocir.shape<2>
    %stride = rocir.make_stride %i1, %i8 : (!rocir.int, !rocir.int) -> !rocir.stride<2>
    
    // Create layout from shape and stride
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
    
    return %layout : !rocir.layout<2>
  }
  
  func.func @test_layout_size(%arg0: !rocir.shape<2>) -> !rocir.int {
    // Query layout size
    %size = rocir.size %arg0 : !rocir.shape<2> -> !rocir.int
    
    return %size : !rocir.int
  }
  
  func.func @test_coord_to_index(%layout: !rocir.layout<2>, %i3: !rocir.int, %i5: !rocir.int) -> !rocir.int {
    %coord = rocir.make_coord %i3, %i5 : (!rocir.int, !rocir.int) -> !rocir.coord<2>
    
    // Convert coordinate to linear index
    %idx = rocir.crd2idx %coord, %layout : (!rocir.coord<2>, !rocir.layout<2>) -> !rocir.int
    
    return %idx : !rocir.int
  }
}
