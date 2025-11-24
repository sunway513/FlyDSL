// Test basic layout creation and manipulation

module {
  func.func @test_make_layout(%i8: !cute.int, %i16: !cute.int, %i1: !cute.int) -> !cute.layout<2> {
    // Create shape and stride
    %shape = cute.make_shape %i8, %i16 : (!cute.int, !cute.int) -> !cute.shape<2>
    %stride = cute.make_stride %i1, %i8 : (!cute.int, !cute.int) -> !cute.stride<2>
    
    // Create layout from shape and stride
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    
    return %layout : !cute.layout<2>
  }
  
  func.func @test_layout_size(%arg0: !cute.shape<2>) -> !cute.int {
    // Query layout size
    %size = cute.size %arg0 : !cute.shape<2> -> !cute.int
    
    return %size : !cute.int
  }
  
  func.func @test_coord_to_index(%layout: !cute.layout<2>, %i3: !cute.int, %i5: !cute.int) -> !cute.int {
    %coord = cute.make_coord %i3, %i5 : (!cute.int, !cute.int) -> !cute.coord<2>
    
    // Convert coordinate to linear index
    %idx = cute.crd2idx %coord, %layout : (!cute.coord<2>, !cute.layout<2>) -> !cute.int
    
    return %idx : !cute.int
  }
}
