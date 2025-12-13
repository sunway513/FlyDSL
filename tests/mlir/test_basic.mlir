// Test basic type parsing and printing

module {
  func.func @test_types(%i1: index, %i2: index) -> !rocir.layout<2> {
    %shape = rocir.make_shape %i1, %i2 : (index, index) -> !rocir.shape<(?,?)>
    %stride = rocir.make_stride %i1, %i2 : (index, index) -> !rocir.stride<(?,?)>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<(?,?)>, !rocir.stride<(?,?)>) -> !rocir.layout<2>
    return %layout : !rocir.layout<2>
  }
  
  func.func @test_index_type(%i: index) -> index {
    return %i : index
  }
  
  func.func @test_all_types(%s: !rocir.shape<(?,?,?)>, %st: !rocir.stride<(?,?,?)>, 
                            %l: !rocir.layout<2>, %c: !rocir.coord<2>) {
    return
  }
}
