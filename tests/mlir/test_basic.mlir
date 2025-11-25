// Test basic type parsing and printing

module {
  func.func @test_types(%i1: !rocir.int, %i2: !rocir.int) -> !rocir.layout<2> {
    %shape = rocir.make_shape %i1, %i2 : (!rocir.int, !rocir.int) -> !rocir.shape<2>
    %stride = rocir.make_stride %i1, %i2 : (!rocir.int, !rocir.int) -> !rocir.stride<2>
    %layout = rocir.make_layout %shape, %stride : (!rocir.shape<2>, !rocir.stride<2>) -> !rocir.layout<2>
    return %layout : !rocir.layout<2>
  }
  
  func.func @test_int_type(%i: !rocir.int) -> !rocir.int {
    return %i : !rocir.int
  }
  
  func.func @test_all_types(%s: !rocir.shape<3>, %st: !rocir.stride<3>, 
                            %l: !rocir.layout<2>, %c: !rocir.coord<2>) {
    return
  }
}
