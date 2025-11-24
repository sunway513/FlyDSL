// Test basic type parsing and printing

module {
  func.func @test_types(%i1: !cute.int, %i2: !cute.int) -> !cute.layout<2> {
    %shape = cute.make_shape %i1, %i2 : (!cute.int, !cute.int) -> !cute.shape<2>
    %stride = cute.make_stride %i1, %i2 : (!cute.int, !cute.int) -> !cute.stride<2>
    %layout = cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
    return %layout : !cute.layout<2>
  }
  
  func.func @test_int_type(%i: !cute.int) -> !cute.int {
    return %i : !cute.int
  }
  
  func.func @test_all_types(%s: !cute.shape<3>, %st: !cute.stride<3>, 
                            %l: !cute.layout<2>, %c: !cute.coord<2>) {
    return
  }
}
