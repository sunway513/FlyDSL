module {
  func.func @test_types() -> !cute.layout<2> {
    %layout = "test.dummy"() : () -> !cute.layout<2>
    return %layout : !cute.layout<2>
  }
  
  func.func @test_int_type() -> !cute.int {
    %i = "test.dummy"() : () -> !cute.int
    return %i : !cute.int
  }
  
  func.func @test_all_types(%s: !cute.shape<3>, %st: !cute.stride<3>, 
                            %l: !cute.layout<2>, %c: !cute.coord<2>) {
    return
  }
}
