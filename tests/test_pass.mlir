module {
  func.func @test_crd2idx(%c: !cute.coord<2>, %l: !cute.layout<2>) -> !cute.int {
    %idx = cute.crd2idx %c, %l : (!cute.coord<2>, !cute.layout<2>) -> !cute.int
    return %idx : !cute.int
  }
}
