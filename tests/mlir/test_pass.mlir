// Test partial pass lowering (crd2idx only)
module {
  func.func @test_crd2idx_simple(%c: !rocir.coord<2>, %l: !rocir.layout<2>) {
    %idx = rocir.crd2idx %c, %l : (!rocir.coord<2>, !rocir.layout<2>) -> !rocir.int
    // Note: %idx is lowered to index type but not used, so no type conversion error
    return
  }
}
