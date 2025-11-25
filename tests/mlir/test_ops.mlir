module {
  func.func @test_cute_ops(%i1: !rocir.int, %i2: !rocir.int, %i3: !rocir.int) {
    %s = rocir.make_shape %i1, %i2, %i3 : (!rocir.int, !rocir.int, !rocir.int) -> !rocir.shape<3>
    %st = rocir.make_stride %i1, %i2, %i3 : (!rocir.int, !rocir.int, !rocir.int) -> !rocir.stride<3>
    %l = rocir.make_layout %s, %st : (!rocir.shape<3>, !rocir.stride<3>) -> !rocir.layout<3>
    %c = rocir.make_coord %i1, %i2 : (!rocir.int, !rocir.int) -> !rocir.coord<2>
    %size = rocir.size %s : !rocir.shape<3> -> !rocir.int
    %idx = rocir.crd2idx %c, %l : (!rocir.coord<2>, !rocir.layout<3>) -> !rocir.int
    return
  }
}
