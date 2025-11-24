module {
  func.func @test_cute_ops(%i1: !cute.int, %i2: !cute.int, %i3: !cute.int) {
    %s = cute.make_shape %i1, %i2, %i3 : (!cute.int, !cute.int, !cute.int) -> !cute.shape<3>
    %st = cute.make_stride %i1, %i2, %i3 : (!cute.int, !cute.int, !cute.int) -> !cute.stride<3>
    %l = cute.make_layout %s, %st : (!cute.shape<3>, !cute.stride<3>) -> !cute.layout<3>
    %c = cute.make_coord %i1, %i2 : (!cute.int, !cute.int) -> !cute.coord<2>
    %size = cute.size %s : !cute.shape<3> -> !cute.int
    %idx = cute.crd2idx %c, %l : (!cute.coord<2>, !cute.layout<3>) -> !cute.int
    return
  }
}
