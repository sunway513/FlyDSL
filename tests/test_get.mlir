// Test cute.get operation
func.func @test_get() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  
  // Create shape (8, 16, 32)
  %shape = cute.make_shape %c8, %c16, %c32 : (index, index, index) -> !cute.shape<3>
  
  // Get element at index 1 (should return 16)
  %elem = cute.get %shape, %c1 : (!cute.shape<3>, index) -> index
  
  return %elem : index
}
