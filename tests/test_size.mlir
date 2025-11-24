// Test cute.size operation
func.func @test_size() -> index {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  
  // Create shape (8, 16, 32)
  %shape = cute.make_shape %c8, %c16, %c32 : (index, index, index) -> !cute.shape<3>
  
  // Compute size = 8 * 16 * 32 = 4096
  %size = cute.size %shape : (!cute.shape<3>) -> index
  
  return %size : index
}
