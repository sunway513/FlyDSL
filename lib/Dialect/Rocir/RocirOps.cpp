//===- RocirOps.cpp - Rocir Operations Implementation ---------------------===//
//
// Implementation of Rocir dialect operations
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirOps.h"
#include "rocir/RocirDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rocir;

//===----------------------------------------------------------------------===//
// ProductEachOp
//===----------------------------------------------------------------------===//

LogicalResult ProductEachOp::verify() {
  // Verify operand is a ShapeType
  auto shapeType = getShape().getType();
  if (!shapeType.isa<ShapeType>()) {
    return emitOpError("operand must be of ShapeType");
  }
  
  // Verify result is also a ShapeType
  auto resultType = getResult().getType();
  if (!resultType.isa<ShapeType>()) {
    return emitOpError("result must be of ShapeType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// MakeLayoutTVOp
//===----------------------------------------------------------------------===//

LogicalResult MakeLayoutTVOp::verify() {
  // Verify thr_layout and val_layout are LayoutType
  auto thrLayoutType = getThrLayout().getType();
  auto valLayoutType = getValLayout().getType();
  
  if (!thrLayoutType.isa<LayoutType>()) {
    return emitOpError("thr_layout operand must be of LayoutType");
  }
  
  if (!valLayoutType.isa<LayoutType>()) {
    return emitOpError("val_layout operand must be of LayoutType");
  }
  
  // Verify tiler_mn result is ShapeType
  auto tilerType = getTilerMn().getType();
  if (!tilerType.isa<ShapeType>()) {
    return emitOpError("tiler_mn result must be of ShapeType");
  }
  
  // Verify layout_tv result is LayoutType
  auto layoutTVType = getLayoutTv().getType();
  if (!layoutTVType.isa<LayoutType>()) {
    return emitOpError("layout_tv result must be of LayoutType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Make Operations
//===----------------------------------------------------------------------===//

LogicalResult MakeShapeOp::verify() {
  // Verify all operands are either Index or ShapeType
  for (auto operand : getValues()) {
    auto type = operand.getType();
    if (!type.isa<IndexType>() && !type.isa<ShapeType>()) {
      return emitOpError("operands must be either Index or ShapeType");
    }
  }
  
  // Verify result is ShapeType
  if (!getResult().getType().isa<ShapeType>()) {
    return emitOpError("result must be of ShapeType");
  }
  
  return success();
}

LogicalResult MakeStrideOp::verify() {
  // Verify all operands are either Index or StrideType
  for (auto operand : getValues()) {
    auto type = operand.getType();
    if (!type.isa<IndexType>() && !type.isa<StrideType>()) {
      return emitOpError("operands must be either Index or StrideType");
    }
  }
  
  // Verify result is StrideType
  if (!getResult().getType().isa<StrideType>()) {
    return emitOpError("result must be of StrideType");
  }
  
  return success();
}

LogicalResult MakeLayoutOp::verify() {
  // Verify shape is ShapeType
  if (!getShape().getType().isa<ShapeType>()) {
    return emitOpError("shape operand must be of ShapeType");
  }
  
  // Verify stride is StrideType
  if (!getStride().getType().isa<StrideType>()) {
    return emitOpError("stride operand must be of StrideType");
  }
  
  // Verify result is LayoutType
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

LogicalResult MakeCoordOp::verify() {
  // Verify all operands are Index
  for (auto operand : getValues()) {
    if (!operand.getType().isa<IndexType>()) {
      return emitOpError("all operands must be of Index type");
    }
  }
  
  // Verify result is CoordType
  if (!getResult().getType().isa<CoordType>()) {
    return emitOpError("result must be of CoordType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Query Operations
//===----------------------------------------------------------------------===//

LogicalResult SizeOp::verify() {
  // Verify input is ShapeType or LayoutType
  auto inputType = getInput().getType();
  if (!inputType.isa<ShapeType>() && !inputType.isa<LayoutType>()) {
    return emitOpError("input must be either ShapeType or LayoutType");
  }
  
  // Verify result is Index
  if (!getResult().getType().isa<IndexType>()) {
    return emitOpError("result must be of Index type");
  }
  
  return success();
}

LogicalResult CosizeOp::verify() {
  // Verify input is LayoutType
  if (!getLayout().getType().isa<LayoutType>()) {
    return emitOpError("input must be of LayoutType");
  }
  
  // Verify result is Index
  if (!getResult().getType().isa<IndexType>()) {
    return emitOpError("result must be of Index type");
  }
  
  return success();
}

LogicalResult GetShapeOp::verify() {
  // Verify input is LayoutType
  if (!getLayout().getType().isa<LayoutType>()) {
    return emitOpError("input must be of LayoutType");
  }
  
  // Verify result is ShapeType
  if (!getResult().getType().isa<ShapeType>()) {
    return emitOpError("result must be of ShapeType");
  }
  
  return success();
}

LogicalResult GetStrideOp::verify() {
  // Verify input is LayoutType
  if (!getLayout().getType().isa<LayoutType>()) {
    return emitOpError("input must be of LayoutType");
  }
  
  // Verify result is StrideType
  if (!getResult().getType().isa<StrideType>()) {
    return emitOpError("result must be of StrideType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Composition Operations
//===----------------------------------------------------------------------===//

LogicalResult CompositionOp::verify() {
  // Verify both operands are LayoutType
  if (!getLhs().getType().isa<LayoutType>()) {
    return emitOpError("lhs operand must be of LayoutType");
  }
  
  if (!getRhs().getType().isa<LayoutType>()) {
    return emitOpError("rhs operand must be of LayoutType");
  }
  
  // Verify result is LayoutType
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

LogicalResult RightInverseOp::verify() {
  // Verify input is LayoutType
  if (!getLayout().getType().isa<LayoutType>()) {
    return emitOpError("input must be of LayoutType");
  }
  
  // Verify result is LayoutType
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Product Operations
//===----------------------------------------------------------------------===//

LogicalResult RakedProductOp::verify() {
  // Verify both operands are LayoutType
  if (!getInput().getType().isa<LayoutType>()) {
    return emitOpError("input operand must be of LayoutType");
  }
  
  if (!getTiler().getType().isa<LayoutType>()) {
    return emitOpError("tiler operand must be of LayoutType");
  }
  
  // Verify result is LayoutType
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

LogicalResult BlockedProductOp::verify() {
  // Verify both operands are LayoutType
  if (!getInput().getType().isa<LayoutType>()) {
    return emitOpError("input operand must be of LayoutType");
  }
  
  if (!getTiler().getType().isa<LayoutType>()) {
    return emitOpError("tiler operand must be of LayoutType");
  }
  
  // Verify result is LayoutType
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Divide Operations
//===----------------------------------------------------------------------===//

LogicalResult LogicalDivideOp::verify() {
  // Verify both operands are LayoutType
  if (!getInput().getType().isa<LayoutType>()) {
    return emitOpError("input operand must be of LayoutType");
  }
  
  if (!getTiler().getType().isa<LayoutType>()) {
    return emitOpError("tiler operand must be of LayoutType");
  }
  
  // Verify result is LayoutType
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

LogicalResult ZippedDivideOp::verify() {
  // Similar verification
  if (!getInput().getType().isa<LayoutType>()) {
    return emitOpError("input operand must be of LayoutType");
  }
  
  if (!getTiler().getType().isa<LayoutType>()) {
    return emitOpError("tiler operand must be of LayoutType");
  }
  
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

LogicalResult TiledDivideOp::verify() {
  // Similar verification
  if (!getInput().getType().isa<LayoutType>()) {
    return emitOpError("input operand must be of LayoutType");
  }
  
  if (!getTiler().getType().isa<LayoutType>()) {
    return emitOpError("tiler operand must be of LayoutType");
  }
  
  if (!getResult().getType().isa<LayoutType>()) {
    return emitOpError("result must be of LayoutType");
  }
  
  return success();
}

#define GET_OP_CLASSES
#include "rocir/RocirOps.cpp.inc"

