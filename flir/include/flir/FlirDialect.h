#ifndef FLIR_DIALECT_H
#define FLIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "flir/FlirTypes.h"

namespace mlir::flir {

// AttrDef helper predicates (used by TableGen AnyAttrOf).
bool isValidDyncIntAttr(::mlir::Attribute attr);

// Type declarations
class IntType : public Type::TypeBase<IntType, Type, detail::IntTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "flir.int";
  static IntType get(MLIRContext *context);
};

class ShapeType : public Type::TypeBase<ShapeType, Type, detail::PatternTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "flir.shape";
  static ShapeType get(MLIRContext *context, int rank);
  static ShapeType get(MLIRContext *context, ::mlir::Attribute pattern);
  int getRank() const;
  ::mlir::Attribute getPattern() const;
};

class StrideType : public Type::TypeBase<StrideType, Type, detail::PatternTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "flir.stride";
  static StrideType get(MLIRContext *context, int rank);
  static StrideType get(MLIRContext *context, ::mlir::Attribute pattern);
  int getRank() const;
  ::mlir::Attribute getPattern() const;
};

class LayoutType : public Type::TypeBase<LayoutType, Type, detail::LayoutTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "flir.layout";
  /// Rank-only layout (backward compatible): layout<rank> / layout<(?,?,?)>
  static LayoutType get(MLIRContext *context, int rank);
  /// Layout with structured shape/stride type information.
  static LayoutType get(MLIRContext *context, ShapeType shape, StrideType stride);
  /// Layout with attribute patterns.
  static LayoutType get(MLIRContext *context, ::mlir::Attribute shapePattern,
                        ::mlir::Attribute stridePattern);
  int getRank() const;
  ::mlir::Attribute getShapePattern() const;
  ::mlir::Attribute getStridePattern() const;
  ShapeType getShapeType() const;
  StrideType getStrideType() const;
};

class CoordType : public Type::TypeBase<CoordType, Type, detail::PatternTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "flir.coord";
  static CoordType get(MLIRContext *context, int rank);
  static CoordType get(MLIRContext *context, ::mlir::Attribute pattern);
  int getRank() const;
  ::mlir::Attribute getPattern() const;
};

} // namespace mlir::flir

// Include generated dialect declarations
#include "flir/FlirDialect.h.inc"

// Include generated attribute declarations
#define GET_ATTRDEF_CLASSES
#include "flir/FlirAttrs.h.inc"

// Include generated operation declarations
#endif // FLIR_DIALECT_H
