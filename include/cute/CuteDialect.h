#ifndef CUTE_DIALECT_H
#define CUTE_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::cute {

// Forward declarations for type storage
namespace detail {
struct IntTypeStorage;
struct RankedTypeStorage;
}

// Type declarations
class IntType : public Type::TypeBase<IntType, Type, detail::IntTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.int";
  static IntType get(MLIRContext *context);
};

class ShapeType : public Type::TypeBase<ShapeType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.shape";
  static ShapeType get(MLIRContext *context, int rank);
  int getRank() const;
};

class StrideType : public Type::TypeBase<StrideType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.stride";
  static StrideType get(MLIRContext *context, int rank);
  int getRank() const;
};

class LayoutType : public Type::TypeBase<LayoutType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.layout";
  static LayoutType get(MLIRContext *context, int rank);
  int getRank() const;
};

class CoordType : public Type::TypeBase<CoordType, Type, detail::RankedTypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "cute.coord";
  static CoordType get(MLIRContext *context, int rank);
  int getRank() const;
};

} // namespace mlir::cute

// Include generated dialect declarations
#include "cute/CuteDialect.h.inc"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "cute/CuteOps.h.inc"

#endif // CUTE_DIALECT_H
