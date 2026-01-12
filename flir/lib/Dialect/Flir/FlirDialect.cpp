//===- FlirDialect.cpp - Flir Dialect Implementation --------------------===//

#include "flir/FlirDialect.h"
#include "flir/FlirOps.h" // Required for generated FlirOps.cpp.inc op class references.
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <functional>
#include <string>

using namespace mlir;
using namespace mlir::flir;

#define GET_ATTRDEF_CLASSES
#include "flir/FlirAttrs.cpp.inc"


//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

IntType IntType::get(MLIRContext *ctx) {
  return Base::get(ctx);
}

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

ShapeType ShapeType::get(MLIRContext *ctx, int rank) {
  // Rank-only: represent as a flat tuple of dynamic leaves.
  Builder b(ctx);
  SmallVector<Attribute, 8> elems;
  elems.reserve(std::max(0, rank));
  for (int32_t i = 0; i < rank; ++i)
    elems.push_back(DyncI64Attr::get(ctx, /*dyncElemIdx=*/i, /*divisibility=*/1));
  if (rank <= 0)
    return get(ctx, Attribute());
  if (rank == 1)
    return get(ctx, elems.front());
  return get(ctx, ArrayAttr::get(ctx, elems));
}

ShapeType ShapeType::get(MLIRContext *ctx, ::mlir::Attribute pattern) {
  return Base::get(ctx, pattern);
}

int ShapeType::getRank() const {
  return getImpl()->rank;
}

::mlir::Attribute ShapeType::getPattern() const {
  return getImpl()->pattern;
}

//===----------------------------------------------------------------------===//
// StrideType
//===----------------------------------------------------------------------===//

StrideType StrideType::get(MLIRContext *ctx, int rank) {
  Builder b(ctx);
  SmallVector<Attribute, 8> elems;
  elems.reserve(std::max(0, rank));
  for (int32_t i = 0; i < rank; ++i)
    elems.push_back(DyncI64Attr::get(ctx, /*dyncElemIdx=*/i, /*divisibility=*/1));
  if (rank <= 0)
    return get(ctx, Attribute());
  if (rank == 1)
    return get(ctx, elems.front());
  return get(ctx, ArrayAttr::get(ctx, elems));
}

int StrideType::getRank() const {
  return getImpl()->rank;
}

StrideType StrideType::get(MLIRContext *ctx, ::mlir::Attribute pattern) {
  return Base::get(ctx, pattern);
}

::mlir::Attribute StrideType::getPattern() const {
  return getImpl()->pattern;
}

//===----------------------------------------------------------------------===//
// LayoutType
//===----------------------------------------------------------------------===//

LayoutType LayoutType::get(MLIRContext *ctx, int rank) {
  // Rank-only: represent both patterns as flat tuple of dynamic leaves.
  Builder b(ctx);
  SmallVector<Attribute, 8> elems;
  elems.reserve(std::max(0, rank));
  for (int32_t i = 0; i < rank; ++i)
    elems.push_back(DyncI64Attr::get(ctx, /*dyncElemIdx=*/i, /*divisibility=*/1));
  if (rank <= 0)
    return get(ctx, Attribute(), Attribute());
  Attribute pat = (rank == 1) ? elems.front() : ArrayAttr::get(ctx, elems);
  return get(ctx, pat, pat);
}

LayoutType LayoutType::get(MLIRContext *ctx, ShapeType shape, StrideType stride) {
  return get(ctx, shape.getPattern(), stride.getPattern());
}

LayoutType LayoutType::get(MLIRContext *ctx, ::mlir::Attribute shapePattern,
                           ::mlir::Attribute stridePattern) {
  return Base::get(ctx, detail::LayoutTypeStorage::KeyTy{/*shapePattern=*/shapePattern,
                                                        /*stridePattern=*/stridePattern});
}

int LayoutType::getRank() const { return getImpl()->rank; }
::mlir::Attribute LayoutType::getShapePattern() const { return getImpl()->shapePattern; }
::mlir::Attribute LayoutType::getStridePattern() const { return getImpl()->stridePattern; }

ShapeType LayoutType::getShapeType() const {
  auto *ctx = getContext();
  return ShapeType::get(ctx, getShapePattern());
}

StrideType LayoutType::getStrideType() const {
  auto *ctx = getContext();
  return StrideType::get(ctx, getStridePattern());
}

//===----------------------------------------------------------------------===//
// CoordType
//===----------------------------------------------------------------------===//

CoordType CoordType::get(MLIRContext *ctx, int rank) {
  Builder b(ctx);
  SmallVector<Attribute, 8> elems;
  elems.reserve(std::max(0, rank));
  for (int32_t i = 0; i < rank; ++i)
    elems.push_back(DyncI64Attr::get(ctx, /*dyncElemIdx=*/i, /*divisibility=*/1));
  if (rank <= 0)
    return get(ctx, Attribute());
  if (rank == 1)
    return get(ctx, elems.front());
  return get(ctx, ArrayAttr::get(ctx, elems));
}

int CoordType::getRank() const {
  return getImpl()->rank;
}

CoordType CoordType::get(MLIRContext *ctx, ::mlir::Attribute pattern) {
  return Base::get(ctx, pattern);
}

::mlir::Attribute CoordType::getPattern() const {
  return getImpl()->pattern;
}

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

#include "flir/FlirDialect.cpp.inc"

void FlirDialect::initialize() {
  addTypes<IntType, ShapeType, StrideType, LayoutType, CoordType>();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "flir/FlirAttrs.cpp.inc"
      >();
  
  addOperations<
#define GET_OP_LIST
#include "flir/FlirOps.cpp.inc"
  >();
}

bool mlir::flir::isValidDyncIntAttr(::mlir::Attribute attr) {
  return llvm::isa<mlir::flir::DyncI32Attr, mlir::flir::DyncI64Attr>(attr);
}

Type FlirDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();

  MLIRContext *ctx = getContext();

  // - tuple: '(' elem (',' elem)* ')'
  // - elem: tuple | '*' | '?' ['{div=INT}'] | INT
  std::function<FailureOr<Attribute>(DialectAsmParser &, int32_t &)> parsePatternElem;
  std::function<FailureOr<Attribute>(DialectAsmParser &, int32_t &)> parsePatternTupleAfterLParen;

  parsePatternTupleAfterLParen = [&](DialectAsmParser &p,
                                     int32_t &dyncIdx) -> FailureOr<Attribute> {
    SmallVector<Attribute, 8> elems;
    if (succeeded(p.parseOptionalRParen()))
      return ArrayAttr::get(ctx, elems);
    while (true) {
      auto e = parsePatternElem(p, dyncIdx);
      if (failed(e))
        return failure();
      elems.push_back(*e);
      if (succeeded(p.parseOptionalComma()))
        continue;
      break;
    }
    if (p.parseRParen())
      return failure();
    return ArrayAttr::get(ctx, elems);
  };

  parsePatternElem = [&](DialectAsmParser &p, int32_t &dyncIdx) -> FailureOr<Attribute> {
    Builder b(ctx);
    if (succeeded(p.parseOptionalLParen()))
      return parsePatternTupleAfterLParen(p, dyncIdx);
    if (succeeded(p.parseOptionalStar()))
      return UnderscoreAttr::get(ctx);
    if (succeeded(p.parseOptionalQuestion())) {
      int32_t divisibility = 1;
      if (succeeded(p.parseOptionalLBrace())) {
        if (p.parseKeyword("div") || p.parseEqual() || p.parseInteger(divisibility) ||
            p.parseRBrace())
          return failure();
      }
      return DyncI64Attr::get(ctx, dyncIdx++, divisibility);
    }
    int64_t v = 0;
    if (p.parseInteger(v))
      return failure();
    return IntegerAttr::get(b.getIntegerType(64), APInt(64, static_cast<uint64_t>(v), true));
  };
  
  if (mnemonic == "int")
    return IntType::get(ctx);

  if (mnemonic == "shape") {
    if (parser.parseLess())
      return Type();
    int32_t dyncIdx = 0;
    auto pat = parsePatternElem(parser, dyncIdx);
    if (failed(pat) || parser.parseGreater())
      return Type();
    return ShapeType::get(ctx, *pat);
  }
  
  if (mnemonic == "stride") {
    if (parser.parseLess())
      return Type();
    int32_t dyncIdx = 0;
    auto pat = parsePatternElem(parser, dyncIdx);
    if (failed(pat) || parser.parseGreater())
      return Type();
    return StrideType::get(ctx, *pat);
  }
  
  if (mnemonic == "layout") {
    if (parser.parseLess())
      return Type();
    // Parse shape pattern.
    int32_t dyncIdx = 0;
    auto shapePat = parsePatternElem(parser, dyncIdx);
    if (failed(shapePat))
      return Type();
    if (parser.parseColon())
      return Type();
    // Parse stride pattern (independent dync numbering).
    dyncIdx = 0;
    auto stridePat = parsePatternElem(parser, dyncIdx);
    if (failed(stridePat) || parser.parseGreater())
      return Type();
    return LayoutType::get(ctx, *shapePat, *stridePat);
  }
  
  if (mnemonic == "coord") {
    if (parser.parseLess())
      return Type();
    int32_t dyncIdx = 0;
    auto pat = parsePatternElem(parser, dyncIdx);
    if (failed(pat) || parser.parseGreater())
      return Type();
    return CoordType::get(ctx, *pat);
  }
  
  parser.emitError(parser.getNameLoc(), "unknown flir type: ") << mnemonic;
  return Type();
}

void FlirDialect::printType(Type type, DialectAsmPrinter &os) const {
  auto printPattern = [&](Attribute a) {
    std::function<void(Attribute)> rec = [&](Attribute x) {
      if (!x) {
        os << "?";
        return;
      }
      if (auto arr = dyn_cast<ArrayAttr>(x)) {
        os << "(";
        bool first = true;
        for (auto e : arr.getValue()) {
          if (!first)
            os << ",";
          rec(e);
          first = false;
        }
        os << ")";
        return;
      }
      if (isa<UnderscoreAttr>(x)) {
        os << "*";
        return;
      }
      if (auto d64 = dyn_cast<DyncI64Attr>(x)) {
        if (d64.getDivisibility() == 1)
          os << "?";
        else
          os << "?{div=" << d64.getDivisibility() << "}";
        return;
      }
      if (auto d32 = dyn_cast<DyncI32Attr>(x)) {
        if (d32.getDivisibility() == 1)
          os << "?";
        else
          os << "?{div=" << d32.getDivisibility() << "}";
        return;
      }
      if (auto intAttr = dyn_cast<IntegerAttr>(x)) {
        os << intAttr.getInt();
        return;
      }
      // Fallback: print as '?' to avoid breaking assembly.
      os << "?";
    };
    rec(a);
  };

  if (auto intType = llvm::dyn_cast<IntType>(type)) {
    os << "int";
  } else if (auto shapeType = llvm::dyn_cast<ShapeType>(type)) {
    int r = shapeType.getRank();
    if (r < 0) {
      os << "shape<" << r << ">";
      return;
    }
    os << "shape<";
    printPattern(shapeType.getPattern());
    os << ">";
  } else if (auto strideType = llvm::dyn_cast<StrideType>(type)) {
    int r = strideType.getRank();
    if (r < 0) {
      os << "stride<" << r << ">";
      return;
    }
    os << "stride<";
    printPattern(strideType.getPattern());
    os << ">";
  } else if (auto layoutType = llvm::dyn_cast<LayoutType>(type)) {
    int r = layoutType.getRank();
    if (r < 0) {
      os << "layout<" << r << ">";
      return;
    }
    os << "layout<";
    printPattern(layoutType.getShapePattern());
    os << ":";
    printPattern(layoutType.getStridePattern());
    os << ">";
  } else if (auto coordType = llvm::dyn_cast<CoordType>(type)) {
    int r = coordType.getRank();
    if (r < 0) {
      os << "coord<" << r << ">";
      return;
    }
    os << "coord<";
    printPattern(coordType.getPattern());
    os << ">";
  }
}
