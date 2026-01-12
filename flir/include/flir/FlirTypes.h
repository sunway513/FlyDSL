#ifndef FLIR_TYPES_H
#define FLIR_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Attributes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

namespace mlir::flir {

namespace detail {

// Storage for IntType - a simple type without parameters
struct IntTypeStorage : public TypeStorage {
  using KeyTy = int; // Dummy key
  
  IntTypeStorage() = default;
  
  bool operator==(const KeyTy &) const { return true; }
  
  static IntTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &) {
    return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage();
  }
};

// Storage for ranked types (Shape, Stride, Layout, Coord)
struct RankedTypeStorage : public TypeStorage {
  using KeyTy = int; // rank
  
  RankedTypeStorage(int rank) : rank(rank) {}
  
  bool operator==(const KeyTy &key) const { return rank == key; }
  
  static RankedTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RankedTypeStorage>()) RankedTypeStorage(key);
  }
  
  int rank;
};

// Storage for structured "int-tuple patterns" (Shape, Stride, Coord):
// the canonical representation is an Attribute tree:
// - leaf: IntegerAttr / DyncI32 / DyncI64 / Underscore
// - tuple: ArrayAttr
struct PatternTypeStorage : public TypeStorage {
  using KeyTy = ::mlir::Attribute; // pattern

  PatternTypeStorage(int rank, ::mlir::Attribute pattern) : rank(rank), pattern(pattern) {}

  bool operator==(const KeyTy &key) const { return pattern == key; }

  static PatternTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    ::mlir::Attribute pattern = key;

    // Derive rank from leaf count.
    int rank = 0;
    std::function<void(::mlir::Attribute)> count = [&](::mlir::Attribute a) {
      if (auto arr = llvm::dyn_cast<::mlir::ArrayAttr>(a)) {
        for (auto e : arr.getValue())
          count(e);
        return;
      }
      ++rank;
    };
    if (pattern)
      count(pattern);
    else
      rank = -1;

    return new (allocator.allocate<PatternTypeStorage>()) PatternTypeStorage(rank, pattern);
  }

  int rank;
  ::mlir::Attribute pattern;
};

// Storage for LayoutType:
// - rank: flattened rank (leaf count)
// - shape/stride patterns: Attribute trees
struct LayoutTypeStorage : public TypeStorage {
  struct KeyTy {
    ::mlir::Attribute shapePattern;
    ::mlir::Attribute stridePattern;
  };

  LayoutTypeStorage(int rank, ::mlir::Attribute shapePattern, ::mlir::Attribute stridePattern)
      : rank(rank), shapePattern(shapePattern), stridePattern(stridePattern) {}

  bool operator==(const KeyTy &key) const {
    return shapePattern == key.shapePattern && stridePattern == key.stridePattern;
  }

  static LayoutTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    ::mlir::Attribute shapePattern = key.shapePattern;
    ::mlir::Attribute stridePattern = key.stridePattern;

    int rank = 0;
    std::function<void(::mlir::Attribute)> count = [&](::mlir::Attribute a) {
      if (auto arr = llvm::dyn_cast<::mlir::ArrayAttr>(a)) {
        for (auto e : arr.getValue())
          count(e);
        return;
      }
      ++rank;
    };
    if (shapePattern)
      count(shapePattern);
    else if (stridePattern)
      count(stridePattern);
    else
      rank = -1;

    return new (allocator.allocate<LayoutTypeStorage>()) LayoutTypeStorage(rank, shapePattern, stridePattern);
  }

  int rank;
  ::mlir::Attribute shapePattern;
  ::mlir::Attribute stridePattern;
};

} // namespace detail

} // namespace mlir::flir

namespace llvm {
template <>
struct DenseMapInfo<mlir::flir::detail::LayoutTypeStorage::KeyTy> {
  using KeyTy = mlir::flir::detail::LayoutTypeStorage::KeyTy;
  static inline KeyTy getEmptyKey() {
    return KeyTy{/*shapePattern=*/DenseMapInfo<::mlir::Attribute>::getEmptyKey(),
                 /*stridePattern=*/DenseMapInfo<::mlir::Attribute>::getEmptyKey()};
  }
  static inline KeyTy getTombstoneKey() {
    return KeyTy{/*shapePattern=*/DenseMapInfo<::mlir::Attribute>::getTombstoneKey(),
                 /*stridePattern=*/DenseMapInfo<::mlir::Attribute>::getTombstoneKey()};
  }
  static unsigned getHashValue(const KeyTy &k) {
    return static_cast<unsigned>(
        llvm::hash_combine(DenseMapInfo<::mlir::Attribute>::getHashValue(k.shapePattern),
                           DenseMapInfo<::mlir::Attribute>::getHashValue(k.stridePattern)));
  }
  static bool isEqual(const KeyTy &a, const KeyTy &b) {
    return a.shapePattern == b.shapePattern && a.stridePattern == b.stridePattern;
  }
};
} // namespace llvm

#endif // FLIR_TYPES_H

