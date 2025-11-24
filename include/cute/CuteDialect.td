//===- CuteDialect.td - CuTe Dialect Definition ------------*- tablegen -*-===//
//
// Part of the CUTLASS Project, under the NVIDIA License.
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_DIALECT
#define CUTE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// CuTe Dialect Definition
//===----------------------------------------------------------------------===//

def Cute_Dialect : Dialect {
  let name = "cute";
  let summary = "CuTe layout-based tensor programming dialect";
  let description = [{
    The CuTe dialect provides high-level layout abstractions for GPU tensor
    operations. It models shapes, strides, coordinates, and layouts as first-class
    values, enabling layout-aware transformations and optimizations.

    Key concepts:
    - Shape: Multi-dimensional size specification
    - Stride: Multi-dimensional stride specification  
    - Layout: Mapping from coordinates to linear indices
    - Tensor: Data view with shape, stride, and memory reference
    - Tile: Hierarchical partitioning specification
  }];
  
  let cppNamespace = "::mlir::cute";
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  // let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// CuTe Type Definitions
//===----------------------------------------------------------------------===//

class Cute_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Cute_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// IntType - Represents a compile-time integer constant
def Cute_IntType : Cute_Type<"Int", "int"> {
  let summary = "Compile-time integer constant";
  let description = [{
    Represents a static integer value known at compile time.
    Used for shape/stride components and other compile-time constants.
  }];
  let parameters = (ins "int64_t":$value);
  let assemblyFormat = "`<` $value `>`";
}

// ShapeType - Multi-dimensional shape specification
def Cute_ShapeType : Cute_Type<"Shape", "shape"> {
  let summary = "Multi-dimensional shape";
  let description = [{
    Represents a hierarchical multi-dimensional shape. Can be:
    - A single integer (rank-1)
    - A tuple of shapes (rank-N)
    
    Examples:
    - !cute.shape<128>              // 1D shape
    - !cute.shape<(64, 64)>         // 2D shape
    - !cute.shape<((16, 4), 64)>    // Hierarchical shape
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$extents,
                        "bool":$isHierarchical);
  let assemblyFormat = "`<` $extents `>`";
}

// StrideType - Multi-dimensional stride specification
def Cute_StrideType : Cute_Type<"Stride", "stride"> {
  let summary = "Multi-dimensional stride";
  let description = [{
    Represents a hierarchical multi-dimensional stride pattern.
    Strides define the memory offset between adjacent elements.
  }];
  let parameters = (ins ArrayRefParameter<"int64_t">:$strides,
                        "bool":$isHierarchical);
  let assemblyFormat = "`<` $strides `>`";
}

// LayoutType - Shape + Stride = Coordinate to Index mapping
def Cute_LayoutType : Cute_Type<"Layout", "layout"> {
  let summary = "Layout: (Shape, Stride) mapping";
  let description = [{
    A Layout combines a Shape and Stride to define a mapping from
    multi-dimensional coordinates to linear indices.
    
    layout(coord) = sum(coord[i] * stride[i])
  }];
  let parameters = (ins "ShapeType":$shape, "StrideType":$stride);
  let assemblyFormat = "`<` $shape `,` $stride `>`";
}

// TileType - Hierarchical partitioning specification
def Cute_TileType : Cute_Type<"Tile", "tile"> {
  let summary = "Tile specification for hierarchical partitioning";
  let description = [{
    Represents a tiling pattern for partitioning tensors.
    Used in tiled operations like TiledMma and TiledCopy.
  }];
  let parameters = (ins "ShapeType":$tileShape);
  let assemblyFormat = "`<` $tileShape `>`";
}

// CoordType - Multi-dimensional coordinate
def Cute_CoordType : Cute_Type<"Coord", "coord"> {
  let summary = "Multi-dimensional coordinate";
  let description = [{
    Represents a coordinate in multi-dimensional space.
    Used for indexing tensors and layouts.
  }];
  let parameters = (ins "int64_t":$rank);
  let assemblyFormat = "`<` $rank `>`";
}

// MemRefType - CuTe-specific memory reference
def Cute_MemRefType : Cute_Type<"MemRef", "memref"> {
  let summary = "Memory reference with element type";
  let description = [{
    Represents a typed pointer to memory with layout information.
  }];
  let parameters = (ins "Type":$elementType, "LayoutType":$layout,
                        "unsigned":$memorySpace);
  let assemblyFormat = "`<` $elementType `,` $layout `,` $memorySpace `>`";
}

// TensorType - Data view with shape, stride, and data pointer
def Cute_TensorType : Cute_Type<"Tensor", "tensor"> {
  let summary = "Tensor: Iterator + Layout view over data";
  let description = [{
    A Tensor is a view over data combining:
    - An iterator (pointer/memref)
    - A layout (shape + stride)
    
    Enables layout-aware tensor operations.
  }];
  let parameters = (ins "Type":$elementType, "LayoutType":$layout);
  let assemblyFormat = "`<` $elementType `,` $layout `>`";
}

//===----------------------------------------------------------------------===//
// CuTe Attribute Definitions
//===----------------------------------------------------------------------===//

class Cute_Attr<string name, string attrMnemonic>
    : AttrDef<Cute_Dialect, name> {
  let mnemonic = attrMnemonic;
}

// StaticShapeAttr - Compile-time shape attribute
def Cute_StaticShapeAttr : Cute_Attr<"StaticShape", "static_shape"> {
  let summary = "Static shape attribute";
  let parameters = (ins ArrayRefParameter<"int64_t">:$extents);
  let assemblyFormat = "`<` $extents `>`";
}

// StaticStrideAttr - Compile-time stride attribute
def Cute_StaticStrideAttr : Cute_Attr<"StaticStride", "static_stride"> {
  let summary = "Static stride attribute";
  let parameters = (ins ArrayRefParameter<"int64_t">:$strides);
  let assemblyFormat = "`<` $strides `>`";
}

//===----------------------------------------------------------------------===//
// CuTe Operation Base
//===----------------------------------------------------------------------===//

class Cute_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cute_Dialect, mnemonic, traits> {
  let extraClassDeclaration = [{
    template <typename OpT>
    using InferredProperties = ::mlir::EmptyProperties;
    ::mlir::EmptyProperties getProperties() { return {}; }
    static bool classof(::mlir::Operation *op) { return Op::classof(op); }
  }];
}

#endif // CUTE_DIALECT
