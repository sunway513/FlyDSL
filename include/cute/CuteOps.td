//===- CuteOps.td - Cute operation definitions ------------*- tablegen -*-===//

#ifndef CUTE_OPS
#define CUTE_OPS

include "CuteDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Cute_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cute_Dialect, mnemonic, traits>;

// Type constraints
def Cute_IntType : Type<CPred<"llvm::isa<::mlir::cute::IntType>($_self)">, "cute.int">;
def Cute_ShapeType : Type<CPred<"llvm::isa<::mlir::cute::ShapeType>($_self)">, "cute.shape">;
def Cute_StrideType : Type<CPred<"llvm::isa<::mlir::cute::StrideType>($_self)">, "cute.stride">;
def Cute_LayoutType : Type<CPred<"llvm::isa<::mlir::cute::LayoutType>($_self)">, "cute.layout">;
def Cute_CoordType : Type<CPred<"llvm::isa<::mlir::cute::CoordType>($_self)">, "cute.coord">;

//===----------------------------------------------------------------------===//
// Construction Operations
//===----------------------------------------------------------------------===//

def Cute_MakeShapeOp : Cute_Op<"make_shape", []> {
  let summary = "Create a CUTE shape from index values";
  let description = [{
    Creates a shape value from runtime index values.
    Example: cute.make_shape %i, %j : (index, index) -> !cute.shape<2>
  }];
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Cute_ShapeType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Cute_MakeStrideOp : Cute_Op<"make_stride", []> {
  let summary = "Create a CUTE stride from index values";
  let description = [{
    Creates a stride value from runtime index values.
    Example: cute.make_stride %i, %j : (index, index) -> !cute.stride<2>
  }];
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Cute_StrideType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

def Cute_MakeLayoutOp : Cute_Op<"make_layout", []> {
  let summary = "Create a CUTE layout from shape and stride";
  let description = [{
    Creates a layout value by combining a shape and stride.
    Example: cute.make_layout %shape, %stride : (!cute.shape<2>, !cute.stride<2>) -> !cute.layout<2>
  }];
  let arguments = (ins Cute_ShapeType:$shape, Cute_StrideType:$stride);
  let results = (outs Cute_LayoutType:$result);
  let assemblyFormat = "$shape `,` $stride attr-dict `:` functional-type(operands, $result)";
}

def Cute_MakeCoordOp : Cute_Op<"make_coord", []> {
  let summary = "Create a CUTE coordinate from index values";
  let description = [{
    Creates a coordinate value from runtime index values.
    Example: cute.make_coord %i, %j : (index, index) -> !cute.coord<2>
  }];
  let arguments = (ins Variadic<Index>:$values);
  let results = (outs Cute_CoordType:$result);
  let assemblyFormat = "$values attr-dict `:` functional-type($values, $result)";
}

//===----------------------------------------------------------------------===//
// Query Operations
//===----------------------------------------------------------------------===//

def Cute_SizeOp : Cute_Op<"size", []> {
  let summary = "Get the total size of a shape or layout";
  let description = [{
    Computes the product of all dimensions in a shape or the size of a layout.
    For shape (M, N), returns M * N.
    Example: %s = cute.size %shape : !cute.shape<2> -> index
  }];
  let arguments = (ins AnyTypeOf<[Cute_ShapeType, Cute_LayoutType]>:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Cute_CosizeOp : Cute_Op<"cosize", []> {
  let summary = "Get the cosize of a layout";
  let description = [{
    Computes the cosize of a layout (codomain size).
    For a layout with shape and stride, returns the span of the mapping.
    Example: %cs = cute.cosize %layout : !cute.layout<2> -> index
  }];
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Cute_GetOp : Cute_Op<"get", []> {
  let summary = "Extract element from shape/stride/coord at given index";
  let description = [{
    Extracts the value at a specific position from shape, stride, or coord.
    Example: %val = cute.get %shape, %idx : !cute.shape<3>, index -> index
  }];
  let arguments = (ins 
    AnyTypeOf<[Cute_ShapeType, Cute_StrideType, Cute_CoordType]>:$input,
    Index:$idx
  );
  let results = (outs Index:$result);
  let assemblyFormat = "$input `,` $idx attr-dict `:` type($input) `,` type($idx) `->` type($result)";
}

def Cute_RankOp : Cute_Op<"rank", []> {
  let summary = "Get the rank (number of dimensions) of a type";
  let description = [{
    Returns the number of dimensions in a shape, stride, coord, or layout.
    Example: %r = cute.rank %shape : !cute.shape<3> -> index
  }];
  let arguments = (ins AnyTypeOf<[Cute_ShapeType, Cute_StrideType, Cute_CoordType, Cute_LayoutType]>:$input);
  let results = (outs Index:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Coordinate Mapping Operations
//===----------------------------------------------------------------------===//

def Cute_Crd2IdxOp : Cute_Op<"crd2idx", []> {
  let summary = "Convert coordinate to linear index using layout";
  let description = [{
    Converts a multi-dimensional coordinate to a linear index using a layout.
    Computes: sum(coord[i] * stride[i]) for all dimensions i.
    Example: %idx = cute.crd2idx %coord, %layout : (!cute.coord<2>, !cute.layout<2>) -> index
  }];
  let arguments = (ins Cute_CoordType:$coord, Cute_LayoutType:$layout);
  let results = (outs Index:$result);
  let assemblyFormat = "$coord `,` $layout attr-dict `:` functional-type(operands, $result)";
}

def Cute_Idx2CrdOp : Cute_Op<"idx2crd", []> {
  let summary = "Convert linear index to coordinate using layout";
  let description = [{
    Converts a linear index to a multi-dimensional coordinate using a layout.
    This is the inverse of crd2idx.
    Example: %coord = cute.idx2crd %idx, %layout : (index, !cute.layout<2>) -> !cute.coord<2>
  }];
  let arguments = (ins Index:$idx, Cute_LayoutType:$layout);
  let results = (outs Cute_CoordType:$result);
  let assemblyFormat = "$idx `,` $layout attr-dict `:` functional-type(operands, $result)";
}

//===----------------------------------------------------------------------===//
// Layout Manipulation Operations
//===----------------------------------------------------------------------===//

def Cute_GetShapeOp : Cute_Op<"get_shape", []> {
  let summary = "Extract shape from layout";
  let description = [{
    Extracts the shape component from a layout.
    Example: %shape = cute.get_shape %layout : !cute.layout<2> -> !cute.shape<2>
  }];
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_ShapeType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Cute_GetStrideOp : Cute_Op<"get_stride", []> {
  let summary = "Extract stride from layout";
  let description = [{
    Extracts the stride component from a layout.
    Example: %stride = cute.get_stride %layout : !cute.layout<2> -> !cute.stride<2>
  }];
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_StrideType:$result);
  let assemblyFormat = "$layout attr-dict `:` type($layout) `->` type($result)";
}

def Cute_CompositionOp : Cute_Op<"composition", []> {
  let summary = "Compose two layouts";
  let description = [{
    Computes the composition of two layouts: layoutA ∘ layoutB.
    The result maps indices through layoutB then through layoutA.
    Example: %comp = cute.composition %layoutA, %layoutB : (!cute.layout<2>, !cute.layout<2>) -> !cute.layout<2>
  }];
  let arguments = (ins Cute_LayoutType:$layoutA, Cute_LayoutType:$layoutB);
  let results = (outs Cute_LayoutType:$result);
  let assemblyFormat = "$layoutA `,` $layoutB attr-dict `:` functional-type(operands, $result)";
}

#endif // CUTE_OPS
