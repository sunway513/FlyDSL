//===- CuteOps.td - CuTe IR Operation Definitions -----------*- tablegen -*-===//
//
// CuTe Layout Algebra Operations
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_OPS
#define CUTE_OPS

include "cute/CuteDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Construction Operations (5 ops)
//===----------------------------------------------------------------------===//

def Cute_MakeShapeOp : Cute_Op<"make_shape", [Pure]> {
  let summary = "Construct multi-dimensional shape";
  let description = [{
    Creates a ShapeType from compile-time integer values.
    
    Example:
    ```mlir
    %shape = cute.make_shape %c64, %c128 : !cute.shape<(64, 128)>
    ```
  }];
  
  let arguments = (ins Variadic<Cute_IntType>:$dims);
  let results = (outs Cute_ShapeType:$result);
  let assemblyFormat = "$dims attr-dict `:` type($dims) `->` type($result)";
}

def Cute_MakeStrideOp : Cute_Op<"make_stride", [Pure]> {
  let summary = "Construct multi-dimensional stride";
  let arguments = (ins Variadic<Cute_IntType>:$strides);
  let results = (outs Cute_StrideType:$result);
  let assemblyFormat = "$strides attr-dict `:` type($strides) `->` type($result)";
}

def Cute_MakeLayoutOp : Cute_Op<"make_layout", [Pure]> {
  let summary = "Construct Layout from shape and stride";
  let arguments = (ins Cute_ShapeType:$shape, Cute_StrideType:$stride);
  let results = (outs Cute_LayoutType:$result);
  let assemblyFormat = "$shape `,` $stride attr-dict `:` type($shape) `,` type($stride) `->` type($result)";
}

def Cute_MakeCoordOp : Cute_Op<"make_coord", [Pure]> {
  let summary = "Construct multi-dimensional coordinate";
  let arguments = (ins Variadic<Index>:$indices);
  let results = (outs Cute_CoordType:$result);
  let assemblyFormat = "$indices attr-dict `:` type($indices) `->` type($result)";
}

def Cute_MakeTileOp : Cute_Op<"make_tile", [Pure]> {
  let summary = "Construct hierarchical tile";
  let arguments = (ins Variadic<AnyTypeOf<[Cute_LayoutType, Cute_TileType]>>:$levels);
  let results = (outs Cute_TileType:$result);
  let assemblyFormat = "$levels attr-dict `:` type($levels) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Query Operations (7 ops)
//===----------------------------------------------------------------------===//

def Cute_GetShapeOp : Cute_Op<"get_shape", [Pure]> {
  let summary = "Extract shape from layout";
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_ShapeType:$result);
}

def Cute_GetStrideOp : Cute_Op<"get_stride", [Pure]> {
  let summary = "Extract stride from layout";
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_StrideType:$result);
}

def Cute_RankOp : Cute_Op<"rank", [Pure]> {
  let summary = "Get rank (number of dimensions)";
  let arguments = (ins AnyTypeOf<[Cute_ShapeType, Cute_LayoutType]>:$input);
  let results = (outs Index:$result);
}

def Cute_SizeOp : Cute_Op<"size", [Pure]> {
  let summary = "Get total element count";
  let arguments = (ins AnyTypeOf<[Cute_ShapeType, Cute_LayoutType]>:$input);
  let results = (outs Index:$result);
}

def Cute_DepthOp : Cute_Op<"depth", [Pure]> {
  let summary = "Get depth of tile hierarchy";
  let arguments = (ins Cute_TileType:$tile);
  let results = (outs Index:$result);
}

def Cute_ShapeAtOp : Cute_Op<"shape_at", [Pure]> {
  let summary = "Get shape at dimension I";
  let arguments = (ins Cute_ShapeType:$shape, IndexAttr:$dim);
  let results = (outs Cute_IntType:$result);
}

def Cute_StrideAtOp : Cute_Op<"stride_at", [Pure]> {
  let summary = "Get stride at dimension I";
  let arguments = (ins Cute_StrideType:$stride, IndexAttr:$dim);
  let results = (outs Cute_IntType:$result);
}

//===----------------------------------------------------------------------===//
// Mapping Operations (2 ops)
//===----------------------------------------------------------------------===//

def Cute_Crd2IdxOp : Cute_Op<"crd2idx", [Pure]> {
  let summary = "Map coordinate to linear index";
  let description = [{
    Maps multi-dimensional coordinate to 1D offset using layout's stride.
    Returns: sum(coord[i] * stride[i])
  }];
  let arguments = (ins Cute_CoordType:$coord, Cute_LayoutType:$layout);
  let results = (outs Index:$result);
}

def Cute_Idx2CrdOp : Cute_Op<"idx2crd", [Pure]> {
  let summary = "Map linear index to coordinate";
  let arguments = (ins Index:$index, Cute_LayoutType:$layout);
  let results = (outs Cute_CoordType:$result);
}

//===----------------------------------------------------------------------===//
// Transform Operations (6 ops)
//===----------------------------------------------------------------------===//

def Cute_CompositionOp : Cute_Op<"composition", [Pure]> {
  let summary = "Compose two layouts: LayoutA(LayoutB(x))";
  let arguments = (ins Cute_LayoutType:$layoutA, Cute_LayoutType:$layoutB);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_ComplementOp : Cute_Op<"complement", [Pure]> {
  let summary = "Generate complement layout";
  let description = [{
    Creates layout covering unmapped indices in [0, cosize(layout)).
  }];
  let arguments = (ins Cute_LayoutType:$layout, OptionalAttr<I64Attr>:$cosize);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_LeftInverseOp : Cute_Op<"left_inverse", [Pure]> {
  let summary = "Compute left inverse layout";
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_RightInverseOp : Cute_Op<"right_inverse", [Pure]> {
  let summary = "Compute right inverse layout";
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_CoalesceOp : Cute_Op<"coalesce", [Pure]> {
  let summary = "Merge adjacent dimensions with compatible strides";
  let arguments = (ins Cute_LayoutType:$layout);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_FilterOp : Cute_Op<"filter", [Pure]> {
  let summary = "Filter layout to only keep dimension N";
  let arguments = (ins Cute_LayoutType:$layout, IndexAttr:$dim);
  let results = (outs Cute_LayoutType:$result);
}

//===----------------------------------------------------------------------===//
// Product Operations (4 ops)
//===----------------------------------------------------------------------===//

def Cute_LogicalProductOp : Cute_Op<"logical_product", [Pure, Commutative]> {
  let summary = "Cartesian product of layouts";
  let arguments = (ins Cute_LayoutType:$lhs, Cute_LayoutType:$rhs);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_BlockedProductOp : Cute_Op<"blocked_product", [Pure]> {
  let summary = "Blocked product for memory coalescing";
  let arguments = (ins Cute_LayoutType:$block, Cute_LayoutType:$repetition);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_RakedProductOp : Cute_Op<"raked_product", [Pure]> {
  let summary = "Raked product for strided access";
  let arguments = (ins Cute_LayoutType:$lhs, Cute_LayoutType:$rhs);
  let results = (outs Cute_LayoutType:$result);
}

def Cute_TiledProductOp : Cute_Op<"tiled_product", [Pure]> {
  let summary = "Hierarchical tiled product";
  let arguments = (ins Cute_LayoutType:$lhs, Cute_TileType:$tile);
  let results = (outs Cute_LayoutType:$result);
}

//===----------------------------------------------------------------------===//
// Partition Operations (5 ops)
//===----------------------------------------------------------------------===//

def Cute_LocalPartitionOp : Cute_Op<"local_partition", [Pure]> {
  let summary = "Partition tensor across threads/blocks";
  let arguments = (ins 
    Cute_TensorType:$tensor,
    Cute_LayoutType:$tiler,
    IndexAttr:$thread_idx
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_LocalTileOp : Cute_Op<"local_tile", [Pure]> {
  let summary = "Extract local tile";
  let arguments = (ins 
    Cute_TensorType:$tensor,
    Cute_TileType:$tile,
    Cute_CoordType:$coord
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_SliceOp : Cute_Op<"slice", [Pure]> {
  let summary = "Slice layout along dimension";
  let arguments = (ins 
    Cute_LayoutType:$layout,
    IndexAttr:$dim,
    Index:$offset
  );
  let results = (outs Cute_LayoutType:$result);
}

def Cute_ZippedDivideOp : Cute_Op<"zipped_divide", [Pure]> {
  let summary = "Divide layout into two interleaved parts";
  let arguments = (ins 
    Cute_LayoutType:$layout,
    Cute_TileType:$tile
  );
  let results = (outs 
    Cute_LayoutType:$major,
    Cute_LayoutType:$minor
  );
}

def Cute_DiceOp : Cute_Op<"dice", [Pure]> {
  let summary = "Dice layout into sub-layouts";
  let arguments = (ins 
    Cute_LayoutType:$layout,
    Cute_ShapeType:$tile_shape
  );
  let results = (outs Cute_LayoutType:$result);
}

//===----------------------------------------------------------------------===//
// Tensor Operations (10 ops)
//===----------------------------------------------------------------------===//

def Cute_MakeTensorOp : Cute_Op<"make_tensor"> {
  let summary = "Create tensor view";
  let arguments = (ins 
    AnyType:$ptr,
    Cute_LayoutType:$layout
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_TensorLoadOp : Cute_Op<"tensor_load"> {
  let summary = "Load element from tensor";
  let arguments = (ins 
    Cute_TensorType:$tensor,
    Cute_CoordType:$coord
  );
  let results = (outs AnyType:$result);
}

def Cute_TensorStoreOp : Cute_Op<"tensor_store"> {
  let summary = "Store element to tensor";
  let arguments = (ins 
    AnyType:$value,
    Cute_TensorType:$tensor,
    Cute_CoordType:$coord
  );
}

def Cute_CopyOp : Cute_Op<"copy"> {
  let summary = "Copy data between tensors";
  let arguments = (ins 
    Cute_TensorType:$src,
    Cute_TensorType:$dst
  );
}

def Cute_ClearOp : Cute_Op<"clear"> {
  let summary = "Clear tensor to zero";
  let arguments = (ins Cute_TensorType:$tensor);
}

def Cute_FillOp : Cute_Op<"fill"> {
  let summary = "Fill tensor with value";
  let arguments = (ins 
    Cute_TensorType:$tensor,
    AnyType:$value
  );
}

def Cute_AxpbyOp : Cute_Op<"axpby"> {
  let summary = "Compute alpha * x + beta * y";
  let arguments = (ins 
    AnyType:$alpha,
    Cute_TensorType:$x,
    AnyType:$beta,
    Cute_TensorType:$y
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_GemvOp : Cute_Op<"gemv"> {
  let summary = "General matrix-vector multiply";
  let arguments = (ins 
    Cute_TensorType:$A,
    Cute_TensorType:$x,
    Cute_TensorType:$y
  );
}

def Cute_GemmOp : Cute_Op<"gemm"> {
  let summary = "General matrix multiply";
  let arguments = (ins 
    Cute_TensorType:$A,
    Cute_TensorType:$B,
    Cute_TensorType:$C
  );
}

def Cute_PrintOp : Cute_Op<"print"> {
  let summary = "Print tensor for debugging";
  let arguments = (ins Cute_TensorType:$tensor);
}

//===----------------------------------------------------------------------===//
// MMA Operations (5 ops)
//===----------------------------------------------------------------------===//

def Cute_MakeTiledMmaOp : Cute_Op<"make_tiled_mma"> {
  let summary = "Construct TiledMma descriptor";
  let arguments = (ins 
    AnyType:$mma_atom,
    Cute_LayoutType:$layout_v,
    Cute_TileType:$tiler_mnk
  );
  let results = (outs AnyType:$result);
}

def Cute_MmaMakeFragmentAOp : Cute_Op<"mma_make_fragment_a"> {
  let summary = "Partition A matrix for MMA";
  let arguments = (ins 
    AnyType:$tiled_mma,
    Cute_TensorType:$tensor
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_MmaMakeFragmentBOp : Cute_Op<"mma_make_fragment_b"> {
  let summary = "Partition B matrix for MMA";
  let arguments = (ins 
    AnyType:$tiled_mma,
    Cute_TensorType:$tensor
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_MmaMakeFragmentCOp : Cute_Op<"mma_make_fragment_c"> {
  let summary = "Partition C matrix for MMA";
  let arguments = (ins 
    AnyType:$tiled_mma,
    Cute_TensorType:$tensor
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_TiledMmaOp : Cute_Op<"tiled_mma"> {
  let summary = "Execute tiled MMA operation";
  let arguments = (ins 
    AnyType:$tiled_mma,
    Cute_TensorType:$fragA,
    Cute_TensorType:$fragB,
    Cute_TensorType:$fragC
  );
  let results = (outs Cute_TensorType:$result);
}

//===----------------------------------------------------------------------===//
// Copy Operations (5 ops)
//===----------------------------------------------------------------------===//

def Cute_MakeTiledCopyOp : Cute_Op<"make_tiled_copy"> {
  let summary = "Construct TiledCopy descriptor";
  let arguments = (ins 
    AnyType:$copy_atom,
    Cute_LayoutType:$layout_v,
    Cute_TileType:$tiler_shape
  );
  let results = (outs AnyType:$result);
}

def Cute_TiledCopyPartitionSOp : Cute_Op<"tiled_copy_partition_s"> {
  let summary = "Partition source tensor for copy";
  let arguments = (ins 
    AnyType:$tiled_copy,
    Cute_TensorType:$tensor
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_TiledCopyPartitionDOp : Cute_Op<"tiled_copy_partition_d"> {
  let summary = "Partition destination tensor for copy";
  let arguments = (ins 
    AnyType:$tiled_copy,
    Cute_TensorType:$tensor
  );
  let results = (outs Cute_TensorType:$result);
}

def Cute_CopyAtomCallOp : Cute_Op<"copy_atom_call"> {
  let summary = "Execute copy atom";
  let arguments = (ins 
    AnyType:$copy_atom,
    Cute_TensorType:$src,
    Cute_TensorType:$dst
  );
}

def Cute_TiledCopyOp : Cute_Op<"tiled_copy"> {
  let summary = "Execute tiled copy operation";
  let arguments = (ins 
    AnyType:$tiled_copy,
    Cute_TensorType:$src,
    Cute_TensorType:$dst
  );
}

#endif // CUTE_OPS
