//===- CuteNvgpuDialect.td - CuTe NVGPU Dialect Definition --*- tablegen -*-===//
//
// CuTe GPU Hardware-Aware Dialect
// Extends cute_ir with GPU-specific types that wrap MLIR nvgpu operations
//
//===----------------------------------------------------------------------===//

#ifndef CUTE_NVGPU_DIALECT
#define CUTE_NVGPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def CuteNvgpu_Dialect : Dialect {
  let name = "cute_nvgpu";
  let cppNamespace = "::mlir::cute::nvgpu";
  let summary = "GPU hardware-aware CuTe IR";
  let description = [{
    CuTe NVGPU IR extends cute_ir with GPU-specific types and operations.
    It wraps MLIR nvgpu operations with CuTe Layout-aware interfaces.
    
    **Design Philosophy**: 
    - Accepts cute_ir::LayoutType/TileType as partition descriptors
    - Encodes GPU instruction metadata (shape, element types, layout)
    - Generates architecture-specific code (SM80/SM90/SM100)
    
    **Lowering Chain**:
    cute_nvgpu_ir → gpu/nvgpu dialect → nvvm → LLVM IR → PTX
  }];
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class CuteNvgpu_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CuteNvgpu_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class CuteNvgpu_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<CuteNvgpu_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class CuteNvgpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<CuteNvgpu_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Enumerations
//===----------------------------------------------------------------------===//

def MmaOperandEnum : I32EnumAttr<"MmaOperand",
    "MMA matrix operand type",
    [
      I32EnumAttrCase<"A", 0, "a">,
      I32EnumAttrCase<"B", 1, "b">,
      I32EnumAttrCase<"C", 2, "c">
    ]> {
  let cppNamespace = "::mlir::cute::nvgpu";
}

def ReductionKindEnum : I32EnumAttr<"ReductionKind",
    "Reduction operation type",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Sum", 1, "sum">,
      I32EnumAttrCase<"Max", 2, "max">,
      I32EnumAttrCase<"Min", 3, "min">
    ]> {
  let cppNamespace = "::mlir::cute::nvgpu";
}

def SparseMetadataFormat : I32EnumAttr<"SparseMetadataFormat",
    "Sparse matrix metadata format",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Compressed2To1", 1, "2:1">,
      I32EnumAttrCase<"Compressed4To2", 2, "4:2">
    ]> {
  let cppNamespace = "::mlir::cute::nvgpu";
}

//===----------------------------------------------------------------------===//
// MMA Atom Types
//===----------------------------------------------------------------------===//

def CuteNvgpu_MmaAtomType : CuteNvgpu_Type<"MmaAtom", "mma_atom"> {
  let summary = "MMA instruction atom descriptor";
  let description = [{
    Encodes metadata for a single MMA instruction:
    - Shape: [M, N, K] dimensions
    - Element types: A, B, C/D operand types
    - Layout: register/shared memory layout
    - Architecture: target GPU (SM80/SM90/SM100)
    
    Example:
    ```mlir
    !cute_nvgpu.mma_atom<!f16, !f16, !f32, [16, 8, 16], SM80>
    ```
  }];
  
  let parameters = (ins
    "Type":$elem_type_a,
    "Type":$elem_type_b,
    "Type":$elem_type_c,
    ArrayRefParameter<"int64_t">:$shape,  // [M, N, K]
    "StringAttr":$arch  // SM80/SM90/SM100
  );
  
  let assemblyFormat = "`<` $elem_type_a `,` $elem_type_b `,` $elem_type_c `,` `[` $shape `]` `,` $arch `>`";
}

def CuteNvgpu_TiledMmaType : CuteNvgpu_Type<"TiledMma", "tiled_mma"> {
  let summary = "Tiled MMA descriptor";
  let description = [{
    TiledMma = MmaAtom + LayoutTV + TilerMNK
    
    Composition:
    - atom: underlying MMA instruction
    - layout_v: thread value layout (from cute_ir)
    - tiler_mnk: hierarchical tiling (from cute_ir)
  }];
  
  let parameters = (ins
    "CuteNvgpu_MmaAtomType":$atom,
    "Type":$layout_v,  // cute_ir::LayoutType
    "Type":$tiler_mnk  // cute_ir::TileType
  );
  
  let assemblyFormat = "`<` $atom `,` $layout_v `,` $tiler_mnk `>`";
}

//===----------------------------------------------------------------------===//
// Copy Atom Types
//===----------------------------------------------------------------------===//

def CuteNvgpu_CopyAtomType : CuteNvgpu_Type<"CopyAtom", "copy_atom"> {
  let summary = "Copy instruction atom descriptor";
  let description = [{
    Encodes metadata for copy operations:
    - Instruction: ldmatrix, cpasync, TMA
    - Element type and vector width
    - Source/destination address spaces
  }];
  
  let parameters = (ins
    "Type":$elem_type,
    "int64_t":$vector_width,
    "StringAttr":$instruction,  // ldmatrix/cpasync/tma
    "StringAttr":$src_space,    // register/shared/global
    "StringAttr":$dst_space
  );
  
  let assemblyFormat = "`<` $elem_type `,` $vector_width `,` $instruction `,` $src_space `->` $dst_space `>`";
}

def CuteNvgpu_TiledCopyType : CuteNvgpu_Type<"TiledCopy", "tiled_copy"> {
  let summary = "Tiled copy descriptor";
  let description = [{
    TiledCopy = CopyAtom + LayoutTV + TilerShape
  }];
  
  let parameters = (ins
    "CuteNvgpu_CopyAtomType":$atom,
    "Type":$layout_v,
    "Type":$tiler_shape
  );
  
  let assemblyFormat = "`<` $atom `,` $layout_v `,` $tiler_shape `>`";
}

//===----------------------------------------------------------------------===//
// TMA (Tensor Memory Accelerator) Types
//===----------------------------------------------------------------------===//

def CuteNvgpu_TmaLoadType : CuteNvgpu_Type<"TmaLoad", "tma_load"> {
  let summary = "TMA load operation descriptor (SM90+)";
  let parameters = (ins
    "Type":$elem_type,
    ArrayRefParameter<"int64_t">:$tile_shape,
    "StringAttr":$swizzle  // none/32B/64B/128B
  );
}

def CuteNvgpu_TmaStoreType : CuteNvgpu_Type<"TmaStore", "tma_store"> {
  let summary = "TMA store operation descriptor (SM90+)";
  let parameters = (ins
    "Type":$elem_type,
    ArrayRefParameter<"int64_t">:$tile_shape,
    "StringAttr":$swizzle
  );
}

def CuteNvgpu_TmaLoadExecType : CuteNvgpu_Type<"TmaLoadExec", "tma_load_exec"> {
  let summary = "Executable TMA load with barrier";
  let parameters = (ins
    "CuteNvgpu_TmaLoadType":$descriptor,
    "Type":$barrier_type  // nvgpu::MBarrier
  );
}

def CuteNvgpu_TmaStoreExecType : CuteNvgpu_Type<"TmaStoreExec", "tma_store_exec"> {
  let summary = "Executable TMA store with barrier";
  let parameters = (ins
    "CuteNvgpu_TmaStoreType":$descriptor,
    "Type":$barrier_type
  );
}

//===----------------------------------------------------------------------===//
// Runtime State Types
//===----------------------------------------------------------------------===//

def CuteNvgpu_AtomValueType : CuteNvgpu_Type<"AtomValue", "atom_value"> {
  let summary = "Runtime value container for Atom fields";
  let description = [{
    Holds runtime-computed values for MmaAtom/CopyAtom fields:
    - TMA descriptor pointers
    - Barrier handles
    - Compute capability check results
  }];
  
  let parameters = (ins
    "Type":$atom_type,
    "StringAttr":$field_name
  );
  
  let assemblyFormat = "`<` $atom_type `,` $field_name `>`";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def CuteNvgpu_AtomFieldAttr : CuteNvgpu_Attr<"AtomField", "atom_field"> {
  let summary = "Atom field identifier";
  let description = [{
    Named fields for MmaAtom/CopyAtom:
    - "tma_descriptor": TMA descriptor pointer
    - "barrier": Barrier handle
    - "compute_capability": Required SM version
  }];
  
  let parameters = (ins
    "StringAttr":$name,
    "Type":$value_type
  );
  
  let assemblyFormat = "`<` $name `:` $value_type `>`";
}

#endif // CUTE_NVGPU_DIALECT
