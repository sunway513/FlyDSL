//===- RocirNvgpuDialect.td - CuTe NVGPU Dialect Definition --*- tablegen -*-===//
//
// CuTe GPU Hardware-Aware Dialect
// Extends rocdsl with GPU-specific types that wrap MLIR nvgpu operations
//
//===----------------------------------------------------------------------===//

#ifndef ROCIR_NVGPU_DIALECT
#define ROCIR_NVGPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def RocirNvgpu_Dialect : Dialect {
  let name = "cute_nvgpu";
  let cppNamespace = "::mlir::rocir::nvgpu";
  let summary = "GPU hardware-aware CuTe IR";
  let description = [{
    CuTe NVGPU IR extends rocdsl with GPU-specific types and operations.
    It wraps MLIR nvgpu operations with CuTe Layout-aware interfaces.
    
    **Design Philosophy**: 
    - Accepts rocdsl::LayoutType/TileType as partition descriptors
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

class RocirNvgpu_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RocirNvgpu_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class RocirNvgpu_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RocirNvgpu_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class RocirNvgpu_Op<string mnemonic, list<Trait> traits = []>
    : Op<RocirNvgpu_Dialect, mnemonic, traits>;

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
  let cppNamespace = "::mlir::rocir::nvgpu";
}

def ReductionKindEnum : I32EnumAttr<"ReductionKind",
    "Reduction operation type",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Sum", 1, "sum">,
      I32EnumAttrCase<"Max", 2, "max">,
      I32EnumAttrCase<"Min", 3, "min">
    ]> {
  let cppNamespace = "::mlir::rocir::nvgpu";
}

def SparseMetadataFormat : I32EnumAttr<"SparseMetadataFormat",
    "Sparse matrix metadata format",
    [
      I32EnumAttrCase<"None", 0, "none">,
      I32EnumAttrCase<"Compressed2To1", 1, "2:1">,
      I32EnumAttrCase<"Compressed4To2", 2, "4:2">
    ]> {
  let cppNamespace = "::mlir::rocir::nvgpu";
}

//===----------------------------------------------------------------------===//
// MMA Atom Types
//===----------------------------------------------------------------------===//

def RocirNvgpu_MmaAtomType : RocirNvgpu_Type<"MmaAtom", "mma_atom"> {
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

def RocirNvgpu_TiledMmaType : RocirNvgpu_Type<"TiledMma", "tiled_mma"> {
  let summary = "Tiled MMA descriptor";
  let description = [{
    TiledMma = MmaAtom + LayoutTV + TilerMNK
    
    Composition:
    - atom: underlying MMA instruction
    - layout_v: thread value layout (from rocdsl)
    - tiler_mnk: hierarchical tiling (from rocdsl)
  }];
  
  let parameters = (ins
    "RocirNvgpu_MmaAtomType":$atom,
    "Type":$layout_v,  // rocdsl::LayoutType
    "Type":$tiler_mnk  // rocdsl::TileType
  );
  
  let assemblyFormat = "`<` $atom `,` $layout_v `,` $tiler_mnk `>`";
}

//===----------------------------------------------------------------------===//
// Copy Atom Types
//===----------------------------------------------------------------------===//

def RocirNvgpu_CopyAtomType : RocirNvgpu_Type<"CopyAtom", "copy_atom"> {
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

def RocirNvgpu_TiledCopyType : RocirNvgpu_Type<"TiledCopy", "tiled_copy"> {
  let summary = "Tiled copy descriptor";
  let description = [{
    TiledCopy = CopyAtom + LayoutTV + TilerShape
  }];
  
  let parameters = (ins
    "RocirNvgpu_CopyAtomType":$atom,
    "Type":$layout_v,
    "Type":$tiler_shape
  );
  
  let assemblyFormat = "`<` $atom `,` $layout_v `,` $tiler_shape `>`";
}

//===----------------------------------------------------------------------===//
// TMA (Tensor Memory Accelerator) Types
//===----------------------------------------------------------------------===//

def RocirNvgpu_TmaLoadType : RocirNvgpu_Type<"TmaLoad", "tma_load"> {
  let summary = "TMA load operation descriptor (SM90+)";
  let parameters = (ins
    "Type":$elem_type,
    ArrayRefParameter<"int64_t">:$tile_shape,
    "StringAttr":$swizzle  // none/32B/64B/128B
  );
}

def RocirNvgpu_TmaStoreType : RocirNvgpu_Type<"TmaStore", "tma_store"> {
  let summary = "TMA store operation descriptor (SM90+)";
  let parameters = (ins
    "Type":$elem_type,
    ArrayRefParameter<"int64_t">:$tile_shape,
    "StringAttr":$swizzle
  );
}

def RocirNvgpu_TmaLoadExecType : RocirNvgpu_Type<"TmaLoadExec", "tma_load_exec"> {
  let summary = "Executable TMA load with barrier";
  let parameters = (ins
    "RocirNvgpu_TmaLoadType":$descriptor,
    "Type":$barrier_type  // nvgpu::MBarrier
  );
}

def RocirNvgpu_TmaStoreExecType : RocirNvgpu_Type<"TmaStoreExec", "tma_store_exec"> {
  let summary = "Executable TMA store with barrier";
  let parameters = (ins
    "RocirNvgpu_TmaStoreType":$descriptor,
    "Type":$barrier_type
  );
}

//===----------------------------------------------------------------------===//
// Runtime State Types
//===----------------------------------------------------------------------===//

def RocirNvgpu_AtomValueType : RocirNvgpu_Type<"AtomValue", "atom_value"> {
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

def RocirNvgpu_AtomFieldAttr : RocirNvgpu_Attr<"AtomField", "atom_field"> {
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

#endif // ROCIR_NVGPU_DIALECT
