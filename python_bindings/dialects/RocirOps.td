//===- RocirOps.td - Rocir dialect Python bindings --------*- tablegen -*-===//
//
// This file defines the TableGen definitions for Python bindings of the
// Rocir dialect. This is a wrapper that properly includes the main
// Rocir dialect definitions with correct include paths.
//
//===----------------------------------------------------------------------===//

#ifndef PYTHON_BINDINGS_ROCIR_OPS
#define PYTHON_BINDINGS_ROCIR_OPS

// First include the base MLIR definitions
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the Rocir dialect
def Rocir_Dialect : Dialect {
  let name = "rocir";
  let summary = "A MLIR dialect for Rocir tensor operations";
  let description = [{
    The Rocir dialect provides types and operations for hierarchical 
    hierarchical tensor layout manipulation.
  }];
  let cppNamespace = "::mlir::rocir";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;
  
  let extraClassDeclaration = [{
    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &printer) const override;
  }];
}

// Base class for Rocir operations
class Rocir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rocir_Dialect, mnemonic, traits>;

// Type constraints
def Rocir_IntType : Type<CPred<"llvm::isa<::mlir::rocir::IntType>($_self)">, "rocir.int">;
def Rocir_ShapeType : Type<CPred<"llvm::isa<::mlir::rocir::ShapeType>($_self)">, "rocir.shape">;
def Rocir_StrideType : Type<CPred<"llvm::isa<::mlir::rocir::StrideType>($_self)">, "rocir.stride">;
def Rocir_LayoutType : Type<CPred<"llvm::isa<::mlir::rocir::LayoutType>($_self)">, "rocir.layout">;
def Rocir_CoordType : Type<CPred<"llvm::isa<::mlir::rocir::CoordType>($_self)">, "rocir.coord">;

// For Python bindings, we only need the dialect and basic type definitions
// The actual operation bindings will be generated from the C++ definitions

#endif // PYTHON_BINDINGS_ROCIR_OPS
