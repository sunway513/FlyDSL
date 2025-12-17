//===- RocirDCE.cpp - Trivial DCE (standalone) ----------------------------===//
//
// Provides a general dead-code elimination pass for our embedded MLIR
// environment. Upstream MLIR's DCE passes are not registered in Python here, so
// we provide a standalone pass under a distinct name ('trivial-dce').
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::rocir;

namespace {

#define GEN_PASS_DEF_ROCIRTRIVIALDCEPASS
#include "rocir/RocirPasses.h.inc"

struct RocirTrivialDCEPass
    : public impl::RocirTrivialDCEPassBase<RocirTrivialDCEPass> {
  using impl::RocirTrivialDCEPassBase<
      RocirTrivialDCEPass>::RocirTrivialDCEPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Iteratively erase dead ops to a fixed point, since erasing one op may
    // render its operands dead.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      module.walk<WalkOrder::PostOrder>([&](Operation *op) {
        if (mlir::isOpTriviallyDead(op))
          toErase.push_back(op);
      });
      for (Operation *op : toErase) {
        op->erase();
        changed = true;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace rocir {

std::unique_ptr<Pass> createRocirTrivialDCEPass() {
  return std::make_unique<RocirTrivialDCEPass>();
}

} // namespace rocir
} // namespace mlir


