#include "cute/CuteDialect.h"
#include "cute/CutePasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::cute;

namespace {

struct Crd2IdxOpLowering : public OpConversionPattern<Crd2IdxOp> {
  using OpConversionPattern<Crd2IdxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Crd2IdxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getOperation()->getLoc();
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.replaceOp(op.getOperation(), zero.getResult());
    return success();
  }
};

#define GEN_PASS_DEF_CUTETOSTANDARDPASS
#include "cute/CutePasses.h.inc"

struct CuteToStandardPass
    : public impl::CuteToStandardPassBase<CuteToStandardPass> {
  
  using impl::CuteToStandardPassBase<CuteToStandardPass>::CuteToStandardPassBase;
  
  void runOnOperation() override {
    auto module = getOperation();
    ConversionTarget target(getContext());
    
    target.addLegalDialect<arith::ArithDialect,
                          memref::MemRefDialect,
                          func::FuncDialect>();
    
    target.addIllegalDialect<CuteDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<Crd2IdxOpLowering>(patterns.getContext());

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}

namespace mlir {
namespace cute {

std::unique_ptr<Pass> createCuteToStandardPass() {
  return std::make_unique<CuteToStandardPass>();
}

}
}
