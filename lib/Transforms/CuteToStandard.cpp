#include "cute/CuteDialect.h"
#include "cute/CuteOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::cute;

namespace {

// Lower make_coord to vector.from_elements
struct MakeCoordOpLowering : public OpConversionPattern<MakeCoordOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MakeCoordOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    auto indices = adaptor.getIndices(); // ValueRange
    
    int64_t rank = indices.size();
    auto vectorType = VectorType::get({rank}, rewriter.getIndexType());
    
    Value vector = rewriter.create<vector::FromElementsOp>(loc, vectorType, indices);
    
    rewriter.replaceOp(op, vector);
    return success();
  }
};

// Lower make_tensor to just return the pointer (memref)
struct MakeTensorOpLowering : public OpConversionPattern<MakeTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MakeTensorOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

struct Crd2IdxOpLowering : public OpConversionPattern<Crd2IdxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      Crd2IdxOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    Value coordVec = adaptor.getCoord();
    
    auto layoutType = op.getLayout().getType().cast<LayoutType>();
    auto shapeType = layoutType.getShape();
    auto strideType = layoutType.getStride();
    
    ArrayRef<int64_t> shape = shapeType.getExtents();
    ArrayRef<int64_t> stride = strideType.getStrides();
    
    int64_t rank = shape.size();
    
    Value result = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    
    for (int64_t i = 0; i < rank; ++i) {
      Value idxVal = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value coordI = rewriter.create<vector::ExtractElementOp>(loc, coordVec, idxVal);
      
      Value strideI = rewriter.create<arith::ConstantIndexOp>(loc, stride[i]);
      Value term = rewriter.create<arith::MulIOp>(loc, coordI, strideI);
      
      result = rewriter.create<arith::AddIOp>(loc, result, term);
    }
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CopyOpLowering : public OpConversionPattern<CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      CopyOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    Value src = adaptor.getSrc(); // MemRef
    Value dst = adaptor.getDst(); // MemRef
    
    auto srcLayout = op.getSrc().getType().cast<TensorType>().getLayout();
    auto dstLayout = op.getDst().getType().cast<TensorType>().getLayout();
    
    auto shape = srcLayout.getShape().getExtents();
    int64_t size = 1;
    for (auto dim : shape)
      size *= dim;
    
    int64_t vectorWidth = analyzeVectorWidth(srcLayout, dstLayout);
    
    if (vectorWidth > 1) {
      generateVectorizedCopy(rewriter, loc, src, dst, size, vectorWidth);
    } else {
      auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto ub = rewriter.create<arith::ConstantIndexOp>(loc, size);
      auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      
      auto loop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
      rewriter.setInsertionPointToStart(loop.getBody());
      
      Value idx = loop.getInductionVar();
      Value data = rewriter.create<memref::LoadOp>(loc, src, idx); 
      rewriter.create<memref::StoreOp>(loc, data, dst, idx);
      
      rewriter.setInsertionPointAfter(loop);
    }
    
    rewriter.eraseOp(op);
    return success();
  }

private:
  int64_t analyzeVectorWidth(LayoutType srcLayout, LayoutType dstLayout) const {
    auto srcStride = srcLayout.getStride().getStrides();
    auto dstStride = dstLayout.getStride().getStrides();
    
    if (srcStride.empty() || dstStride.empty()) return 1;

    if (srcStride.back() == 1 && dstStride.back() == 1) {
      int64_t innerDim = srcLayout.getShape().getExtents().back();
      for (int64_t width : {16, 8, 4, 2}) {
        if (innerDim % width == 0)
          return width;
      }
    }
    return 1;
  }
  
  void generateVectorizedCopy(ConversionPatternRewriter &rewriter,
                               Location loc, Value src, Value dst,
                               int64_t size, int64_t vectorWidth) const {
    auto elemType = src.getType().cast<MemRefType>().getElementType();
    auto vecType = VectorType::get({vectorWidth}, elemType);
    
    auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ub = rewriter.create<arith::ConstantIndexOp>(loc, size);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);
    
    auto loop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(loop.getBody());
    
    Value idx = loop.getInductionVar();
    Value vec = rewriter.create<vector::LoadOp>(loc, vecType, src, ValueRange{idx});
    rewriter.create<vector::StoreOp>(loc, vec, dst, ValueRange{idx});
  }
};

struct LocalPartitionOpLowering : public OpConversionPattern<LocalPartitionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      LocalPartitionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    auto loc = op.getLoc();
    Value tensor = adaptor.getTensor(); // MemRef
    
    auto tilerType = op.getTiler().getType().cast<LayoutType>();
    int64_t threadIdx = op.getThreadIdx();
    
    int64_t tileSize = computeTileSize(tilerType);
    int64_t offset = threadIdx * tileSize;
    
    auto memRefType = tensor.getType().cast<MemRefType>();
    
    if (memRefType.getRank() == 1) {
        auto resultLayout = op.getResult().getType().cast<TensorType>().getLayout();
        int64_t resultSize = 1;
        for(auto d : resultLayout.getShape().getExtents()) resultSize *= d;
        
        Value subview = rewriter.create<memref::SubViewOp>(
            loc, tensor, 
            ArrayRef<int64_t>{offset}, 
            ArrayRef<int64_t>{resultSize}, 
            ArrayRef<int64_t>{1}
        );
        rewriter.replaceOp(op, subview);
    } else {
        rewriter.replaceOp(op, tensor);
    }

    return success();
  }

private:
  int64_t computeTileSize(LayoutType layout) const {
    int64_t size = 1;
    for (auto dim : layout.getShape().getExtents())
      size *= dim;
    return size;
  }
};

struct CuteToStandardPass : public PassWrapper<CuteToStandardPass, 
                                                OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CuteToStandardPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, 
                    scf::SCFDialect,
                    memref::MemRefDialect,
                    func::FuncDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    
    ConversionTarget target(*context);
    target.addLegalDialect<arith::ArithDialect, 
                          scf::SCFDialect,
                          memref::MemRefDialect,
                          func::FuncDialect,
                          vector::VectorDialect>();
    target.addIllegalDialect<CuteDialect>();
    
    RewritePatternSet patterns(context);
    patterns.add<MakeCoordOpLowering,
                 MakeTensorOpLowering,
                 Crd2IdxOpLowering,
                 CopyOpLowering,
                 LocalPartitionOpLowering>(context);
    
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::cute::createCuteToStandardPass() {
  return std::make_unique<CuteToStandardPass>();
}
