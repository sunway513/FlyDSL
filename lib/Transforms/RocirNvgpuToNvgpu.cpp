//===- RocirNvgpuToNvgpu.cpp - Lower Rocir NVGPU to nvgpu dialect ---------===//
//
// This pass converts Rocir NVGPU operations to MLIR nvgpu operations
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "rocir/RocirNvgpuDialect.h"
#include "rocir/RocirNvgpuOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace rocir {

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct RocirNvgpuToNvgpuPass : public PassWrapper<RocirNvgpuToNvgpuPass,
                                                  OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RocirNvgpuToNvgpuPass)

  RocirNvgpuToNvgpuPass() = default;
  RocirNvgpuToNvgpuPass(const RocirNvgpuToNvgpuPass &) {}
  RocirNvgpuToNvgpuPass(std::string arch, bool tma) 
      : targetArch(arch), enableTMA(tma) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<nvgpu::NVGPUDialect,
                    gpu::GPUDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto gpuModule = getOperation();
    auto *context = &getContext();
    
    ConversionTarget target(*context);
    target.addLegalDialect<nvgpu::NVGPUDialect,
                          gpu::GPUDialect,
                          vector::VectorDialect>();
    // target.addIllegalDialect<RocirNvgpuDialect>();
    
    RewritePatternSet patterns(context);
    
    // Add all lowering patterns
    // patterns.add<WarpgroupMmaOpLowering, ...>(context);
    
    // if (failed(applyPartialConversion(gpuModule, target, std::move(patterns)))) {
    //   signalPassFailure();
    // }
  }

  Option<std::string> targetArch{
      *this, "target-arch",
      llvm::cl::desc("Target GPU architecture (sm_80, sm_90, sm_100)"),
      llvm::cl::init("sm_90")};
  
  Option<bool> enableTMA{
      *this, "enable-tma",
      llvm::cl::desc("Enable TMA operations (SM90+)"),
      llvm::cl::init(true)};
};

} // namespace rocir

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createRocirNvgpuToNvgpuPass() {
  return std::make_unique<cute::RocirNvgpuToNvgpuPass>();
}

std::unique_ptr<Pass> createRocirNvgpuToNvgpuPass(std::string targetArch, bool enableTMA) {
  return std::make_unique<cute::RocirNvgpuToNvgpuPass>(targetArch, enableTMA);
}

} // namespace mlir
