//===- CuteNvgpuToNvgpu.cpp - Lower cute_nvgpu to nvgpu dialect ----------===//
//
// This pass converts cute_nvgpu operations to MLIR nvgpu operations
//
//===----------------------------------------------------------------------===//

#include "cute/CuteDialect.h"
#include "cute/CuteOps.h"
#include "cute/CuteNvgpuDialect.h"
#include "cute/CuteNvgpuOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace cute {

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CuteNvgpuToNvgpuPass : public PassWrapper<CuteNvgpuToNvgpuPass,
                                                  OperationPass<gpu::GPUModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CuteNvgpuToNvgpuPass)

  CuteNvgpuToNvgpuPass() = default;
  CuteNvgpuToNvgpuPass(const CuteNvgpuToNvgpuPass &) {}
  CuteNvgpuToNvgpuPass(std::string arch, bool tma) 
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
    // target.addIllegalDialect<CuteNvgpuDialect>();
    
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

} // namespace cute

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createCuteNvgpuToNvgpuPass() {
  return std::make_unique<cute::CuteNvgpuToNvgpuPass>();
}

std::unique_ptr<Pass> createCuteNvgpuToNvgpuPass(std::string targetArch, bool enableTMA) {
  return std::make_unique<cute::CuteNvgpuToNvgpuPass>(targetArch, enableTMA);
}

} // namespace mlir
