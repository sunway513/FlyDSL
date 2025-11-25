#ifndef ROCIR_PASSES_H
#define ROCIR_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace rocir {

std::unique_ptr<Pass> createRocirToStandardPass();
std::unique_ptr<Pass> createCuteToRocmPass();

}
}

#endif // ROCIR_PASSES_H
