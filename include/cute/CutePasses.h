#ifndef CUTE_PASSES_H
#define CUTE_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace cute {

std::unique_ptr<Pass> createCuteToStandardPass();
std::unique_ptr<Pass> createCuteToRocmPass();

}
}

#endif // CUTE_PASSES_H
