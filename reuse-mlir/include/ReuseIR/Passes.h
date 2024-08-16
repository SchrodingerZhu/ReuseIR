#pragma once

#include "ReuseIR/Common.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace reuse_ir {
std::unique_ptr<Pass> createConvertReuseIRToLLVMPass();

#define GEN_PASS_CLASSES
#include "ReuseIR/Passes.h.inc"

} // namespace reuse_ir
} // namespace mlir
