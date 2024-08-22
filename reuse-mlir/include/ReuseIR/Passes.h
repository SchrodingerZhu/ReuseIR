#pragma once

#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace reuse_ir {
std::unique_ptr<Pass> createConvertReuseIRToLLVMPass();
std::unique_ptr<Pass> createReuseIRClosureOutliningPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "ReuseIR/Passes.h.inc"

} // namespace reuse_ir
} // namespace mlir
