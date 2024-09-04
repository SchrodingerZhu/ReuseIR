#pragma once

#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace reuse_ir {
std::unique_ptr<Pass> createConvertReuseIRToLLVMPass();
std::unique_ptr<Pass> createReuseIRClosureOutliningPass();
std::unique_ptr<Pass> createReuseIRExpandControlFlowPass();
std::unique_ptr<Pass> createReuseIRExpandControlFlowPass(
    const struct ReuseIRExpandControlFlowOptions &options);
std::unique_ptr<Pass> createReuseIRAcquireReleaseFusionPass();
std::unique_ptr<Pass> createReuseIRInferUnionTagPass();
std::unique_ptr<Pass> createReuseIRPrintReuseAnalysisPass();

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "ReuseIR/Passes.h.inc"

inline constexpr llvm::StringLiteral NESTED = "reuse_ir.nested";
inline constexpr llvm::StringLiteral RELEASE = "reuse_ir.release";

} // namespace reuse_ir
} // namespace mlir
