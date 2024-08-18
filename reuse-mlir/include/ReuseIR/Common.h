#pragma once

#if defined(__GNUG__) && __has_attribute(visibility)
#define REUSE_IR_DECL_SCOPE [[gnu::visibility("hidden")]] reuse_ir
#else
#define REUSE_IR_DECL_SCOPE reuse_ir
#endif

#if __has_include("mlir/Support/LogicalResult.h")
#include "mlir/Support/LogicalResult.h"
namespace mlir::reuse_ir {
using mlir::LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
} // namespace mlir::reuse_ir
#else
#include "llvm/Support/LogicalResult.h"
namespace mlir::reuse_ir {
using llvm::LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
} // namespace mlir::reuse_ir
#endif
