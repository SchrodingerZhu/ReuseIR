#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Common.h"

#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
// IncOp
llvm::LogicalResult IncOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (auto attr = rcPtrTy.getFrozen())
    if (!attr.getValue())
      return emitOpError(
          "Cannot increase a non-frozen but freezable RC pointer");
  if (getCount() && *getCount() == 0)
    return emitError("The amount of increment must be non-zero");
  return llvm::success();
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#pragma GCC visibility push(hidden)
#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.cpp.inc"
#pragma GCC visibility pop
