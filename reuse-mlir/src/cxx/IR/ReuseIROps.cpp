#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
// IncOp
mlir::reuse_ir::LogicalResult IncOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (auto attr = rcPtrTy.getFrozen())
    if (!attr.getValue())
      return emitOpError(
          "cannot increase a non-frozen but freezable RC pointer");
  if (getCount() && *getCount() == 0)
    return emitError("the amount of increment must be non-zero");
  return mlir::reuse_ir::success();
}
// ProjOp
mlir::reuse_ir::LogicalResult ProjOp::verify() {
  mlir::Type innerType;
  if (auto type = llvm::dyn_cast<RefType>(getObject().getType()))
    innerType = type.getPointee();

  if (auto type = llvm::dyn_cast<RcType>(getObject().getType()))
    innerType = type.getPointee();

  innerType = innerType ? innerType : getObject().getType();

  // TODO: handle verification recursively.
  return mlir::reuse_ir::success();
}

} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.cpp.inc"
