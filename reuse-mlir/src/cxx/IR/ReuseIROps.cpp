#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
// BorrowOp
mlir::reuse_ir::LogicalResult BorrowOp::verify() {
  RcType input = getObject().getType();
  RefType output = getType();
  if (input.getFreezingKind() != output.getFreezingKind())
    return emitError("the borrowed reference must have the consistent "
                       "freezing state with the RC pointer");
  if (input.getPointee() != output.getPointee())
    return emitError("the borrowed reference must have the consistent "
                       "pointee type with the RC pointer");
  return mlir::reuse_ir::success();
}
// IncOp
mlir::reuse_ir::LogicalResult IncOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
    return emitOpError("cannot increase a non-frozen but freezable RC pointer");
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

template <StringLiteral Literal> struct ParseKeywordAsUnitAttr {
  OptionalParseResult operator()(OpAsmParser &parser, UnitAttr &attr) {
    if (parser.parseOptionalKeyword(Literal).succeeded())
      attr = UnitAttr::get(parser.getContext());
    return success();
  }
};

template <StringLiteral Literal> struct PrintKeywordAsUnitAttr {
  void operator()(OpAsmPrinter &printer, Operation *, const UnitAttr &attr) {
    if (attr)
      printer.printKeywordOrString(Literal);
  }
};

static ParseKeywordAsUnitAttr<"as_reference"_str> parseAsReferenceAttr;
static PrintKeywordAsUnitAttr<"as_reference"_str> printAsReferenceAttr;
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.cpp.inc"
