#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROpsEnums.h"
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
// ValueToRefOp
mlir::reuse_ir::LogicalResult ValueToRefOp::verify() {
  if (getResult().getType().getFreezingKind().getValue() !=
      FreezingKind::nonfreezing)
    return emitOpError("must return a nonfreezing reference");
  if (getValue().getType() != getResult().getType().getPointee())
    return emitOpError("must return a reference whose pointee is of the "
                       "same type of the input");
  return mlir::reuse_ir::success();
}

// ProjOp
mlir::reuse_ir::LogicalResult ProjOp::verify() {
  RefType input = getObject().getType();
  if (input.getFreezingKind() != getType().getFreezingKind())
    return emitOpError(
        "must return a reference with the same freezing kind as the input");
  if (!isProjectable(input.getPointee()))
    return emitOpError(
        "must operate on a reference to a composite or an array");
  auto targetType =
      llvm::TypeSwitch<mlir::Type, mlir::Type>(input.getPointee())
          .Case<CompositeType>([&](const CompositeType &ty) {
            size_t size = ty.getMemberTypes().size();
            return getIndex() < size ? ty.getMemberTypes()[getIndex()]
                                     : mlir::Type{};
          })
          .Case<ArrayType>([&](const ArrayType &ty) {
            if (getIndex() >= ty.getSizes()[0])
              return mlir::Type{};
            if (ty.getSizes().size() == 1)
              return ty.getElementType();
            return cast<mlir::Type>(ArrayType::get(
                getContext(), ty.getElementType(), ty.getSizes().drop_front()));
          })
          .Default([](auto &&) { return mlir::Type{}; });
  if (!targetType)
    return emitOpError("cannot project with an out-of-bound index");
  if (targetType != getType().getPointee())
    return emitOpError("expected to return a reference to ")
           << targetType << ", but found a reference to "
           << getType().getPointee();

  return mlir::reuse_ir::success();
}

// LoadOp
mlir::reuse_ir::LogicalResult LoadOp::verify() {
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
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.cpp.inc"
