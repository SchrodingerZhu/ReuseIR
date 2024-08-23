#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROpsEnums.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

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
// RcAcquireOp
mlir::reuse_ir::LogicalResult RcAcquireOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
    return emitOpError("cannot be applied to an unfrozen RC pointer");
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
  RefType input = getObject().getType();
  mlir::Type targetType{};
  if (auto mref = dyn_cast<MRefType>(input.getPointee())) {
    if (input.getFreezingKind().getValue() == FreezingKind::nonfreezing)
      return emitOpError(
          "cannot load a mutable RC pointer through a nonfreezing reference");
    targetType = NullableType::get(getContext(),
                                   RcType::get(getContext(), mref.getPointee(),
                                               mref.getAtomicKind(),
                                               input.getFreezingKind()));
  } else
    targetType = input.getPointee();
  if (targetType != getType())
    return emitOpError("expected to return a value of ")
           << targetType << ", but " << getType() << " is found instead";
  return mlir::reuse_ir::success();
}

// ClosureNewOp
ClosureType ClosureNewOp::getClosureType() {
  return llvm::TypeSwitch<mlir::Type, ClosureType>(getClosure().getType())
      .Case<RcType>(
          [](const RcType &ty) { return cast<ClosureType>(ty.getPointee()); })
      .Default([](const mlir::Type &ty) { return cast<ClosureType>(ty); });
}

mlir::reuse_ir::LogicalResult ClosureNewOp::verify() {
  if (getNumRegions() > 1)
    return emitOpError("cannot have more than one region");
  Region *region = &getRegion(0);
  ClosureType closureTy = getClosureType();
  if (region->getArguments().size() != closureTy.getInputTypes().size())
    return emitOpError("the number of arguments in the region must match the "
                       "number of input types in the closure type");
  if (std::any_of(region->getArguments().begin(), region->getArguments().end(),
                  [&](BlockArgument arg) {
                    return arg.getType() !=
                           closureTy.getInputTypes()[arg.getArgNumber()];
                  }))
    return emitOpError("the types of arguments in the region must match the "
                       "input types in the closure type");
  return mlir::reuse_ir::success();
}

// ClosureYieldOp
mlir::reuse_ir::LogicalResult ClosureYieldOp::verify() {
  ClosureNewOp op = getParentOp();
  ClosureType closureTy = op.getClosureType();
  if (getValue() && !closureTy.getOutputType())
    return emitOpError("cannot yield a value in a closure without output");
  if (!getValue() && closureTy.getOutputType())
    return emitOpError("must yield a value in a closure with output");
  if (getValue().getType() != closureTy.getOutputType())
    return emitOpError("expected to yield a value of ")
           << closureTy.getOutputType() << ", but " << getValue().getType()
           << " is found instead";
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
