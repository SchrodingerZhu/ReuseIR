#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROpsEnums.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
// BorrowOp
mlir::reuse_ir::LogicalResult RcBorrowOp::verify() {
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
            size_t size = ty.getInnerTypes().size();
            return getIndex().getZExtValue() < size
                       ? ty.getInnerTypes()[getIndex().getZExtValue()]
                       : mlir::Type{};
          })
          .Case<ArrayType>([&](const ArrayType &ty) {
            if (getIndex().getZExtValue() >= ty.getSizes()[0])
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

template <typename EmitError>
static LogicalResult verifyTokenForRC(Operation *op, TokenType token, RcType rc,
                                      EmitError emitOpError,
                                      bool isReturn = false) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("cannot find the module containing the operation");
  DataLayout dataLayout{module};
  auto rcBoxTy = RcBoxType::get(op->getContext(), rc.getPointee(),
                                rc.getAtomicKind(), rc.getFreezingKind());
  auto size = dataLayout.getTypeSize(rcBoxTy);
  auto align = dataLayout.getTypeABIAlignment(rcBoxTy);
  if (token.getAlignment() != align || token.getSize() != size)
    return emitOpError("expected")
           << (isReturn ? " to return " : " ") << "a nullable token of size "
           << size.getFixedValue() << " and alignment " << align
           << ", but found a nullable token of type " << token;
  return success();
}

// RcReleaseOp
mlir::reuse_ir::LogicalResult RcReleaseOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
    return emitOpError("cannot be applied to an unfrozen RC pointer");
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::frozen &&
      getNumResults() > 0)
    return emitOpError("cannot have any result when applied to a frozen RC "
                       "pointer");
  if (rcPtrTy.getFreezingKind().getValue() == FreezingKind::nonfreezing &&
      getNumResults() != 1)
    return emitOpError("must have a result when applied to a nonfreezing RC "
                       "pointer");
  // get current module
  if (!getToken())
    return mlir::reuse_ir::success();
  auto tokenTy = cast<TokenType>(getToken().getType().getPointer());
  return verifyTokenForRC(
      this->getOperation(), tokenTy, rcPtrTy,
      [&](const Twine &msg) { return emitOpError(msg); }, true);
}

// RcDecreaseOp
mlir::reuse_ir::LogicalResult RcDecreaseOp::verify() {
  RcType rcPtrTy = getRcPtr().getType();
  if (rcPtrTy.getFreezingKind().getValue() != FreezingKind::nonfreezing)
    return emitOpError("can only be applied to a nonfreezing RC pointer");
  return mlir::reuse_ir::success();
}

// CompositeAssembleOp
mlir::reuse_ir::LogicalResult CompositeAssembleOp::verify() {
  CompositeType compositeTy = dyn_cast<CompositeType>(getType());
  if (!compositeTy)
    return emitOpError("must return a composite type");
  if (!compositeTy.isComplete())
    return emitOpError("cannot assemble an incomplete composite type");
  if (compositeTy.getInnerTypes().size() != getNumOperands())
    return emitOpError("the number of operands must match the number of "
                       "fields in the composite type");
  for (size_t i = 0; i < getNumOperands(); ++i)
    if (getOperand(i).getType() != compositeTy.getInnerTypes()[i])
      return emitOpError("the type of operand #")
             << i << " must match the type of the field in the composite type";
  return mlir::reuse_ir::success();
}

// UnionAssembleOp
mlir::reuse_ir::LogicalResult UnionAssembleOp::verify() {
  UnionType unionTy = dyn_cast<UnionType>(getType());
  if (!unionTy)
    return emitOpError("must return a union type");
  if (!unionTy.isComplete())
    return emitOpError("cannot assemble an incomplete union type");
  if (getTag().getZExtValue() >= unionTy.getInnerTypes().size())
    return emitOpError("the tag must be within the range of the union type");
  if (getOperand().getType() !=
      unionTy.getInnerTypes()[getTag().getZExtValue()])
    return emitOpError("the type of the operand must match the type of the "
                       "field in the union type corresponding to the tag");
  return mlir::reuse_ir::success();
}

// RcCreateOp
mlir::reuse_ir::LogicalResult RcCreateOp::verify() {
  RcType rcPtrTy = getType();
  switch (rcPtrTy.getFreezingKind().getValue()) {
  case FreezingKind::frozen:
    return emitOpError("cannot create a frozen RC pointer");
  case FreezingKind::nonfreezing:
    if (getRegion())
      return emitOpError("cannot have a region when creating a nonfreezing RC "
                         "pointer");
    break;
  case FreezingKind::unfrozen:
    if (!getRegion())
      return emitOpError("must have a region when creating an unfrozen RC "
                         "pointer");
    if (getToken())
      return emitOpError("cannot have a token when creating an unfrozen RC "
                         "pointer");
    break;
  }
  if (auto token = getToken()) {
    auto tokenTy = cast<TokenType>(token.getType().getPointer());
    return verifyTokenForRC(this->getOperation(), tokenTy, rcPtrTy,
                            [&](const Twine &msg) { return emitOpError(msg); });
  }
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

static ParseResult parseFusedIndices(OpAsmParser &parser,
                                     DenseI64ArrayAttr &attr) {
  if (parser.parseOptionalKeyword("fused").failed()) {
    attr = DenseI64ArrayAttr::get(parser.getContext(), {});
    return success();
  }
  if (parser.parseLParen() || parser.parseAttribute(attr) ||
      parser.parseRParen())
    return LogicalResult::failure();
  return success();
}
static void printFusedIndices(OpAsmPrinter &printer, Operation *,
                              const DenseI64ArrayAttr &attr) {
  if (attr.empty())
    return;
  printer << "fused(" << attr << ")";
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.cpp.inc"
