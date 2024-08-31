#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <memory>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

class TokenEnsureExpansionPattern : public OpRewritePattern<TokenEnsureOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TokenEnsureOp op,
                                PatternRewriter &rewriter) const final {
    auto tokenTy =
        cast<TokenType>(op.getNullableToken().getType().getPointer());
    auto isNonNull =
        rewriter.create<NullableCheckOp>(op->getLoc(), op.getNullableToken());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, isNonNull,
        [&](OpBuilder &builder, Location loc) {
          auto coerced = builder.create<NullableCoerceOp>(
              loc, tokenTy, op.getNullableToken());
          builder.create<scf::YieldOp>(loc, coerced->getResults());
        },
        [&](OpBuilder &builder, Location loc) {
          auto allocated = builder.create<TokenAllocOp>(loc, tokenTy);
          builder.create<scf::YieldOp>(loc, allocated->getResults());
        });
    return LogicalResult::success();
  }
};

class TokenFreeExpansionPattern : public OpRewritePattern<TokenFreeOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TokenFreeOp op,
                                PatternRewriter &rewriter) const final {
    auto nullableTy = dyn_cast<NullableType>(op.getToken().getType());
    if (!nullableTy)
      return LogicalResult::failure();
    auto tokenTy = cast<TokenType>(nullableTy.getPointer());
    auto isNonNull =
        rewriter.create<NullableCheckOp>(op->getLoc(), op.getToken());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, isNonNull,
        [&](OpBuilder &builder, Location loc) {
          auto coerced =
              builder.create<NullableCoerceOp>(loc, tokenTy, op.getToken());
          builder.create<TokenFreeOp>(loc, coerced);
          builder.create<scf::YieldOp>(loc);
        },
        nullptr);
    return LogicalResult::success();
  }
};

class RcReleaseExpansionPattern : public OpRewritePattern<RcReleaseOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RcReleaseOp op,
                                PatternRewriter &rewriter) const final {
    RcType rcTy = op.getRcPtr().getType();
    // we only need to expand for nonfreezing RC pointers
    if (rcTy.getFreezingKind().getValue() != FreezingKind::nonfreezing)
      return LogicalResult::failure();
    auto decreaseOp =
        rewriter.create<RcDecreaseOp>(op->getLoc(), op.getRcPtr());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, decreaseOp,
        [&](OpBuilder &builder, Location loc) {
          auto refTy = RefType::get(getContext(), rcTy.getPointee(),
                                    rcTy.getFreezingKind());
          auto borrowed = builder.create<RcBorrowOp>(loc, refTy, op.getRcPtr());
          builder.create<DestroyOp>(loc, borrowed, op.getTagAttr(),
                                    op.getFusedIndices());
          auto resTy = cast<NullableType>(op.getResultTypes()[0]);
          auto token = builder.create<RcTokenizeOp>(loc, resTy.getPointer(),
                                                    op.getRcPtr());
          auto nonnull = builder.create<NullableNonNullOp>(loc, resTy, token);
          builder.create<scf::YieldOp>(loc, nonnull->getResults());
        },
        [&](OpBuilder &builder, Location loc) {
          auto null = builder.create<NullableNullOp>(loc, op.getResultTypes());
          builder.create<scf::YieldOp>(loc, null->getResults());
        });
    return LogicalResult::success();
  }
};

struct ReuseIRExpandControlFlowPass
    : public ReuseIRExpandControlFlowBase<ReuseIRExpandControlFlowPass> {
  using ReuseIRExpandControlFlowBase::ReuseIRExpandControlFlowBase;
  void runOnOperation() override final;
};

void ReuseIRExpandControlFlowPass::runOnOperation() {
  auto module = getOperation();
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  patterns.add<TokenEnsureExpansionPattern, TokenFreeExpansionPattern,
               RcReleaseExpansionPattern>(&getContext());

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  module->walk([&](Operation *op) {
    if (isa<TokenEnsureOp, TokenFreeOp, RcReleaseOp>(op))
      ops.push_back(op);
  });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRExpandControlFlowPass() {
  return std::make_unique<ReuseIRExpandControlFlowPass>();
}
} // namespace reuse_ir
} // namespace mlir
