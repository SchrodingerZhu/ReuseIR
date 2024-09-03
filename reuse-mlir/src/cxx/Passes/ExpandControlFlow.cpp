#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <memory>
#include <numeric>

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
    if (op->hasAttr("nested"))
      return LogicalResult::failure();
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

class DestroyExpansionPattern : public OpRewritePattern<DestroyOp> {
  CompositeLayoutCache &cache;
  bool generateDestroy(PatternRewriter &rewriter, Value target) const {
    auto refTy = cast<RefType>(target.getType());
    // primitive types
    if (refTy.getPointee().isIntOrIndexOrFloat()) {
      return true;
    }
    // rc pointer, apply RcReleaseOp
    if (auto rcTy = dyn_cast<RcType>(refTy.getPointee())) {
      auto rcBoxTy =
          RcBoxType::get(getContext(), rcTy.getPointee(), rcTy.getAtomicKind(),
                         rcTy.getFreezingKind());
      auto size = cache.get(rcBoxTy).getSize();
      auto align = cache.get(rcBoxTy).getAlignment();
      auto tokenTy = TokenType::get(getContext(), align.value(), size);
      auto nullableTy = NullableType::get(getContext(), tokenTy);
      auto loaded = rewriter.create<LoadOp>(target.getLoc(), rcTy, target);
      auto release = rewriter.create<RcReleaseOp>(
          target.getLoc(), nullableTy, loaded, nullptr,
          rewriter.getDenseI64ArrayAttr({}));
      // avoid nested release to be expanded
      release->setAttr("nested", rewriter.getUnitAttr());
      return true;
    }

    // for composite type, iterate through the fields and generate the destroy
    // recursively
    if (auto compositeTy = dyn_cast<CompositeType>(refTy.getPointee())) {
      for (auto [i, ty] : llvm::enumerate(compositeTy.getInnerTypes())) {
        auto fieldRefTy =
            RefType::get(getContext(), ty, refTy.getFreezingKind());
        auto field = rewriter.create<ProjOp>(target.getLoc(), fieldRefTy,
                                             target, rewriter.getIndexAttr(i));
        generateDestroy(rewriter, field);
      }
      return true;
    }

    // for mref type, insert release if it is loaded into a nonnull rc pointer
    if (auto mrefTy = dyn_cast<MRefType>(refTy.getPointee())) {
      auto rcTy = RcType::get(getContext(), mrefTy.getPointee(),
                              mrefTy.getAtomicKind(), refTy.getFreezingKind());
      auto nullableTy = NullableType::get(getContext(), rcTy);
      auto loaded =
          rewriter.create<LoadOp>(target.getLoc(), nullableTy, target);
      auto exists = rewriter.create<NullableCheckOp>(target.getLoc(), loaded);
      rewriter.create<scf::IfOp>(
          target.getLoc(), exists, [&](OpBuilder &builder, Location loc) {
            auto coerced = builder.create<NullableCoerceOp>(loc, rcTy, loaded);
            // freezeable pointer has no token
            builder.create<RcReleaseOp>(loc, Type{}, coerced, nullptr,
                                        rewriter.getDenseI64ArrayAttr({}));
            builder.create<scf::YieldOp>(loc);
          });
      return true;
    }

    // for union type, expand it to index_switch
    if (auto unionTy = dyn_cast<UnionType>(refTy.getPointee())) {
      auto tag = rewriter.create<UnionGetTagOp>(target.getLoc(), target);
      llvm::SmallVector<int64_t> cases;
      cases.resize(unionTy.getInnerTypes().size());
      std::iota(cases.begin(), cases.end(), 0);
      auto switchOp = rewriter.create<scf::IndexSwitchOp>(
          target.getLoc(), TypeRange{}, tag, cases,
          unionTy.getInnerTypes().size());
      for (auto [idx, ty] : llvm::enumerate(unionTy.getInnerTypes())) {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = switchOp.getCaseRegions()[idx];
        auto *blk = rewriter.createBlock(&region);
        rewriter.setInsertionPointToStart(blk);
        auto fieldRef = RefType::get(getContext(), ty, refTy.getFreezingKind());
        auto ref = rewriter.create<UnionInspectOp>(
            target.getLoc(), fieldRef, target, rewriter.getIndexAttr(idx));
        generateDestroy(rewriter, ref.getResult());
        rewriter.create<scf::YieldOp>(target.getLoc());
      }
      // insert unreachable for the default case
      {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = switchOp.getDefaultRegion();
        auto *blk = rewriter.createBlock(&region);
        rewriter.setInsertionPointToStart(blk);
        rewriter.create<UnreachableOp>(target.getLoc(), Type{});
        rewriter.create<scf::YieldOp>(target.getLoc());
      }
      return true;
    }

    // for other types: closure, opaque, etc. left a destroy placeholder to be
    // lowered in LLVM lowering pass.
    return false;
  }

public:
  using OpRewritePattern::OpRewritePattern;

  template <typename... Args>
  DestroyExpansionPattern(CompositeLayoutCache &cache, Args &&...args)
      : OpRewritePattern(std::forward<Args>(args)...), cache(cache) {}

  LogicalResult matchAndRewrite(DestroyOp op,
                                PatternRewriter &rewriter) const final {
    if (generateDestroy(rewriter, op.getObject())) {
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct ReuseIRExpandControlFlowPass
    : public ReuseIRExpandControlFlowBase<ReuseIRExpandControlFlowPass> {
  using ReuseIRExpandControlFlowBase::ReuseIRExpandControlFlowBase;
  void runOnOperation() override final;
};

void ReuseIRExpandControlFlowPass::runOnOperation() {
  auto module = getOperation();
  DataLayout dataLayout(module);
  CompositeLayoutCache cache(dataLayout);
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  patterns.add<DestroyExpansionPattern>(cache, &getContext());
  patterns.add<TokenEnsureExpansionPattern, TokenFreeExpansionPattern,
               RcReleaseExpansionPattern>(&getContext());

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;

  // // Apply patterns.
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRExpandControlFlowPass() {
  return std::make_unique<ReuseIRExpandControlFlowPass>();
}
} // namespace reuse_ir
} // namespace mlir
