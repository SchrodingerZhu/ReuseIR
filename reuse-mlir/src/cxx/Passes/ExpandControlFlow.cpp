#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <memory>
#include <numeric>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

#define GEN_PASS_DEF_REUSEIREXPANDCONTROLFLOW
#include "ReuseIR/Passes.h.inc"

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

template <bool OUTLINE_NESTED>
class RcReleaseExpansionPattern : public OpRewritePattern<RcReleaseOp> {
  void emitCallToOutlinedRelease(RcReleaseOp op,
                                 PatternRewriter &rewriter) const {
    std::string uniqueName;
    llvm::raw_string_ostream os(uniqueName);
    formatMangledNameTo(op.getRcPtr().getType(), os);
    os << "::release";
    SymbolTable symbolTable(op->getParentOfType<ModuleOp>());
    auto func =
        dyn_cast_if_present<func::FuncOp>(symbolTable.lookup(uniqueName));
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op->getParentOfType<func::FuncOp>());
      auto funcTy = FunctionType::get(rewriter.getContext(),
                                      op.getRcPtr().getType(), TypeRange{});
      func = rewriter.create<func::FuncOp>(op->getLoc(), uniqueName, funcTy);
      func.setSymVisibility("private");
      func->setAttr("llvm.linkage", LLVM::LinkageAttr::get(
                                        rewriter.getContext(),
                                        LLVM::linkage::Linkage::LinkonceODR));
      auto *entry = func.addEntryBlock();
      rewriter.setInsertionPointToStart(entry);
      rewriter.create<RcReleaseOp>(op->getLoc(), op.getToken().getType(),
                                   entry->getArgument(0), op.getTagAttr());
      // TODO: the token ought to be cleaned up
      rewriter.create<func::ReturnOp>(op->getLoc());
    }
    rewriter.create<func::CallOp>(op->getLoc(), func, op.getRcPtr());
  }

public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RcReleaseOp op,
                                PatternRewriter &rewriter) const final {
    RcType rcTy = op.getRcPtr().getType();
    // we only need to expand for nonfreezing RC pointers
    if (rcTy.getFreezingKind().getValue() != FreezingKind::nonfreezing)
      return LogicalResult::failure();
    if (op->hasAttr("nested")) {
      if (OUTLINE_NESTED) {
        emitCallToOutlinedRelease(op, rewriter);
        rewriter.eraseOp(op);
        return LogicalResult::success();
      }
      return LogicalResult::failure();
    }
    auto decreaseOp =
        rewriter.create<RcDecreaseOp>(op->getLoc(), op.getRcPtr());
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, decreaseOp,
        [&](OpBuilder &builder, Location loc) {
          auto refTy = RefType::get(getContext(), rcTy.getPointee(),
                                    rcTy.getFreezingKind());
          auto borrowed = builder.create<RcBorrowOp>(loc, refTy, op.getRcPtr());
          builder.create<DestroyOp>(loc, borrowed, op.getTagAttr());
          auto resTy = cast<NullableType>(op.getToken().getType());
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
  bool generateDestroy(PatternRewriter &rewriter, Value target,
                       IntegerAttr tag) const {
    auto refTy = cast<RefType>(target.getType());
    // primitive types
    if (refTy.getPointee().isIntOrIndexOrFloat()) {
      return true;
    }
    // rc pointer, apply RcReleaseOp
    if (auto rcTy = dyn_cast<RcType>(refTy.getPointee())) {
      auto loaded = rewriter.create<LoadOp>(target.getLoc(), rcTy, target);
      if (rcTy.getFreezingKind().getValue() == FreezingKind::nonfreezing) {
        auto rcBoxTy =
            RcBoxType::get(getContext(), rcTy.getPointee(),
                           rcTy.getAtomicKind(), rcTy.getFreezingKind());
        auto size = cache.get(rcBoxTy).getSize();
        auto align = cache.get(rcBoxTy).getAlignment();
        auto tokenTy = TokenType::get(getContext(), align.value(), size);
        auto nullableTy = NullableType::get(getContext(), tokenTy);
        auto release = rewriter.create<RcReleaseOp>(target.getLoc(), nullableTy,
                                                    loaded, nullptr);
        // avoid nested release to be expanded
        release->setAttr("nested", rewriter.getUnitAttr());
      } else
        rewriter.create<RcReleaseOp>(target.getLoc(), Type{}, loaded, nullptr);

      return true;
    }

    // for composite type, iterate through the fields and generate the destroy
    // recursively
    if (auto compositeTy = dyn_cast<CompositeType>(refTy.getPointee())) {
      for (auto [i, ty] : llvm::enumerate(compositeTy.getInnerTypes())) {
        if (ty.isIntOrIndexOrFloat())
          continue;
        auto fieldRefTy =
            RefType::get(getContext(), ty, refTy.getFreezingKind());
        auto field = rewriter.create<ProjOp>(target.getLoc(), fieldRefTy,
                                             target, rewriter.getIndexAttr(i));
        generateDestroy(rewriter, field, {});
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
            builder.create<RcReleaseOp>(loc, Type{}, coerced, nullptr);
            builder.create<scf::YieldOp>(loc);
          });
      return true;
    }

    // for union type, expand it to index_switch
    if (auto unionTy = dyn_cast<UnionType>(refTy.getPointee())) {
      if (tag) {
        auto ty = unionTy.getInnerTypes()[tag.getAPSInt().getZExtValue()];
        auto fieldRef = RefType::get(getContext(), ty, refTy.getFreezingKind());
        auto ref = rewriter.create<UnionInspectOp>(target.getLoc(), fieldRef,
                                                   target, tag);
        generateDestroy(rewriter, ref.getResult(), {});
      }
      auto getTag = rewriter.create<UnionGetTagOp>(target.getLoc(), target);
      llvm::SmallVector<int64_t> cases;
      cases.resize(unionTy.getInnerTypes().size());
      std::iota(cases.begin(), cases.end(), 0);
      auto switchOp = rewriter.create<scf::IndexSwitchOp>(
          target.getLoc(), TypeRange{}, getTag, cases,
          unionTy.getInnerTypes().size());
      for (auto [idx, ty] : llvm::enumerate(unionTy.getInnerTypes())) {
        OpBuilder::InsertionGuard guard(rewriter);
        auto &region = switchOp.getCaseRegions()[idx];
        auto *blk = rewriter.createBlock(&region);
        rewriter.setInsertionPointToStart(blk);
        if (!ty.isIntOrIndexOrFloat()) {
          auto fieldRef =
              RefType::get(getContext(), ty, refTy.getFreezingKind());
          auto ref = rewriter.create<UnionInspectOp>(
              target.getLoc(), fieldRef, target, rewriter.getIndexAttr(idx));
          generateDestroy(rewriter, ref.getResult(), {});
        }
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
    if (generateDestroy(rewriter, op.getObject(), op.getTagAttr())) {
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct ReuseIRExpandControlFlowPass
    : public impl::ReuseIRExpandControlFlowBase<ReuseIRExpandControlFlowPass> {
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
  patterns.add<TokenEnsureExpansionPattern, TokenFreeExpansionPattern>(
      &getContext());

  if (outlineNestedRelease)
    patterns.add<RcReleaseExpansionPattern<true>>(&getContext());
  else
    patterns.add<RcReleaseExpansionPattern<false>>(&getContext());

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
std::unique_ptr<Pass> createReuseIRExpandControlFlowPass(
    const ReuseIRExpandControlFlowOptions &options) {
  return std::make_unique<ReuseIRExpandControlFlowPass>(options);
};
} // namespace reuse_ir
} // namespace mlir
