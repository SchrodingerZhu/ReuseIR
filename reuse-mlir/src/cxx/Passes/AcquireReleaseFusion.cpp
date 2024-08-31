#include "ReuseIR/Analysis/AliasAnalysis.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <memory>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

struct ReuseIRAcquireReleaseFusionPass
    : public ReuseIRAcquireReleaseFusionBase<ReuseIRAcquireReleaseFusionPass> {
  using ReuseIRAcquireReleaseFusionBase::ReuseIRAcquireReleaseFusionBase;
  void compositeAcquireReleaseFusion(IRRewriter &rewriter);
  void unionAcquireReleaseFusion(IRRewriter &rewriter);
  void trivialAcquireReleaseFusion(IRRewriter &rewriter);
  void runOnOperation() override final;
};

struct Acquire {
  RcAcquireOp operation;
  TypedValue<RcType> borrowSource;
  size_t projIndex;
};

struct Release {
  RcReleaseOp operation;
  SmallVector<int64_t> fusedIndices;

  bool contains(size_t index) const {
    return std::find(fusedIndices.begin(), fusedIndices.end(), index) !=
           fusedIndices.end();
  }
};

void ReuseIRAcquireReleaseFusionPass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  compositeAcquireReleaseFusion(rewriter);
  unionAcquireReleaseFusion(rewriter);
  trivialAcquireReleaseFusion(rewriter);
}

void ReuseIRAcquireReleaseFusionPass::trivialAcquireReleaseFusion(
    IRRewriter &rewriter) {
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();
  auto &postDomInfo = getAnalysis<PostDominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reuse_ir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  SmallVector<Release> releaseOps;
  SmallVector<RcAcquireOp> allAcquireOps;
  DenseSet<RcReleaseOp> toRewrite;
  DenseSet<Operation *> toErase;
  func->walk([&](Operation *op) {
    if (auto release = dyn_cast<RcReleaseOp>(op))
      releaseOps.emplace_back(
          release, SmallVector<int64_t>(release.getFusedIndices().begin(),
                                        release.getFusedIndices().end()));
    else if (auto acq = dyn_cast_or_null<RcAcquireOp>(op))
      allAcquireOps.push_back(acq);
  });
  // fuse trivial release after acquire
  for (auto &acq : allAcquireOps) {
    // skip already fused operations
    if (toErase.contains(acq))
      continue;
    for (auto &rel : releaseOps) {
      // skip already fused operations
      if (!rel.fusedIndices.empty() || toErase.contains(rel.operation))
        continue;
      // check the following conditions:
      // - acq and rel are applied to the same object
      // - acq dominates rel
      // - acq post-dominates rel
      // - rel has no other users
      if (aliasAnalysis.alias(acq.getRcPtr(), rel.operation.getRcPtr()) ==
              AliasResult::MustAlias &&
          domInfo.dominates(acq, rel.operation) &&
          postDomInfo.postDominates(rel.operation, acq)) {
        toErase.insert(acq);
        if (!rel.operation.getToken() || rel.operation.getToken().use_empty())
          toErase.insert(rel.operation);
        else
          toRewrite.insert(rel.operation);
      }
    }
  }
  // apply the changes
  for (auto *op : toErase)
    rewriter.eraseOp(op);
  for (auto op : toRewrite) {
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<NullableNullOp>(op, op->getResultTypes());
  }
}

void ReuseIRAcquireReleaseFusionPass::compositeAcquireReleaseFusion(
    IRRewriter &rewriter) {
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();
  auto &postDomInfo = getAnalysis<PostDominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reuse_ir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  SmallVector<Release> releaseOps;
  SmallVector<Acquire> acquireOps;
  DenseSet<Operation *> toErase;
  func->walk([&](Operation *op) {
    if (auto release = dyn_cast<RcReleaseOp>(op))
      releaseOps.emplace_back(
          release, SmallVector<int64_t>(release.getFusedIndices().begin(),
                                        release.getFusedIndices().end()));
    else if (auto acq = dyn_cast_or_null<RcAcquireOp>(op))
      if (auto load = dyn_cast_or_null<LoadOp>(acq.getRcPtr().getDefiningOp()))
        if (auto proj =
                dyn_cast_or_null<ProjOp>(load.getObject().getDefiningOp()))
          if (auto borrow = dyn_cast_or_null<RcBorrowOp>(
                  proj.getObject().getDefiningOp()))
            acquireOps.push_back(
                {acq, borrow.getObject(), proj.getIndex().getZExtValue()});
  });
  for (auto &release : releaseOps) {
    auto rc = release.operation.getRcPtr();
    for (auto acq : acquireOps) {
      if ( // avoid repeated fusion
          !toErase.contains(acq.operation) &&
          // destroy the container of an acquire operation
          aliasAnalysis.alias(rc, acq.borrowSource) == AliasResult::MustAlias &&
          // the acquire operation dominates the release operation and the
          // release operation post-dominates the acquire operation
          domInfo.dominates(acq.operation, release.operation) &&
          postDomInfo.postDominates(release.operation, acq.operation) &&
          // skip already fused operations
          !release.contains(acq.projIndex)) {
        release.fusedIndices.push_back(acq.projIndex);
        toErase.insert(acq.operation);
      }
    }
  }

  // apply the changes
  for (auto *op : toErase)
    rewriter.eraseOp(op);
  for (auto &rel : releaseOps)
    if (!toErase.contains(rel.operation))
      rel.operation.setFusedIndices(rel.fusedIndices);
}

void ReuseIRAcquireReleaseFusionPass::unionAcquireReleaseFusion(
    IRRewriter &rewriter) {
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();
  auto &postDomInfo = getAnalysis<PostDominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reuse_ir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  SmallVector<Release> releaseOps;
  SmallVector<Acquire> acquireOps;
  DenseSet<Operation *> toErase;
  func->walk([&](Operation *op) {
    if (auto release = dyn_cast<RcReleaseOp>(op))
      releaseOps.emplace_back(
          release, SmallVector<int64_t>(release.getFusedIndices().begin(),
                                        release.getFusedIndices().end()));
    else if (auto acq = dyn_cast_or_null<RcAcquireOp>(op))
      if (auto load = dyn_cast_or_null<LoadOp>(acq.getRcPtr().getDefiningOp()))
        if (auto proj =
                dyn_cast_or_null<ProjOp>(load.getObject().getDefiningOp()))
          if (auto inspect = dyn_cast_or_null<UnionInspectOp>(
                  proj.getObject().getDefiningOp()))
            if (auto borrow = dyn_cast_or_null<RcBorrowOp>(
                    inspect.getUnionRef().getDefiningOp()))
              acquireOps.push_back(
                  {acq, borrow.getObject(), proj.getIndex().getZExtValue()});
  });
  for (auto &release : releaseOps) {
    auto rc = release.operation.getRcPtr();
    for (auto acq : acquireOps) {
      if ( // avoid repeated fusion
          !toErase.contains(acq.operation) &&
          // destroy the container of an acquire operation
          aliasAnalysis.alias(rc, acq.borrowSource) == AliasResult::MustAlias &&
          // the acquire operation dominates the release operation and the
          // release operation post-dominates the acquire operation
          domInfo.dominates(acq.operation, release.operation) &&
          postDomInfo.postDominates(release.operation, acq.operation) &&
          // skip already fused operations
          !release.contains(acq.projIndex)) {
        release.fusedIndices.push_back(acq.projIndex);
        toErase.insert(acq.operation);
      }
    }
  }

  // apply the changes
  for (auto *op : toErase)
    rewriter.eraseOp(op);
  for (auto &rel : releaseOps)
    if (!toErase.contains(rel.operation))
      rel.operation.setFusedIndices(rel.fusedIndices);
}
} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRAcquireReleaseFusionPass() {
  return std::make_unique<ReuseIRAcquireReleaseFusionPass>();
}
} // namespace reuse_ir
} // namespace mlir
