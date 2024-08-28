#include "ReuseIR/Analysis/AliasAnalysis.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
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
  void runOnOperation() override final;
};

struct Acquire {
  RcAcquireOp acquire;
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
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reuse_ir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  SmallVector<Release> releaseOps;
  SmallVector<Acquire> acquireOps;
  DenseSet<RcAcquireOp> toErase;
  func->walk([&](Operation *op) {
    if (auto release = dyn_cast<RcReleaseOp>(op))
      releaseOps.emplace_back(
          release, SmallVector<int64_t>(release.getFusedIndices().begin(),
                                        release.getFusedIndices().end()));
    else if (auto acq = dyn_cast<RcAcquireOp>(op))
      if (auto load = dyn_cast<LoadOp>(acq.getRcPtr().getDefiningOp()))
        if (auto proj = dyn_cast<ProjOp>(load.getObject().getDefiningOp()))
          if (auto borrow =
                  dyn_cast<RcBorrowOp>(proj.getObject().getDefiningOp()))
            acquireOps.push_back({acq, borrow.getObject(), proj.getIndex()});
  });
  for (auto &release : releaseOps) {
    auto rc = release.operation.getRcPtr();
    for (auto acq : acquireOps) {
      if ( // avoid repeated fusion
          !toErase.contains(acq.acquire) &&
          // destroy the container of an acquire operation
          aliasAnalysis.alias(rc, acq.borrowSource) == AliasResult::MustAlias &&
          // the acquire operation dominates the destroy
          domInfo.dominates(acq.acquire, release.operation) &&
          // skip already fused operations
          !release.contains(acq.projIndex)) {
        release.fusedIndices.push_back(acq.projIndex);
        toErase.insert(acq.acquire);
      }
    }
  }
  for (auto acq : toErase)
    acq.erase();
  for (auto &rel : releaseOps)
    rel.operation.setFusedIndices(rel.fusedIndices);
}

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRAcquireReleaseFusionPass() {
  return std::make_unique<ReuseIRAcquireReleaseFusionPass>();
}
} // namespace reuse_ir
} // namespace mlir
