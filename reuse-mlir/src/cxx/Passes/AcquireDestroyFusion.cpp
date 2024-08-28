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

struct ReuseIRAcquireDestroyFusionPass
    : public ReuseIRAcquireDestroyFusionBase<ReuseIRAcquireDestroyFusionPass> {
  using ReuseIRAcquireDestroyFusionBase::ReuseIRAcquireDestroyFusionBase;
  void runOnOperation() override final;
};

void ReuseIRAcquireDestroyFusionPass::runOnOperation() {
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reuse_ir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  DenseMap<DestroyOp, SmallVector<int64_t>> destroyOps;
  struct AcquireTuple {
    RcAcquireOp acquire;
    mlir::Value projSource;
    size_t projIndex;
  };
  SmallVector<AcquireTuple> acquireOps;
  DenseSet<RcAcquireOp> toErase;
  // TODO:
  func->walk([&](Operation *op) {
    if (auto destroy = dyn_cast<DestroyOp>(op))
      destroyOps.insert(
          {destroy, SmallVector<int64_t>{destroy.getFusedIndices().begin(),
                                         destroy.getFusedIndices().end()}});
    else if (auto acq = dyn_cast<RcAcquireOp>(op))
      if (auto load = dyn_cast<LoadOp>(acq.getRcPtr().getDefiningOp()))
        if (auto proj = dyn_cast<ProjOp>(load.getObject().getDefiningOp()))
          acquireOps.push_back({acq, proj.getObject(), proj.getIndex()});
  });
  for (auto &destroy : destroyOps) {
    auto ref = destroy.getFirst().getObject();
    for (auto acq : acquireOps) {
      if (/* destroy the container of an acquire operation */ aliasAnalysis
                  .alias(ref, acq.projSource) == AliasResult::MustAlias &&
          /* the acquire operation dominates the destroy */
          domInfo.dominates(acq.acquire, destroy.getFirst()) &&
          /* avoid repeated analysis */ !toErase.contains(acq.acquire) &&
          std::find(destroy.getSecond().begin(), destroy.getSecond().end(),
                    acq.projIndex) == destroy.getSecond().end()) {
        destroy.getSecond().push_back(acq.projIndex);
        toErase.insert(acq.acquire);
      }
    }
  }
  for (auto acq : toErase)
    acq.erase();
  for (auto &destroy : destroyOps)
    destroy.getFirst().setFusedIndices(destroy.getSecond());
}

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRAcquireDestroyFusionPass() {
  return std::make_unique<ReuseIRAcquireDestroyFusionPass>();
}
} // namespace reuse_ir
} // namespace mlir
