#include "ReuseIR/Analysis/AliasAnalysis.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

struct ReuseIRInferUnionTagPass
    : public ReuseIRInferUnionTagBase<ReuseIRInferUnionTagPass> {
  using ReuseIRInferUnionTagBase::ReuseIRInferUnionTagBase;
  void runOnOperation() override final;
};

void ReuseIRInferUnionTagPass::runOnOperation() {
  auto func = getOperation();

  // get the dominance information
  auto &domInfo = getAnalysis<DominanceInfo>();

  mlir::AliasAnalysis aliasAnalysis(getOperation());
  aliasAnalysis.addAnalysisImplementation(::mlir::reuse_ir::AliasAnalysis());

  // Collect operations to be considered by the pass.
  SmallVector<UnionInspectOp> inspectOps;
  SmallVector<RcReleaseOp> releaseOps;
  func->walk([&](Operation *op) {
    if (auto inspect = dyn_cast<UnionInspectOp>(op))
      inspectOps.push_back(inspect);
    else if (auto release = dyn_cast<RcReleaseOp>(op))
      releaseOps.push_back(release);
  });

  // apply the changes
  for (auto release : releaseOps)
    for (auto inspect : inspectOps)
      if (auto borrow = dyn_cast_or_null<RcBorrowOp>(
              inspect.getUnionRef().getDefiningOp()))
        if (aliasAnalysis.alias(release.getRcPtr(), borrow.getObject()) ==
                AliasResult::MustAlias &&
            domInfo.dominates(inspect.getOperation(), release.getOperation()))
          release.setTagAttr(inspect.getIndexAttr());
}
} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRInferUnionTagPass() {
  return std::make_unique<ReuseIRInferUnionTagPass>();
}
} // namespace reuse_ir
} // namespace mlir
