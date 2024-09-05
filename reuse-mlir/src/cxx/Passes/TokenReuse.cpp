#include "ReuseIR/Analysis/ReuseAnalysis.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "ReuseIR/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <memory>

namespace mlir {
namespace reuse_ir {

struct ReuseIRTokenReusePass
    : public ReuseIRTokenReuseBase<ReuseIRTokenReusePass> {
  using ReuseIRTokenReuseBase::ReuseIRTokenReuseBase;
  void runOnOperation() override final;
};

void ReuseIRTokenReusePass::runOnOperation() {
  auto module = getOperation()->getParentOfType<ModuleOp>();
  DataLayout dataLayout(module);
  CompositeLayoutCache cache(dataLayout);
  mlir::AliasAnalysis aliasAnalysis(module);
  DominanceInfo dominanceInfo(module);
  aliasAnalysis.addAnalysisImplementation<mlir::reuse_ir::AliasAnalysis>({});
  auto config = DataFlowConfig().setInterprocedural(false);
  DataFlowSolver solver(config);
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::reuse_ir::ReuseAnalysis>(cache, aliasAnalysis,
                                                 dominanceInfo);
  solver.load<dataflow::SparseConstantPropagation>();
  if (failed(solver.initializeAndRun(getOperation()))) {
    emitError(getOperation()->getLoc(), "dataflow solver failed");
    return signalPassFailure();
  }
  // first walk all operations:
  // - if an operation reuses a token, add it to its token slot.
  // - if an operation frees tokens, insert token free operations.
  IRRewriter rewriter(&getContext());
  getOperation().walk([&](Operation *op) {
    // TODO
  });
  // Walk all blocks, if a token alive at the end of the block but it is
  // not alive at the beginning of a successor block, insert token free at the
  // beginning successor block. it the token does not dominate the successor
  // block, insert an intermediate block to free the token.
  getOperation().walk(
      [&](Block *block) { llvm::errs() << "block: " << *block << "\n"; });
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::unique_ptr<Pass> createReuseIRTokenReusePass() {
  return std::make_unique<ReuseIRTokenReusePass>();
}
} // namespace reuse_ir
} // namespace mlir
