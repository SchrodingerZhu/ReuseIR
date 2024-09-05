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
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include <memory>

namespace mlir {
namespace reuse_ir {

struct ReuseIRPrintReuseAnalysisPass
    : public ReuseIRPrintReuseAnalysisBase<ReuseIRPrintReuseAnalysisPass> {
  using ReuseIRPrintReuseAnalysisBase::ReuseIRPrintReuseAnalysisBase;
  void runOnOperation() override final;
};

void ReuseIRPrintReuseAnalysisPass::runOnOperation() {
  auto module = getOperation();
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
  getOperation().walk([&](Operation *op) {
    const auto *lattice =
        solver.lookupState<dataflow::reuse_ir::ReuseLattice>(op);
    if (!lattice)
      return;
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    lattice->print(os);
    llvm::SmallVector<Attribute> types;
    op->setAttr("reuse-analysis", StringAttr::get(op->getContext(), buffer));
  });
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::unique_ptr<Pass> createReuseIRPrintReuseAnalysisPass() {
  return std::make_unique<ReuseIRPrintReuseAnalysisPass>();
}
} // namespace reuse_ir
} // namespace mlir
