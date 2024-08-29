#include "ReuseIR/Analysis/AnticipatedAllocation.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace reuse_ir {
struct TestAnticipatedAllocationPass
    : public PassWrapper<TestAnticipatedAllocationPass,
                         OperationPass<func::FuncOp>> {
  TestAnticipatedAllocationPass() = default;
  StringRef getArgument() const final { return "test-anticipated-allocation"; }
  void runOnOperation() override;
};

void TestAnticipatedAllocationPass::runOnOperation() {
  SymbolTableCollection symbolTable;
  auto module = getOperation()->getParentOfType<ModuleOp>();
  DataLayout dataLayout(module);
  auto config = DataFlowConfig().setInterprocedural(false);
  DataFlowSolver solver(config);
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::reuse_ir::AnticipatedAllocAnalysis>(dataLayout,
                                                            symbolTable);
  solver.load<dataflow::SparseConstantPropagation>();
  if (failed(solver.initializeAndRun(getOperation()))) {
    emitError(getOperation()->getLoc(), "dataflow solver failed");
    return signalPassFailure();
  }
  getOperation().walk([&](Operation *op) {
    const auto *lattice =
        solver.lookupState<dataflow::reuse_ir::AnticipatedAllocLattice>(op);
    if (!lattice)
      return;
    auto &allocations = lattice->getAnticipatedAllocs();
    llvm::SmallVector<Attribute> types;
    std::transform(allocations.begin(), allocations.end(),
                   std::back_inserter(types), TypeAttr::get);
    op->setAttr("anticipated_allocation",
                ArrayAttr::get(op->getContext(), types));
  });
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
std::unique_ptr<Pass> createTestAnticipatedAllocationPass() {
  return std::make_unique<TestAnticipatedAllocationPass>();
}
} // namespace reuse_ir
} // namespace mlir
