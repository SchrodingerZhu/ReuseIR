#pragma once
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::dataflow {
namespace reuse_ir {
using namespace mlir::reuse_ir;
class AnticipatedAllocLattice : public AbstractDenseLattice {
  llvm::DenseSet<TokenType> anticipatedAllocs{};

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  ChangeResult meet(const AbstractDenseLattice &rhs) override final;
  ChangeResult insert(TokenType token);
  ChangeResult clear();
  void print(llvm::raw_ostream &os) const override final {
    os << "{";
    llvm::interleaveComma(anticipatedAllocs, os);
    os << "}";
  }
  const llvm::DenseSet<TokenType> &getAnticipatedAllocs() const {
    return anticipatedAllocs;
  }
};
class AnticipatedAllocAnalysis
    : public DenseBackwardDataFlowAnalysis<AnticipatedAllocLattice> {
private:
  CompositeLayoutCache layoutCache;
  TokenType getToken(RcType type);

public:
  template <typename... Args>
  AnticipatedAllocAnalysis(DataFlowSolver &solver, DataLayout &dataLayout,
                           SymbolTableCollection &symbolTable)
      : DenseBackwardDataFlowAnalysis<AnticipatedAllocLattice>(solver,
                                                               symbolTable),
        layoutCache(dataLayout) {}

  LogicalResult visitOperation(Operation *op,
                               const AnticipatedAllocLattice &after,
                               AnticipatedAllocLattice *before) override final;
  void setToExitState(AnticipatedAllocLattice *lattice) override final;
};
} // namespace reuse_ir
} // namespace mlir::dataflow
