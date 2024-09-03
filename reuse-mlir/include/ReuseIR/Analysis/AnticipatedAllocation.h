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
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir::dataflow {
namespace reuse_ir {
using namespace mlir::reuse_ir;
class AnticipatedAllocLattice : public AbstractDenseLattice {
  llvm::DenseMap<TokenType, llvm::DenseSet<Operation *>> anticipatedAllocs{};

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  ChangeResult meet(const AbstractDenseLattice &rhs) override final;
  ChangeResult insert(TokenType token, Operation *operation);
  ChangeResult clear();
  void print(llvm::raw_ostream &os) const override final {
    os << "{";
    llvm::interleaveComma(anticipatedAllocs, os,
                          [](auto keyVal) { return keyVal.getFirst(); });
    os << "}";
  }
  const llvm::DenseMap<TokenType, llvm::DenseSet<Operation *>> &
  getAnticipatedAllocs() const {
    return anticipatedAllocs;
  }
};
class AnticipatedAllocAnalysis
    : public DenseBackwardDataFlowAnalysis<AnticipatedAllocLattice> {
private:
  CompositeLayoutCache layoutCache;
  TokenType getToken(RcType type);
#if LLVM_VERSION_MAJOR < 20
  using RetType = void;
#else
  using RetType = LogicalResult;
#endif

public:
  template <typename... Args>
  AnticipatedAllocAnalysis(DataFlowSolver &solver, DataLayout &dataLayout,
                           SymbolTableCollection &symbolTable)
      : DenseBackwardDataFlowAnalysis<AnticipatedAllocLattice>(solver,
                                                               symbolTable),
        layoutCache(dataLayout) {}

  RetType visitOperation(Operation *op, const AnticipatedAllocLattice &after,
                         AnticipatedAllocLattice *before) override final;
  void setToExitState(AnticipatedAllocLattice *lattice) override final;
};
} // namespace reuse_ir
} // namespace mlir::dataflow