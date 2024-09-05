#pragma once
#include "ReuseIR/Analysis/AliasAnalysis.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include <llvm-20/llvm/ADT/DenseSet.h>
#include <memory>

namespace mlir::dataflow {
namespace reuse_ir {
using namespace mlir::reuse_ir;
class TokenHeuristic {
private:
  CompositeLayoutCache &cache;
  mlir::AliasAnalysis &aliasAnalysis;

private:
  long similarity(Value token, RcCreateOp op) const;

  static inline constexpr size_t MIN_ALLOC_STEP_SIZE = 2 * sizeof(void *);
  static inline constexpr size_t MIN_ALLOC_STEP_BITS =
      __builtin_ctz(MIN_ALLOC_STEP_SIZE);
  static inline constexpr size_t INTERMEDIATE_BITS = 2;
  static size_t toExpMand(size_t value);

  bool possiblyInplaceReallocable(size_t alignment, size_t oldSize,
                                  size_t newSize) const;

public:
  TokenHeuristic(CompositeLayoutCache &cache,
                 mlir::AliasAnalysis &aliasAnalysis);

  ssize_t operator()(RcCreateOp op, Value token) const;
};

class ReuseLattice : public AbstractDenseLattice {
  Value reuseToken{};
  llvm::DenseSet<Value> freeToken{};
  llvm::DenseSet<Value> aliveToken{};
  bool notJoined = false;

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  ChangeResult join(const AbstractDenseLattice &rhs) override final;
  void print(llvm::raw_ostream &os) const override final;
  Value getReuseToken() const { return reuseToken; }
  const llvm::DenseSet<Value> &getFreeToken() const { return freeToken; }
  const llvm::DenseSet<Value> &getAliveToken() const { return aliveToken; }
  ChangeResult setNewState(Value reuseToken, llvm::DenseSet<Value> freeToken,
                           llvm::DenseSet<Value> aliveToken) {
    if (reuseToken != this->reuseToken || freeToken != this->freeToken ||
        aliveToken != this->aliveToken) {
      this->reuseToken = std::move(reuseToken);
      this->freeToken = std::move(freeToken);
      this->aliveToken = std::move(aliveToken);
      return ChangeResult::Change;
    }
    return ChangeResult::NoChange;
  }
};

class ReuseAnalysis : public DenseForwardDataFlowAnalysis<ReuseLattice> {
private:
  TokenHeuristic tokenHeuristic;
  DominanceInfo &domInfo;
#if LLVM_VERSION_MAJOR < 20
  using RetType = void;
#else
  using RetType = LogicalResult;
#endif
  void customVisitBlock(Block *block);
  void customVisitRegionBranchOperation(ProgramPoint point,
                                        RegionBranchOpInterface branch,
                                        AbstractDenseLattice *after);

public:
  ReuseAnalysis(DataFlowSolver &solver, CompositeLayoutCache &layoutCache,
                mlir::AliasAnalysis &aliasAnalysis, DominanceInfo &domInfo);

  RetType visitOperation(Operation *op, const ReuseLattice &before,
                         ReuseLattice *after) override final;
  void setToEntryState(ReuseLattice *lattice) override final;
  LogicalResult visit(ProgramPoint point) override final;
};
} // namespace reuse_ir
} // namespace mlir::dataflow
