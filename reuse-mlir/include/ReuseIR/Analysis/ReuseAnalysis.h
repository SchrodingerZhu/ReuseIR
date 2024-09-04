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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
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

enum class ReuseKind { NONE, FREE, REUSE, JOIN };

class ReuseLattice : public AbstractDenseLattice {
  ReuseKind reuseKind = ReuseKind::NONE;
  llvm::DenseSet<Value> tokenUsed{};
  llvm::DenseSet<Value> aliveToken{};

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  ChangeResult join(const AbstractDenseLattice &rhs) override final;
  void print(llvm::raw_ostream &os) const override final;
  ReuseKind getReuseKind() const { return reuseKind; }
  const llvm::DenseSet<Value> &getTokenUsed() const { return tokenUsed; }
  const llvm::DenseSet<Value> &getAliveToken() const { return aliveToken; }
  bool operator==(const ReuseLattice &rhs) const;
  ChangeResult setAction(ReuseKind kind);
  ChangeResult clearAliveToken();
  ChangeResult eraseAliveToken(Value token);
  ChangeResult clearTokenUsed();
  ChangeResult addUsedToken(ValueRange token);
  ChangeResult setUsedToken(Value token);
  ChangeResult addAliveTokenIfNoUsed(ValueRange token);
};

class ReuseAnalysis : public DenseForwardDataFlowAnalysis<ReuseLattice> {
private:
  TokenHeuristic tokenHeuristic;
#if LLVM_VERSION_MAJOR < 20
  using RetType = void;
#else
  using RetType = LogicalResult;
#endif

public:
  ReuseAnalysis(DataFlowSolver &solver, CompositeLayoutCache &layoutCache,
                mlir::AliasAnalysis &aliasAnalysis);

  RetType visitOperation(Operation *op, const ReuseLattice &before,
                         ReuseLattice *after) override final;
  void setToEntryState(ReuseLattice *lattice) override final;
};
} // namespace reuse_ir
} // namespace mlir::dataflow