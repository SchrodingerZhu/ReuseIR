#include "ReuseIR/Analysis/AnticipatedAllocation.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::dataflow {
namespace reuse_ir {
ChangeResult AnticipatedAllocLattice::meet(const AbstractDenseLattice &rhs) {
  ChangeResult changed = ChangeResult::NoChange;
  const auto &rhsLattice = static_cast<const AnticipatedAllocLattice &>(rhs);
  for (auto &alloc : rhsLattice.anticipatedAllocs) {
    auto iter = anticipatedAllocs.find(alloc.getFirst());
    if (iter != anticipatedAllocs.end()) {
      for (auto &op : alloc.getSecond())
        changed = changed | (iter->getSecond().insert(op).second
                                 ? ChangeResult::Change
                                 : ChangeResult::NoChange);
    } else {
      anticipatedAllocs.insert(alloc);
      changed = ChangeResult::Change;
    }
  }
  return changed;
}

TokenType AnticipatedAllocAnalysis::getToken(RcType type) {
  auto rcBox = RcBoxType::get(type.getContext(), type.getPointee(),
                              type.getAtomicKind(), type.getFreezingKind());
  auto layout = layoutCache.get(rcBox);
  return TokenType::get(type.getContext(), layout.getAlignment().value(),
                        layout.getSize());
}

AnticipatedAllocAnalysis::RetType
AnticipatedAllocAnalysis::visitOperation(Operation *op,
                                         const AnticipatedAllocLattice &after,
                                         AnticipatedAllocLattice *before) {
  ChangeResult changed = ChangeResult::NoChange;
  if (auto alloc = dyn_cast<RcCreateOp>(op)) {
    if (!alloc.getToken()) {
      auto token = getToken(alloc.getType());
      changed = before->insert(token, op);
    }
  }
  propagateIfChanged(before, before->meet(after) | changed);
#if LLVM_VERSION_MAJOR < 20
  return;
#else
  return LogicalResult::success();
#endif
}

void AnticipatedAllocAnalysis::setToExitState(
    AnticipatedAllocLattice *lattice) {
  propagateIfChanged(lattice, lattice->clear());
}
ChangeResult AnticipatedAllocLattice::insert(TokenType token, Operation *op) {
  auto iter = anticipatedAllocs.find(token);
  if (iter != anticipatedAllocs.end())
    return iter->getSecond().insert(op).second ? ChangeResult::Change
                                               : ChangeResult::NoChange;
  anticipatedAllocs.insert({token, {op}});
  return ChangeResult::Change;
}
ChangeResult AnticipatedAllocLattice::clear() {
  auto result =
      anticipatedAllocs.empty() ? ChangeResult::NoChange : ChangeResult::Change;
  anticipatedAllocs.clear();
  return result;
}
} // namespace reuse_ir
} // namespace mlir::dataflow
