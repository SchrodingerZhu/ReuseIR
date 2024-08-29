#include "ReuseIR/Analysis/AnticipatedAllocation.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::dataflow {
namespace reuse_ir {
ChangeResult AnticipatedAllocLattice::meet(const AbstractDenseLattice &rhs) {
  bool changed = false;
  const auto &rhsLattice = static_cast<const AnticipatedAllocLattice &>(rhs);
  for (auto &alloc : rhsLattice.anticipatedAllocs)
    if (anticipatedAllocs.insert(alloc).second)
      changed = true;
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

TokenType AnticipatedAllocAnalysis::getToken(RcType type) {
  auto rcBox = RcBoxType::get(type.getContext(), type.getPointee(),
                              type.getAtomicKind(), type.getFreezingKind());
  auto layout = layoutCache.get(rcBox);
  return TokenType::get(type.getContext(), layout.getAlignment().value(),
                        layout.getSize());
}

LogicalResult
AnticipatedAllocAnalysis::visitOperation(Operation *op,
                                         const AnticipatedAllocLattice &after,
                                         AnticipatedAllocLattice *before) {
  ChangeResult changed = ChangeResult::NoChange;
  if (auto alloc = dyn_cast<RcCreateOp>(op)) {
    if (!alloc.getToken()) {
      auto token = getToken(alloc.getType());
      changed = before->insert(token);
    }
  }
  propagateIfChanged(before, before->meet(after) | changed);
  return LogicalResult::success();
}

void AnticipatedAllocAnalysis::setToExitState(
    AnticipatedAllocLattice *lattice) {
  propagateIfChanged(lattice, lattice->clear());
}
ChangeResult AnticipatedAllocLattice::insert(TokenType token) {
  return anticipatedAllocs.insert(token).second ? ChangeResult::Change
                                                : ChangeResult::NoChange;
}
ChangeResult AnticipatedAllocLattice::clear() {
  auto result =
      anticipatedAllocs.empty() ? ChangeResult::NoChange : ChangeResult::Change;
  anticipatedAllocs.clear();
  return result;
}
} // namespace reuse_ir
} // namespace mlir::dataflow
