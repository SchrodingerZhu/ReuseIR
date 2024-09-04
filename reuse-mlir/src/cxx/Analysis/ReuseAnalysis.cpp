#include "ReuseIR/Analysis/ReuseAnalysis.h"

#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace mlir::dataflow {
namespace reuse_ir {
void ReuseLattice::print(llvm::raw_ostream &os) const {
  {
    os << "{";
    switch (reuseKind) {
    case ReuseKind::NONE:
      os << "none";
      break;
    case ReuseKind::FREE:
      os << "free";
      break;
    case ReuseKind::REUSE:
      os << "reuse";
      break;
    case ReuseKind::JOIN:
      os << "join";
      break;
    }
    os << ": [";
    llvm::interleaveComma(tokenUsed, os);
    os << "], alive: [";
    llvm::interleaveComma(aliveToken, os);
    os << "]}";
  }
}

ChangeResult ReuseLattice::join(const AbstractDenseLattice &rhs) {
  auto &other = static_cast<const ReuseLattice &>(rhs);
  if (reuseKind != ReuseKind::JOIN) {
    reuseKind = ReuseKind::JOIN;
    aliveToken = other.getAliveToken();
    return ChangeResult::Change;
  }
  // intersection
  llvm::SmallVector<Value> toErase;
  for (auto token : aliveToken)
    if (!other.getAliveToken().contains(token))
      toErase.push_back(token);
  for (auto token : toErase)
    aliveToken.erase(token);
  return toErase.empty() ? ChangeResult::NoChange : ChangeResult::Change;
}

void ReuseAnalysis::setToEntryState(ReuseLattice *lattice) {
  propagateIfChanged(lattice, lattice->setAction(ReuseKind::NONE) |
                                  lattice->clearAliveToken() |
                                  lattice->clearTokenUsed());
}

ReuseAnalysis::RetType ReuseAnalysis::visitOperation(Operation *op,
                                                     const ReuseLattice &before,
                                                     ReuseLattice *after) {
  auto changed = ChangeResult::NoChange;

  for (auto alive : before.getAliveToken())
    changed |= after->addAliveTokenIfNoUsed(alive);

  if (auto release = dyn_cast<RcReleaseOp>(op)) {
    if (release.getToken() && release.getToken().use_empty())
      propagateIfChanged(
          after, changed | after->addAliveTokenIfNoUsed(release.getToken()));
    else
      propagateIfChanged(after, changed);
  }

  if (isa<CallOpInterface>(op)) {
    changed |= after->setAction(ReuseKind::FREE);
    for (auto alive : after->getAliveToken())
      changed |= after->addUsedToken(alive);
    propagateIfChanged(after, changed | after->clearAliveToken());
  }

  do {
    if (auto create = dyn_cast<RcCreateOp>(op)) {
      auto rcType = create.getType();
      if (rcType.getFreezingKind().getValue() != FreezingKind::nonfreezing)
        break;
      Value reuseCandidate{};
      long currentHeuristic = -10;
      for (auto alive : after->getAliveToken()) {
        auto heuristic = tokenHeuristic(create, alive);
        if (!reuseCandidate || heuristic > currentHeuristic) {
          reuseCandidate = alive;
          currentHeuristic = heuristic;
        }
      }
      // reusable
      if (currentHeuristic >= -1)
        propagateIfChanged(after, changed | after->setAction(ReuseKind::REUSE) |
                                      after->setUsedToken(reuseCandidate) |
                                      after->eraseAliveToken(reuseCandidate));
      else
        propagateIfChanged(after, changed | after->clearTokenUsed());
    }
  } while (false);

#if LLVM_VERSION_MAJOR < 20
  return;
#else
  return LogicalResult::success();
#endif
}
bool ReuseLattice::operator==(const ReuseLattice &rhs) const {
  return reuseKind == rhs.reuseKind && tokenUsed == rhs.tokenUsed &&
         aliveToken == rhs.aliveToken;
}
ChangeResult ReuseLattice::setAction(ReuseKind kind) {
  auto old = reuseKind;
  reuseKind = kind;
  return old == reuseKind ? ChangeResult::NoChange : ChangeResult::Change;
}
ChangeResult ReuseLattice::clearAliveToken() {
  if (aliveToken.empty())
    return ChangeResult::NoChange;
  aliveToken.clear();
  return ChangeResult::Change;
}
ChangeResult ReuseLattice::eraseAliveToken(Value token) {
  return aliveToken.erase(token) ? ChangeResult::Change
                                 : ChangeResult::NoChange;
}
ChangeResult ReuseLattice::clearTokenUsed() {
  if (tokenUsed.empty())
    return ChangeResult::NoChange;
  tokenUsed.clear();
  return ChangeResult::Change;
}
ChangeResult ReuseLattice::addUsedToken(ValueRange token) {
  bool changed = false;
  for (auto t : token)
    changed |= tokenUsed.insert(t).second;
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}
ChangeResult ReuseLattice::setUsedToken(Value token) {
  if (tokenUsed.empty()) {
    tokenUsed.insert(token);
    return ChangeResult::Change;
  }
  if (tokenUsed.contains(token))
    return ChangeResult::NoChange;
  tokenUsed.clear();
  tokenUsed.insert(token);
  return ChangeResult::Change;
}
ChangeResult ReuseLattice::addAliveTokenIfNoUsed(ValueRange token) {
  bool changed = false;
  for (auto t : token)
    if (!tokenUsed.contains(t))
      changed |= aliveToken.insert(t).second;
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}
ssize_t TokenHeuristic::operator()(RcCreateOp op, Value token) const {
  auto type = op.getType();
  auto nullableTy = cast<NullableType>(token.getType());
  auto tokenTy = cast<TokenType>(nullableTy.getPointer());
  auto rcBoxTy = RcBoxType::get(type.getContext(), type.getPointee(),
                                type.getAtomicKind(), type.getFreezingKind());
  const auto &layout = cache.get(rcBoxTy);
  if (layout.getAlignment() != tokenTy.getAlignment())
    return -2;
  if (layout.getSize() != tokenTy.getSize())
    return possiblyInplaceReallocable(layout.getAlignment().value(),
                                      tokenTy.getSize(), layout.getSize())
               ? -1
               : -2;
  return similarity(token, op);
}
long TokenHeuristic::similarity(Value token, RcCreateOp op) const {
  auto rcReleaseOp = dyn_cast_or_null<RcReleaseOp>(token.getDefiningOp());
  if (!rcReleaseOp)
    return 0;

  auto assembleOp = dyn_cast_or_null<CompositeAssembleOp>(
      rcReleaseOp.getOperand().getDefiningOp());
  if (!assembleOp)
    return 0;
  long score = 0;
  for (auto field : assembleOp.getFields()) {
    auto *defOp = field.getDefiningOp();
    while (defOp && defOp->getNumResults() && defOp->getNumOperands() &&
           !isa<RcType>(defOp->getResultTypes()[0])) {
      if (isa_and_nonnull<LoadOp, ProjOp, UnionInspectOp, RcBorrowOp>(defOp)) {
        defOp = defOp->getOperand(0).getDefiningOp();
        continue;
      }
    }
    if (defOp && defOp->getNumResults() && defOp->getNumOperands() &&
        isa<RcType>(defOp->getResultTypes()[0]) &&
        aliasAnalysis.alias(rcReleaseOp.getRcPtr(), defOp->getResult(0)) ==
            AliasResult::MustAlias)
      score++;
  }
  return score;
}
size_t TokenHeuristic::toExpMand(size_t value) {
  auto oneAtBit = [](size_t bit) { return 1 << bit; };
  constexpr size_t LEADING_BIT =
      oneAtBit(INTERMEDIATE_BITS + MIN_ALLOC_STEP_BITS) >> 1;
  constexpr size_t MANTISSA_MASK = oneAtBit(INTERMEDIATE_BITS) - 1;
  constexpr size_t BITS = sizeof(size_t) * CHAR_BIT;

  value = value - 1;

  size_t e = BITS - INTERMEDIATE_BITS - MIN_ALLOC_STEP_BITS -
             __builtin_clz(value | LEADING_BIT);
  size_t b = (e == 0) ? 0 : 1;
  size_t m = (value >> (MIN_ALLOC_STEP_BITS + e - b)) & MANTISSA_MASK;

  return (e << INTERMEDIATE_BITS) + m;
}
bool TokenHeuristic::possiblyInplaceReallocable(size_t alignment,
                                                size_t oldSize,
                                                size_t newSize) const {
  auto alignedSize = [](size_t alignment, size_t size) {
    return ((alignment - 1) | (size - 1)) + 1;
  };

  constexpr size_t GB = 1024 * 1024 * 1024;
  auto oldAlignedSize = alignedSize(alignment, oldSize);
  auto newAlignedSize = alignedSize(alignment, newSize);
  if (oldAlignedSize >= GB || newAlignedSize >= GB)
    return false;
  auto oldExpMand = toExpMand(oldAlignedSize >> MIN_ALLOC_STEP_BITS);
  auto newExpMand = toExpMand(newAlignedSize >> MIN_ALLOC_STEP_BITS);
  return newExpMand == oldExpMand;
}
ReuseAnalysis::ReuseAnalysis(DataFlowSolver &solver,
                             CompositeLayoutCache &layoutCache,
                             mlir::AliasAnalysis &aliasAnalysis)
    : DenseForwardDataFlowAnalysis(solver),
      tokenHeuristic(layoutCache, aliasAnalysis) {}
TokenHeuristic::TokenHeuristic(CompositeLayoutCache &cache,
                               mlir::AliasAnalysis &aliasAnalysis)
    : cache(cache), aliasAnalysis(aliasAnalysis) {}
} // namespace reuse_ir
} // namespace mlir::dataflow
