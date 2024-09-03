#include "ReuseIR/Analysis/AliasAnalysis.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <memory>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

struct ReuseIRAcquireReleaseFusionPass
    : public ReuseIRAcquireReleaseFusionBase<ReuseIRAcquireReleaseFusionPass> {
  using ReuseIRAcquireReleaseFusionBase::ReuseIRAcquireReleaseFusionBase;
  void runOnOperation() override final {
    // TODO
  };
};
} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRAcquireReleaseFusionPass() {
  return std::make_unique<ReuseIRAcquireReleaseFusionPass>();
}
} // namespace reuse_ir
} // namespace mlir
