#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/Passes.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {
struct ConvertReuseIRToLLVMPass
    : public ConvertReuseIRToLLVMBase<ConvertReuseIRToLLVMPass> {
  using ConvertReuseIRToLLVMBase::ConvertReuseIRToLLVMBase;
  void runOnOperation() override;
};

void ConvertReuseIRToLLVMPass::runOnOperation() { llvm_unreachable("TODO"); }

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createConvertReuseIRToLLVMPass() {
  return std::make_unique<ConvertReuseIRToLLVMPass>();
}
} // namespace reuse_ir
} // namespace mlir
