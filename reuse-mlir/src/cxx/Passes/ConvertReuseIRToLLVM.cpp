#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/ErrorHandling.h"
#include <llvm-20/llvm/Support/raw_ostream.h>
#include <memory>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

class IncOpLowering : public mlir::OpConversionPattern<IncOp> {
public:
  using OpConversionPattern<IncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(IncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    RcType rcPtrTy = op.getRcPtr().getType();
    mlir::reuse_ir::RcBoxType rcBoxTy = RcBoxType::get(
        getContext(), rcPtrTy.getPointee(), rcPtrTy.getAtomic(),
        rcPtrTy.getFrozen() ? mlir::BoolAttr::get(getContext(), true)
                            : mlir::BoolAttr());
    auto boxStruct = llvm::cast<mlir::LLVM::LLVMStructType>(
        typeConverter->convertType(rcBoxTy));
    auto rcField = rewriter.create<mlir::LLVM::GEPOp>(
        op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()), boxStruct,
        adaptor.getRcPtr(), mlir::LLVM::GEPArg{0});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConvertReuseIRToLLVMPass
    : public ConvertReuseIRToLLVMBase<ConvertReuseIRToLLVMPass> {
  using ConvertReuseIRToLLVMBase::ConvertReuseIRToLLVMBase;
  void runOnOperation() override final;
};

void ConvertReuseIRToLLVMPass::runOnOperation() {
  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  mlir::LLVMTypeConverter converter(&getContext());
  populateLLVMTypeConverter(dataLayout, converter);
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<IncOpLowering>(converter, &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalDialect<mlir::reuse_ir::ReuseIRDialect>();
  llvm::SmallVector<mlir::Operation *> ops;
  module.walk([&](mlir::func::FuncOp op) {
    op->walk([&](mlir::Operation *op) { ops.push_back(op); });
  });
  if (failed(applyPartialConversion(ops, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createConvertReuseIRToLLVMPass() {
  return std::make_unique<ConvertReuseIRToLLVMPass>();
}
} // namespace reuse_ir
} // namespace mlir
