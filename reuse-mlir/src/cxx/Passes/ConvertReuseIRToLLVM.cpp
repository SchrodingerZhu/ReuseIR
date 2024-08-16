#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <optional>

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
        rcPtrTy.getFrozen() ? rewriter.getBoolAttr(true) : mlir::BoolAttr());
    auto boxStruct = llvm::cast<mlir::LLVM::LLVMStructType>(
        typeConverter->convertType(rcBoxTy));
    auto rcTy = boxStruct.getBody()[0];
    auto value = adaptor.getCount() ? *adaptor.getCount() : 1;
    auto amount =
        rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rcTy, value);
    if (rcPtrTy.getFrozen()) {
      llvm::StringRef func =
          rcPtrTy.getAtomic() && rcPtrTy.getAtomic().getValue()
              ? "__reuse_ir_acquire_atomic_freezable"
              : "__reuse_ir_acquire_freezable";
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, func, mlir::ValueRange{}, mlir::ValueRange{adaptor.getRcPtr(), amount});
    } else {
      auto rcField = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          boxStruct, adaptor.getRcPtr(), mlir::LLVM::GEPArg{0});
      if (rcPtrTy.getAtomic() && rcPtrTy.getAtomic().getValue()) {
        rewriter.replaceOpWithNewOp<mlir::LLVM::AtomicRMWOp>(
            op, mlir::LLVM::AtomicBinOp::add, rcField, amount,
            mlir::LLVM::AtomicOrdering::seq_cst);
      } else {
        auto rcVal =
            rewriter.create<mlir::LLVM::LoadOp>(op.getLoc(), rcTy, rcField);
        auto newRcVal =
            rewriter.create<mlir::LLVM::AddOp>(op.getLoc(), rcTy, rcVal, amount)
                .getRes();
        rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, newRcVal, rcField);
      }
    }
    return mlir::success();
  }
};

struct ConvertReuseIRToLLVMPass
    : public ConvertReuseIRToLLVMBase<ConvertReuseIRToLLVMPass> {
  using ConvertReuseIRToLLVMBase::ConvertReuseIRToLLVMBase;
  void runOnOperation() override final;
};

static void emitRuntimeFunctions(mlir::Location loc,
                                 mlir::IntegerType targetIdxTy,
                                 mlir::OpBuilder &builder) {
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_acquire_freezable"),
      builder.getFunctionType(
          {mlir::LLVM::LLVMPointerType::get(builder.getContext()), targetIdxTy},
          {}),
      builder.getStringAttr("private"), nullptr, nullptr);
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_acquire_atomic_freezable"),
      builder.getFunctionType(
          {mlir::LLVM::LLVMPointerType::get(builder.getContext()), targetIdxTy},
          {}),
      builder.getStringAttr("private"), nullptr, nullptr);
}

void ConvertReuseIRToLLVMPass::runOnOperation() {
  auto module = getOperation();
  mlir::DataLayout dataLayout(module);
  {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(&module.getBodyRegion().back());
    auto targetIdxTy = builder.getIntegerType(
        dataLayout.getTypeSizeInBits(builder.getIndexType()));
    emitRuntimeFunctions(module->getLoc(), targetIdxTy, builder);
  }
  mlir::LLVMTypeConverter converter(&getContext());
  populateLLVMTypeConverter(dataLayout, converter);
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<IncOpLowering>(converter, &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::func::FuncDialect>();
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
