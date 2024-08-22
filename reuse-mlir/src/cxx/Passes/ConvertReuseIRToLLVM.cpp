#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIROpsEnums.h"
#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "ReuseIR/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <numeric>
#include <optional>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

template <typename Op>
class ReuseIRConvPatternWithLayoutCache : public mlir::OpConversionPattern<Op> {
protected:
  CompositeLayoutCache &cache;

  const LLVMTypeConverter &getLLVMTypeConverter() const {
    return static_cast<const LLVMTypeConverter &>(*this->typeConverter);
  }

public:
  template <typename... Args>
  ReuseIRConvPatternWithLayoutCache(CompositeLayoutCache &cache, Args &&...args)
      : mlir::OpConversionPattern<Op>(std::forward<Args>(args)...),
        cache(cache) {}
};

class ClosureVTableOpLowering
    : public ReuseIRConvPatternWithLayoutCache<ClosureVTableOp> {
public:
  using ReuseIRConvPatternWithLayoutCache::ReuseIRConvPatternWithLayoutCache;

  mlir::reuse_ir::LogicalResult matchAndRewrite(
      ClosureVTableOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto vtableTy =
        LLVM::LLVMStructType::getLiteral(getContext(), {ptrTy, ptrTy, ptrTy});
    auto glbOp = rewriter.create<LLVM::GlobalOp>(
        op->getLoc(), vtableTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        op.getName(), /*value=*/nullptr,
        /*alignment=*/cache.getDataLayout().getTypeABIAlignment(vtableTy),
        /*addrSpace=*/0, /*dsoLocal=*/true, /*threadLocal=*/false);
    Block *block = rewriter.createBlock(&glbOp.getInitializerRegion());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      mlir::Value value =
          rewriter.create<LLVM::UndefOp>(op->getLoc(), vtableTy);
      auto symbols = llvm::SmallVector<llvm::StringRef, 3>{
          adaptor.getFunc(), adaptor.getClone(), adaptor.getDrop()};
      auto enums = llvm::enumerate(symbols);
      value = std::accumulate(enums.begin(), enums.end(), value,
                              [&](mlir::Value value, auto pair) {
                                return rewriter.create<LLVM::InsertValueOp>(
                                    op->getLoc(), vtableTy, value,
                                    rewriter.create<LLVM::AddressOfOp>(
                                        op->getLoc(), ptrTy, pair.value()),
                                    pair.index());
                              });
      rewriter.create<LLVM::ReturnOp>(op->getLoc(), value);
    }
    rewriter.replaceOp(op, glbOp);
    return mlir::reuse_ir::success();
  }
};

class ClosureNewOpLowering
    : public ReuseIRConvPatternWithLayoutCache<ClosureNewOp> {
public:
  using ReuseIRConvPatternWithLayoutCache::ReuseIRConvPatternWithLayoutCache;

  mlir::reuse_ir::LogicalResult matchAndRewrite(
      ClosureNewOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ty = typeConverter->convertType(op.getType());
    // TODO: lower this properly
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, ty);
    return mlir::reuse_ir::success();
  }
};

class ProjOpLowering : public ReuseIRConvPatternWithLayoutCache<ProjOp> {
public:
  using ReuseIRConvPatternWithLayoutCache::ReuseIRConvPatternWithLayoutCache;

  mlir::reuse_ir::LogicalResult matchAndRewrite(
      ProjOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    if (auto pointee =
            dyn_cast<CompositeType>(op.getObject().getType().getPointee())) {
      const CompositeLayout &layout = cache.get(pointee);
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, ptrTy, layout.getLLVMType(getLLVMTypeConverter()),
          adaptor.getObject(),
          llvm::ArrayRef<LLVM::GEPArg>{0,
                                       layout.getField(op.getIndex()).index});
      return LogicalResult::success();
    }
    if (auto pointee =
            dyn_cast<ArrayType>(op.getObject().getType().getPointee())) {
      mlir::Type innerTy;
      if (pointee.getSizes().size() == 0)
        innerTy = pointee.getElementType();
      else
        innerTy = ArrayType::get(getContext(), pointee.getElementType(),
                                 pointee.getSizes().drop_front());
      auto ty = typeConverter->convertType(innerTy);
      if (!ty)
        return LogicalResult::failure();
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
          op, ptrTy, ty, adaptor.getObject(),
          llvm::ArrayRef<LLVM::GEPArg>{op.getIndex()});
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

class ValueToRefOpLowering
    : public ReuseIRConvPatternWithLayoutCache<ValueToRefOp> {
public:
  using ReuseIRConvPatternWithLayoutCache::ReuseIRConvPatternWithLayoutCache;

  mlir::reuse_ir::LogicalResult matchAndRewrite(
      ValueToRefOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    uint64_t alignment = cache.getDataLayout().getTypePreferredAlignment(
        op.getValue().getType());
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    auto alloca = rewriter.create<mlir::LLVM::AllocaOp>(
        op->getLoc(), ptrTy, adaptor.getValue().getType(),
        rewriter.create<mlir::LLVM::ConstantOp>(
            op->getLoc(), getLLVMTypeConverter().getIndexType(), 1),
        alignment);
    rewriter.create<mlir::LLVM::StoreOp>(op->getLoc(), adaptor.getValue(),
                                         alloca, alignment);
    rewriter.replaceOp(op, alloca);
    return mlir::reuse_ir::success();
  }
};

class BorrowOpLowering : public ReuseIRConvPatternWithLayoutCache<BorrowOp> {
  static inline constexpr size_t NONFREEZING_DATA_OFFSET = 1;
  static inline constexpr size_t FREEZABLE_DATA_OFFSET = 3;

public:
  using ReuseIRConvPatternWithLayoutCache::ReuseIRConvPatternWithLayoutCache;

  mlir::reuse_ir::LogicalResult matchAndRewrite(
      BorrowOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    RcType rcTy = op.getObject().getType();
    RcBoxType box =
        RcBoxType::get(getContext(), rcTy.getPointee(), rcTy.getAtomicKind(),
                       rcTy.getFreezingKind());
    const CompositeLayout &layout = cache.get(box);
    mlir::LLVM::LLVMStructType structTy =
        layout.getLLVMType(getLLVMTypeConverter());
    CompositeLayout::Field targetField = layout.getField(
        rcTy.getFreezingKind().getValue() == FreezingKind::nonfreezing
            ? NONFREEZING_DATA_OFFSET
            : FREEZABLE_DATA_OFFSET);

    // GEP to targetField.offset.
    auto ptrTy = LLVM::LLVMPointerType::get(getContext());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, ptrTy, structTy, adaptor.getObject(),
        llvm::ArrayRef<LLVM::GEPArg>{0, targetField.index});
    return mlir::reuse_ir::success();
  }
};

class LoadOpLowering : public ReuseIRConvPatternWithLayoutCache<LoadOp> {
public:
  using ReuseIRConvPatternWithLayoutCache::ReuseIRConvPatternWithLayoutCache;
  mlir::reuse_ir::LogicalResult matchAndRewrite(
      LoadOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, typeConverter->convertType(op.getType()), adaptor.getObject(),
        cache.getDataLayout().getTypeABIAlignment(op.getType()));
    return mlir::reuse_ir::success();
  }
};

class AllocOpLowering : public mlir::OpConversionPattern<AllocOp> {
public:
  using OpConversionPattern<AllocOp>::OpConversionPattern;
  mlir::reuse_ir::LogicalResult matchAndRewrite(
      AllocOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    TokenType tokenTy = op.getToken().getType();
    const auto *cvt = static_cast<const LLVMTypeConverter *>(typeConverter);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), cvt->getIndexType(), tokenTy.getSize());
    auto alignment = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), cvt->getIndexType(), tokenTy.getAlignment());
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, "__reuse_ir_alloc", mlir::LLVM::LLVMPointerType::get(getContext()),
        mlir::ValueRange{size, alignment});
    return mlir::reuse_ir::success();
  }
};

class FreeOpLowering : public mlir::OpConversionPattern<FreeOp> {
public:
  using OpConversionPattern<FreeOp>::OpConversionPattern;
  mlir::reuse_ir::LogicalResult matchAndRewrite(
      FreeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    TokenType tokenTy = op.getToken().getType();
    const auto *cvt = static_cast<const LLVMTypeConverter *>(typeConverter);
    auto size = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), cvt->getIndexType(), tokenTy.getSize());
    auto alignment = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), cvt->getIndexType(), tokenTy.getAlignment());
    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        op, "__reuse_ir_dealloc", mlir::ValueRange{},
        mlir::ValueRange{adaptor.getToken(), size, alignment});
    return mlir::reuse_ir::success();
  }
};

class IncOpLowering : public mlir::OpConversionPattern<IncOp> {
public:
  using OpConversionPattern<IncOp>::OpConversionPattern;

  mlir::reuse_ir::LogicalResult matchAndRewrite(
      IncOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override final {
    RcType rcPtrTy = op.getRcPtr().getType();
    mlir::reuse_ir::RcBoxType rcBoxTy =
        RcBoxType::get(getContext(), rcPtrTy.getPointee(),
                       rcPtrTy.getAtomicKind(), rcPtrTy.getFreezingKind());
    auto boxStruct = llvm::cast<mlir::LLVM::LLVMStructType>(
        typeConverter->convertType(rcBoxTy));
    auto rcTy = boxStruct.getBody()[0];
    auto value = adaptor.getCount() ? *adaptor.getCount() : 1;
    auto amount =
        rewriter.create<mlir::LLVM::ConstantOp>(op.getLoc(), rcTy, value);
    if (rcPtrTy.getFreezingKind().getValue() != FreezingKind::nonfreezing) {
      llvm::StringRef func =
          rcPtrTy.getAtomicKind().getValue() == AtomicKind::atomic
              ? "__reuse_ir_acquire_atomic_freezable"
              : "__reuse_ir_acquire_freezable";
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
          op, func, mlir::ValueRange{},
          mlir::ValueRange{adaptor.getRcPtr(), amount});
    } else {
      auto rcField = rewriter.create<mlir::LLVM::GEPOp>(
          op.getLoc(), mlir::LLVM::LLVMPointerType::get(getContext()),
          boxStruct, adaptor.getRcPtr(), mlir::LLVM::GEPArg{0});
      if (rcPtrTy.getAtomicKind().getValue() == AtomicKind::atomic) {
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
    return mlir::reuse_ir::success();
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
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_acquire_freezable"),
      builder.getFunctionType({ptrTy, targetIdxTy}, {}),
      builder.getStringAttr("private"), nullptr, nullptr);
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_acquire_atomic_freezable"),
      builder.getFunctionType({ptrTy, targetIdxTy}, {}),
      builder.getStringAttr("private"), nullptr, nullptr);
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_alloc"),
      builder.getFunctionType({targetIdxTy, targetIdxTy}, {ptrTy}),
      builder.getStringAttr("private"), nullptr, nullptr);
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_dealloc"),
      builder.getFunctionType({ptrTy, targetIdxTy, targetIdxTy}, {}),
      builder.getStringAttr("private"), nullptr, nullptr);
  builder.create<mlir::func::FuncOp>(
      loc, builder.getStringAttr("__reuse_ir_realloc"),
      builder.getFunctionType({ptrTy, targetIdxTy, targetIdxTy, targetIdxTy},
                              {ptrTy}),
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
  CompositeLayoutCache cache(dataLayout);
  populateLLVMTypeConverter(cache, converter);
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
  patterns.add<IncOpLowering, AllocOpLowering, FreeOpLowering>(converter,
                                                               &getContext());
  patterns.add<BorrowOpLowering, ValueToRefOpLowering, ProjOpLowering,
               LoadOpLowering, ClosureVTableOpLowering, ClosureNewOpLowering>(
      cache, converter, &getContext());
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addIllegalDialect<mlir::reuse_ir::ReuseIRDialect>();
  llvm::SmallVector<mlir::Operation *> ops;
  module.walk([&](mlir::Operation *op) { ops.push_back(op); });
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
