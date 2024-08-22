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
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <memory>
#include <optional>

namespace mlir {

namespace REUSE_IR_DECL_SCOPE {

struct ReuseIRClosureOutliningPattern : public OpRewritePattern<ClosureNewOp> {
  llvm::DenseMap<llvm::StringRef, size_t> &assignedLambda;
  DataLayout dataLayout;
  template <typename... Args>
  ReuseIRClosureOutliningPattern(
      llvm::DenseMap<llvm::StringRef, size_t> &assignedLambda,
      DataLayout dataLayout, Args &&...args)
      : OpRewritePattern(std::forward<Args>(args)...),
        assignedLambda(assignedLambda), dataLayout(dataLayout) {}

  LogicalResult matchAndRewrite(ClosureNewOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getVtable())
      return failure();
    auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp());
    if (!funcOp)
      return failure();
    auto moduleOp = dyn_cast<ModuleOp>(funcOp->getParentOp());
    if (!moduleOp)
      return failure();
    llvm::SmallVector<mlir::Type> argTypes;
    std::transform(op->getRegion(0).args_begin(), op->getRegion(0).args_end(),
                   std::back_inserter(argTypes),
                   [](BlockArgument arg) { return arg.getType(); });
    auto argPackTy = CompositeType::get(getContext(), argTypes);
    auto nonfreezing =
        FreezingKindAttr::get(getContext(), FreezingKind::nonfreezing);
    auto argPackRefTy = RefType::get(getContext(), argPackTy, nonfreezing);
    // block number from the function
    auto lambdaNumber = assignedLambda[funcOp.getName()]++;
    std::string lambdaName =
        (funcOp.getName() + llvm::Twine("$$lambda") + llvm::Twine(lambdaNumber))
            .str();
    std::string lambdaFuncName = (lambdaName + llvm::Twine("$$func")).str();
    std::string lambdaCloneName = (lambdaName + llvm::Twine("$$clone")).str();
    std::string lambdaDropName = (lambdaName + llvm::Twine("$$drop")).str();
    std::string lambdaVtableName = (lambdaName + llvm::Twine("$$vtable")).str();
    auto funcTy = FunctionType::get(getContext(), argPackRefTy,
                                    op.getClosureType().getOutputType());
    auto cloneTy = FunctionType::get(getContext(), argPackRefTy, argPackRefTy);
    auto dropTy = FunctionType::get(getContext(), argPackRefTy, {});
    rewriter.setInsertionPoint(funcOp);
    auto lambdaFuncOp =
        rewriter.create<func::FuncOp>(op->getLoc(), lambdaFuncName, funcTy);
    auto lambdaCloneOp =
        rewriter.create<func::FuncOp>(op->getLoc(), lambdaCloneName, cloneTy);
    auto lambdaDropOp =
        rewriter.create<func::FuncOp>(op->getLoc(), lambdaDropName, dropTy);
    // TODO: for now, we only declare the function, we will emit them later
    lambdaFuncOp.setPrivate();
    lambdaCloneOp.setPrivate();
    lambdaDropOp.setPrivate();
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      auto funcOpBlock = rewriter.createBlock(&lambdaFuncOp.getFunctionBody());
      auto arg = funcOpBlock->addArgument(argPackRefTy, op.getLoc());
      rewriter.setInsertionPointToStart(funcOpBlock);
      llvm::SmallVector<Value> args;
      std::transform(
          op->getRegion(0).args_begin(), op->getRegion(0).args_end(),
          std::back_inserter(args), [&](const mlir::BlockArgument &innerArg) {
            mlir::Value ref =
                rewriter
                    .create<ProjOp>(op->getLoc(),
                                    RefType::get(getContext(),
                                                 innerArg.getType(),
                                                 nonfreezing),
                                    arg, innerArg.getArgNumber())
                    ->getOpResult(0);
            mlir::Value loaded =
                rewriter.create<LoadOp>(op->getLoc(), innerArg.getType(), ref);
            return loaded;
          });
      rewriter.inlineBlockBefore(&op->getRegion(0).front(), funcOpBlock,
                                 funcOpBlock->end(), args);
      funcOpBlock->walk([&](ClosureYieldOp yieldOp) {
        rewriter.replaceOpWithNewOp<func::ReturnOp>(yieldOp,
                                                    yieldOp.getOperands());
      });
      rewriter.inlineRegionBefore(op->getRegion(0),
                                  lambdaFuncOp.getFunctionBody(),
                                  lambdaFuncOp.getFunctionBody().end());
    }
    rewriter.create<ClosureVTableOp>(op->getLoc(), lambdaVtableName,
                                     op.getClosureType(), lambdaFuncName,
                                     lambdaCloneName, lambdaDropName);
    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<ClosureNewOp>(
        op, op.getType(), SymbolRefAttr::get(getContext(), lambdaVtableName),
        0);
    return LogicalResult::success();
  };
};

struct ReuseIRClosureOutliningPass
    : public ReuseIRClosureOutliningBase<ReuseIRClosureOutliningPass> {
  using ReuseIRClosureOutliningBase::ReuseIRClosureOutliningBase;
  void runOnOperation() override final;
};

void ReuseIRClosureOutliningPass::runOnOperation() {
  auto module = getOperation();
  DataLayout dataLayout(module);
  // Collect rewrite patterns.
  RewritePatternSet patterns(&getContext());
  llvm::DenseMap<llvm::StringRef, size_t> assignedLambda;
  patterns.add<ReuseIRClosureOutliningPattern>(assignedLambda, dataLayout,
                                               &getContext());

  // Collect operations to be considered by the pass.
  SmallVector<Operation *, 16> ops;
  getOperation()->walk([&](ClosureNewOp op) {
    if (!op.getVtable())
      ops.push_back(op);
  });

  // Configure rewrite to ignore new ops created during the pass.
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingOps;

  // Apply patterns.
  if (failed(applyOpPatternsAndFold(ops, std::move(patterns), config)))
    signalPassFailure();
}

} // namespace REUSE_IR_DECL_SCOPE

namespace reuse_ir {
std::unique_ptr<Pass> createReuseIRClosureOutliningPass() {
  return std::make_unique<ReuseIRClosureOutliningPass>();
}
} // namespace reuse_ir
} // namespace mlir
