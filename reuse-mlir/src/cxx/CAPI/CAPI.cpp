#include "ReuseIR/CAPI.h"
#include "ReuseIR/IR/ReuseIRDialect.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "ReuseIR/IR/ReuseIROpsEnums.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
namespace mlir {
namespace reuse_ir {
extern "C" {
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ReuseIR, reuse_ir, ReuseIRDialect)

// polyfill for pre-20 LLVM
void reuseIRSetLinkageForFunc(MlirOperation op, Linkage linkage) {
  Operation *funcOp = cast<func::FuncOp>(unwrap(op));
  mlir::LLVM::Linkage llvmLinkage;
  switch (linkage) {
  case PRIVATE:
    llvmLinkage = mlir::LLVM::Linkage::Private;
    break;
  case INTERNAL:
    llvmLinkage = mlir::LLVM::Linkage::Internal;
    break;
  case AVAILABLE_EXTERNALLY:
    llvmLinkage = mlir::LLVM::Linkage::AvailableExternally;
    break;
  case LINKONCE:
    llvmLinkage = mlir::LLVM::Linkage::Linkonce;
    break;
  case WEAK:
    llvmLinkage = mlir::LLVM::Linkage::Weak;
    break;
  case COMMON:
    llvmLinkage = mlir::LLVM::Linkage::Common;
    break;
  case APPENDING:
    llvmLinkage = mlir::LLVM::Linkage::Appending;
    break;
  case EXTERN_WEAK:
    llvmLinkage = mlir::LLVM::Linkage::ExternWeak;
    break;
  case LINKONCE_ODR:
    llvmLinkage = mlir::LLVM::Linkage::LinkonceODR;
    break;
  case WEAK_ODR:
    llvmLinkage = mlir::LLVM::Linkage::WeakODR;
    break;
  case EXTERNAL:
    llvmLinkage = mlir::LLVM::Linkage::External;
    break;
  }
  funcOp->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(
                                      funcOp->getContext(), llvmLinkage));
}

MlirAttribute reuseIRFreezingKindGetNonfreezing(MlirContext context) {
  return wrap(
      FreezingKindAttr::get(unwrap(context), FreezingKind::nonfreezing));
}
MlirAttribute reuseIRFreezingKindGetUnfrozen(MlirContext context) {
  return wrap(FreezingKindAttr::get(unwrap(context), FreezingKind::unfrozen));
}
MlirAttribute reuseIRFreezingKindGetFrozen(MlirContext context) {
  return wrap(FreezingKindAttr::get(unwrap(context), FreezingKind::frozen));
}

MlirAttribute reuseIRAtomicKindGetNonatomic(MlirContext context) {
  return wrap(AtomicKindAttr::get(unwrap(context), AtomicKind::nonatomic));
}
MlirAttribute reuseIRAtomicKindGetAtomic(MlirContext context) {
  return wrap(AtomicKindAttr::get(unwrap(context), AtomicKind::atomic));
}
MlirType reuseIRGetRcType(MlirType inner, MlirAttribute atomicKind,
                          MlirAttribute freezeKind) {
  auto innerTy = unwrap(inner);
  return wrap(RcType::get(innerTy.getContext(), unwrap(inner),
                          cast<AtomicKindAttr>(unwrap(atomicKind)),
                          cast<FreezingKindAttr>(unwrap(freezeKind))));
}
} // extern "C"
} // namespace reuse_ir
} // namespace mlir
