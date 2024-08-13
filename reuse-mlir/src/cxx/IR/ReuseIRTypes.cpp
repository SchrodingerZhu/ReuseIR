#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "ReuseIR/IR/ReuseIROpsTypes.cpp.inc"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {

::llvm::TypeSize RcType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeSizeInBits(
      mlir::LLVM::LLVMPointerType::get(getContext()));
};
uint64_t RcType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(
      mlir::LLVM::LLVMPointerType::get(getContext()));
}
uint64_t
RcType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                              ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(
      mlir::LLVM::LLVMPointerType::get(getContext()));
}

void ReuseIRDialect::registerTypes() {
  (void)generatedTypePrinter;
  (void)generatedTypeParser;
  // Register tablegen'd types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "ReuseIR/IR/ReuseIROpsTypes.cpp.inc"
      >();
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
