#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>

#define GET_TYPEDEF_CLASSES
#include "ReuseIR/IR/ReuseIROpsTypes.cpp.inc"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
#pragma push_macro("GENERATE_POINTER_ALIKE_LAYOUT")
#define GENERATE_POINTER_ALIKE_LAYOUT(TYPE)                                    \
  ::llvm::TypeSize TYPE::getTypeSizeInBits(                                    \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypeSizeInBits(                                       \
        mlir::LLVM::LLVMPointerType::get(getContext()));                       \
  };                                                                           \
  uint64_t TYPE::getABIAlignment(                                              \
      const ::mlir::DataLayout &dataLayout,                                    \
      [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {          \
    return dataLayout.getTypeABIAlignment(                                     \
        mlir::LLVM::LLVMPointerType::get(getContext()));                       \
  }                                                                            \
  uint64_t TYPE::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,   \
                                       ::mlir::DataLayoutEntryListRef params)  \
      const {                                                                  \
    return dataLayout.getTypePreferredAlignment(                               \
        mlir::LLVM::LLVMPointerType::get(getContext()));                       \
  }
GENERATE_POINTER_ALIKE_LAYOUT(RcType)
GENERATE_POINTER_ALIKE_LAYOUT(TokenType)
GENERATE_POINTER_ALIKE_LAYOUT(MRefType)
#pragma pop_macro("GENERATE_POINTER_ALIKE_LAYOUT")

::llvm::TypeSize UnitType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getZero();
};
uint64_t UnitType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return 0;
}
uint64_t
UnitType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                ::mlir::DataLayoutEntryListRef params) const {
  return 0;
}

::llvm::TypeSize RcBoxType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto idxTy = mlir::IndexType::get(getContext());
  llvm::TypeSize ptrSize = dataLayout.getTypeSize(ptrTy);
  llvm::TypeSize idxSize = dataLayout.getTypeSize(idxTy);
  llvm::TypeSize headerSize = std::max(ptrSize, idxSize);
  if (getFreezable())
    headerSize *= 2;
  auto size = llvm::TypeSize::getFixed(
      llvm::alignTo(headerSize, dataLayout.getTypeABIAlignment(getDataType())));
  size += dataLayout.getTypeSize(getDataType());
  size = llvm::TypeSize::getFixed(
      llvm::alignTo(size, getABIAlignment(dataLayout, params)));
  return size * 8;
};
uint64_t RcBoxType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto idxTy = mlir::IndexType::get(getContext());
  return std::max({dataLayout.getTypeABIAlignment(ptrTy),
                   dataLayout.getTypeABIAlignment(idxTy),
                   dataLayout.getTypeABIAlignment(getDataType())});
}
uint64_t
RcBoxType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto idxTy = mlir::IndexType::get(getContext());
  return std::max({dataLayout.getTypePreferredAlignment(ptrTy),
                   dataLayout.getTypePreferredAlignment(idxTy),
                   dataLayout.getTypePreferredAlignment(getDataType())});
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
