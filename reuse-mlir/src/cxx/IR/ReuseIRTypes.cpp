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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cstddef>
#include <numeric>

#define GET_TYPEDEF_CLASSES
#include "ReuseIR/IR/ReuseIROpsTypes.cpp.inc"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {

void populateLLVMTypeConverter(mlir::DataLayout layout,
                               mlir::DataLayoutEntryListRef params,
                               mlir::LLVMTypeConverter &converter) {
  converter.addConversion([](RcType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](TokenType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](MRefType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([](RegionCtxType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
  converter.addConversion([&converter](
                              CompositeType type) -> std::optional<Type> {
    llvm::SmallVector<mlir::Type> fieldTypes;
    if (mlir::failed(converter.convertTypes(type.getMemberTypes(), fieldTypes)))
      return std::nullopt;
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(),
                                                  fieldTypes);
  });
  converter.addConversion([](UnitType type) -> Type {
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), {});
  });
  converter.addConversion([&converter](
                              ClosureType type) -> std::optional<Type> {
    auto convertedIndexType =
        converter.convertType(mlir::IndexType::get(type.getContext()));
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(type.getContext());
    llvm::SmallVector<mlir::Type> inputTypes;
    if (mlir::failed(converter.convertTypes(type.getInputTypes(), inputTypes)))
      return std::nullopt;
    auto inputPackTy =
        mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), inputTypes);
    return mlir::LLVM::LLVMStructType::getLiteral(
        type.getContext(), {ptrTy, ptrTy, ptrTy, convertedIndexType,
                            convertedIndexType, inputPackTy});
  });
  converter.addConversion([&converter](ArrayType type) -> Type {
    auto eltTy = converter.convertType(type.getElementType());
    for (auto size : llvm::reverse(type.getSizes()))
      eltTy = mlir::LLVM::LLVMArrayType::get(eltTy, size);
    return eltTy;
  });
  converter.addConversion([](OpaqueType type) -> Type {
    auto alignment = type.getAlignment().getUInt();
    auto size = type.getSize().getUInt();
    auto cnt = size / alignment;
    auto vTy = mlir::LLVM::LLVMFixedVectorType::get(
        mlir::IntegerType::get(type.getContext(), 8), alignment);
    auto dataArea = mlir::LLVM::LLVMArrayType::get(vTy, cnt);
    return mlir::LLVM::LLVMStructType::getLiteral(type.getContext(), dataArea);
  });
} // namespace REUSE_IR_DECL_SCOPE

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
GENERATE_POINTER_ALIKE_LAYOUT(RegionCtxType)
#pragma pop_macro("GENERATE_POINTER_ALIKE_LAYOUT")

static uint64_t
maxABIAlignmentOfPtrAndIndex(const ::mlir::DataLayout &dataLayout,
                             mlir::MLIRContext *ctx) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  auto idxTy = mlir::IndexType::get(ctx);
  return std::max(dataLayout.getTypeABIAlignment(ptrTy),
                  dataLayout.getTypeABIAlignment(idxTy));
}

static uint64_t
maxPreferredAlignmentOfPtrAndIndex(const ::mlir::DataLayout &dataLayout,
                                   mlir::MLIRContext *ctx) {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
  auto idxTy = mlir::IndexType::get(ctx);
  return std::max(dataLayout.getTypePreferredAlignment(ptrTy),
                  dataLayout.getTypePreferredAlignment(idxTy));
}

// Unit DataLayoutInterface:
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

// RcBox DataLayoutInterface:
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
  return std::max(maxABIAlignmentOfPtrAndIndex(dataLayout, getContext()),
                  dataLayout.getTypeABIAlignment(getDataType()));
}
uint64_t
RcBoxType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  return std::max(maxPreferredAlignmentOfPtrAndIndex(dataLayout, getContext()),
                  dataLayout.getTypePreferredAlignment(getDataType()));
}

// Ref DataLayoutInterface:
::llvm::TypeSize RefType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto idxTy = mlir::IndexType::get(getContext());
  llvm::TypeSize size = dataLayout.getTypeSize(ptrTy);
  llvm::TypeSize idxSize = dataLayout.getTypeSize(idxTy);
  llvm::Align idxAlign{dataLayout.getTypeABIAlignment(idxTy)};
  if (getRank() != 0) {
    size = llvm::TypeSize::getFixed(llvm::alignTo(size, idxAlign));
    size += idxSize * getRank();
    if (getStrided())
      size += idxSize * getRank();
    size = llvm::TypeSize::getFixed(
        llvm::alignTo(size, getABIAlignment(dataLayout, params)));
  }
  return size * 8;
}

uint64_t RefType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return maxABIAlignmentOfPtrAndIndex(dataLayout, getContext());
}

uint64_t
RefType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                               ::mlir::DataLayoutEntryListRef params) const {
  return maxPreferredAlignmentOfPtrAndIndex(dataLayout, getContext());
}

// Array DataLayoutInterface:
::llvm::TypeSize ArrayType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  size_t numOfElems = std::reduce(getSizes().begin(), getSizes().end(), 1,
                                  std::multiplies<size_t>());
  return dataLayout.getTypeSizeInBits(getElementType()) * numOfElems;
}

uint64_t ArrayType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypeABIAlignment(getElementType());
}

uint64_t
ArrayType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  return dataLayout.getTypePreferredAlignment(getElementType());
}

// Vector DataLayoutInterface:
::llvm::TypeSize VectorType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto idxTy = mlir::IndexType::get(getContext());
  llvm::TypeSize size = dataLayout.getTypeSize(ptrTy);
  llvm::TypeSize idxSize = dataLayout.getTypeSize(idxTy);
  llvm::Align idxAlign{dataLayout.getTypeABIAlignment(idxTy)};
  size = llvm::TypeSize::getFixed(llvm::alignTo(size, idxAlign));
  size += idxSize * 2;
  size = llvm::TypeSize::getFixed(
      llvm::alignTo(size, getABIAlignment(dataLayout, params)));
  return size * 8;
}

uint64_t VectorType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return maxABIAlignmentOfPtrAndIndex(dataLayout, getContext());
}

uint64_t
VectorType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return maxPreferredAlignmentOfPtrAndIndex(dataLayout, getContext());
}

// Opaque DataLayoutInterface:
::llvm::TypeSize OpaqueType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return llvm::TypeSize::getFixed(getSize().getUInt());
}

uint64_t OpaqueType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getAlignment().getUInt();
}

uint64_t
OpaqueType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return getAlignment().getUInt();
}

// Token Verifier:
::llvm::LogicalResult
TokenType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                  size_t alignment, size_t size) {
  if (!llvm::isPowerOf2_64(alignment))
    return emitError() << "Alignment must be a power of 2";
  if (size == 0)
    return emitError() << "Size must be non-zero";
  if (size % alignment != 0)
    return emitError() << "Size must be a multiple of alignment";
  return ::llvm::success();
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
