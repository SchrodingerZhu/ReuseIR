#include "ReuseIR/IR/ReuseIRTypes.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRDialect.h"

#include "ReuseIR/IR/ReuseIROpsEnums.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
static LogicalResult parseAtomicKind(AsmParser &parser, AtomicKindAttr &attr) {
  std::string buffer;
  if (parser.parseKeywordOrString(&buffer).succeeded()) {
    if (auto kind = symbolizeAtomicKind(buffer)) {
      attr = AtomicKindAttr::get(parser.getContext(), *kind);
      return LogicalResult::success();
    }
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "failed to parse atomic kind");
}
static LogicalResult parseFreezingKind(AsmParser &parser,
                                       FreezingKindAttr &attr) {
  std::string buffer;
  if (parser.parseKeywordOrString(&buffer).succeeded()) {
    if (auto kind = symbolizeFreezingKind(buffer)) {
      attr = FreezingKindAttr::get(parser.getContext(), *kind);
      return LogicalResult::success();
    }
  }
  return parser.emitError(parser.getCurrentLocation(),
                          "failed to parse freezing kind");
}

static void printAtomicKind(AsmPrinter &printer, const AtomicKindAttr &attr) {
  printer.printKeywordOrString(stringifyAtomicKind(attr.getValue()));
}

static void printFreezingKind(AsmPrinter &printer,
                              const FreezingKindAttr &attr) {
  printer.printKeywordOrString(stringifyFreezingKind(attr.getValue()));
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

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
GENERATE_POINTER_ALIKE_LAYOUT(RefType)
GENERATE_POINTER_ALIKE_LAYOUT(TokenType)
GENERATE_POINTER_ALIKE_LAYOUT(MRefType)
GENERATE_POINTER_ALIKE_LAYOUT(RegionCtxType)
#pragma pop_macro("GENERATE_POINTER_ALIKE_LAYOUT")

// RcBox DataLayoutInterface:
::llvm::TypeSize RcBoxType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
};

uint64_t RcBoxType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}
uint64_t
RcBoxType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Array DataLayoutInterface:
::llvm::TypeSize ArrayType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  uint64_t numOfElems = std::reduce(getSizes().begin(), getSizes().end(), 1,
                                    std::multiplies<uint64_t>());
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
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t VectorType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t
VectorType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Opaque DataLayoutInterface:
::llvm::TypeSize OpaqueType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t OpaqueType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t
OpaqueType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                  ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Closure DataLayoutInterface:
::llvm::TypeSize ClosureType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t ClosureType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t ClosureType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Union DataLayoutInterface:
::llvm::TypeSize UnionType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t UnionType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t
UnionType::getPreferredAlignment(const ::mlir::DataLayout &dataLayout,
                                 ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Composite DataLayoutInterface:
::llvm::TypeSize CompositeType::getTypeSizeInBits(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}

uint64_t CompositeType::getABIAlignment(
    const ::mlir::DataLayout &dataLayout,
    [[maybe_unused]] ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

uint64_t CompositeType::getPreferredAlignment(
    const ::mlir::DataLayout &dataLayout,
    ::mlir::DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}

// Token Verifier:
::mlir::reuse_ir::LogicalResult
TokenType::verify(::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
                  size_t alignment, size_t size) {
  if (!llvm::isPowerOf2_64(alignment))
    return emitError() << "alignment must be a power of 2";
  if (size == 0)
    return emitError() << "size must be non-zero";
  if (size % alignment != 0)
    return emitError() << "size must be a multiple of alignment";
  return ::mlir::reuse_ir::success();
}

// TokenType mangle
void TokenType::formatMangledNameTo(::llvm::raw_string_ostream &buffer) const {
  buffer << "5TokenILm" << getSize() << "Elm" << getAlignment() << "EE";
}

::mlir::reuse_ir::LogicalResult CompositeType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<mlir::Type> memberTypes) {
  for (auto type : memberTypes) {
    if (llvm::isa<RefType>(type))
      return emitError() << "cannot have a reference type in a composite type";
    if (auto rcTy = llvm::dyn_cast<RcType>(type)) {
      if (rcTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
        return emitError() << "cannot have a non-frozen but freezable RC type "
                              "in a composite type, use mref instead";
    }
  }
  return ::mlir::reuse_ir::success();
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

namespace mlir {
namespace reuse_ir {
// ReuseIRCompositeLayoutInterface
::mlir::reuse_ir::CompositeLayout
RcBoxType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(getContext());
  auto ptrEqSize =
      ::mlir::IntegerType::get(getContext(), layout.getTypeSizeInBits(ptrTy));
  return getFreezingKind().getValue() != FreezingKind::nonfreezing
             ? CompositeLayout{layout, {ptrEqSize, ptrTy, ptrTy, getDataType()}}
             : CompositeLayout{layout, {ptrEqSize, getDataType()}};
}
::mlir::reuse_ir::CompositeLayout
VectorType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto idxTy = ::mlir::IndexType::get(getContext());
  auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(getContext());
  return {layout, {ptrTy, idxTy, idxTy}};
}
::mlir::reuse_ir::CompositeLayout
OpaqueType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = ::mlir::LLVM::LLVMPointerType::get(getContext());
  auto size = getSize().getUInt();
  auto align = getAlignment().getUInt();
  auto dataTy = ::mlir::LLVM::LLVMFixedVectorType::get(
      ::mlir::IntegerType::get(getContext(), 8), align);
  auto dataArea = ::mlir::LLVM::LLVMArrayType::get(dataTy, size / align);
  return {layout, {ptrTy, ptrTy, dataArea}};
}
::mlir::reuse_ir::CompositeLayout
UnionType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto tagType = getTagType();
  auto [dataSz, dataAlign] = getDataLayout(layout);
  auto cnt = dataSz.getFixedValue() / dataAlign.value();
  auto vTy = mlir::LLVM::LLVMFixedVectorType::get(
      mlir::IntegerType::get(getContext(), 8), dataAlign.value());
  auto dataArea = mlir::LLVM::LLVMArrayType::get(vTy, cnt);
  return {layout, {tagType, dataArea}};
}

::mlir::reuse_ir::CompositeLayout
CompositeType::getCompositeLayout(::mlir::DataLayout layout) const {
  return {layout, getMemberTypes()};
}

::mlir::reuse_ir::CompositeLayout
ClosureType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  return {layout, {ptrTy, ptrTy, ptrTy}};
}

void populateLLVMTypeConverter(CompositeLayoutCache &cache,
                               mlir::LLVMTypeConverter &converter) {
  converter.addConversion([](RefType type) -> Type {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  });
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
  converter.addConversion(
      [&converter, &cache](ReuseIRCompositeLayoutInterface type) -> Type {
        return cache.get(type).getLLVMType(converter);
      });
  converter.addConversion([&converter](ArrayType type) -> Type {
    auto eltTy = converter.convertType(type.getElementType());
    for (auto size : llvm::reverse(type.getSizes()))
      eltTy = mlir::LLVM::LLVMArrayType::get(eltTy, size);
    return eltTy;
  });
}

} // namespace reuse_ir
} // namespace mlir
