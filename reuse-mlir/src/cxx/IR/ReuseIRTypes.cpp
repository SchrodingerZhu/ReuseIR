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
GENERATE_POINTER_ALIKE_LAYOUT(NullableType)
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

void ReuseIRDialect::registerTypes() {
  (void)generatedTypePrinter;
  (void)generatedTypeParser;
  // Register tablegen'd types.
  addTypes<CompositeType,
#define GET_TYPEDEF_LIST
#include "ReuseIR/IR/ReuseIROpsTypes.cpp.inc"
           >();
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

namespace mlir {
namespace reuse_ir {

Type ReuseIRDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  StringRef mnemonic;
  Type genType;

  // Try to parse as a tablegen'd type.
  OptionalParseResult parseResult =
      generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;

  // Type is not tablegen'd: try to parse as a raw C++ type.
  return StringSwitch<function_ref<Type()>>(mnemonic)
      .Case("composite", [&] { return CompositeType::parse(parser); })
      .Default([&] {
        parser.emitError(typeLoc) << "unknown reuse_ir type: " << mnemonic;
        return Type();
      })();
}

void ReuseIRDialect::printType(Type type, DialectAsmPrinter &os) const {
  // Try to print as a tablegen'd type.
  if (generatedTypePrinter(type, os).succeeded())
    return;

  // Type is not tablegen'd: try printing as a raw C++ type.
  TypeSwitch<Type>(type)
      .Case<CompositeType>([&](CompositeType type) {
        os << type.getMnemonic();
        type.print(os);
      })
      .Default([](Type) {
        llvm::report_fatal_error("printer is missing a handler for this type");
      });
}

Type CompositeType::parse(mlir::AsmParser &parser) {
  FailureOr<AsmParser::CyclicParseReset> cyclicParseGuard;
  const auto loc = parser.getCurrentLocation();
  const auto eLoc = parser.getEncodedSourceLoc(loc);
  auto *context = parser.getContext();

  if (parser.parseLess())
    return {};

  mlir::StringAttr name;
  parser.parseOptionalAttribute(name);

  // Is a self reference: ensure referenced type was parsed.
  if (name && parser.parseOptionalGreater().succeeded()) {
    auto type = getChecked(eLoc, context, name);
    if (succeeded(parser.tryStartCyclicParse(type))) {
      parser.emitError(loc, "invalid self-reference within composite type");
      return {};
    }
    return type;
  }

  // Is a named composite definition: ensure name has not been parsed yet.
  if (name) {
    auto type = getChecked(eLoc, context, name);
    cyclicParseGuard = parser.tryStartCyclicParse(type);
    if (failed(cyclicParseGuard)) {
      parser.emitError(loc, "composite already defined");
      return {};
    }
  }

  // Parse record members or lack thereof.
  bool incomplete = true;
  llvm::SmallVector<mlir::Type> members;
  if (parser.parseOptionalKeyword("incomplete").failed()) {
    incomplete = false;
    const auto delimiter = AsmParser::Delimiter::Braces;
    const auto parseElementFn = [&parser, &members]() {
      return parser.parseType(members.emplace_back());
    };
    if (parser.parseCommaSeparatedList(delimiter, parseElementFn).failed())
      return {};
  }

  if (parser.parseGreater())
    return {};

  // Try to create the proper record type.
  ArrayRef<mlir::Type> membersRef(members); // Needed for template deduction.
  mlir::Type type = {};
  if (name && incomplete) { // Identified & incomplete
    type = getChecked(eLoc, context, name);
  } else if (name && !incomplete) { // Identified & complete
    type = getChecked(eLoc, context, membersRef, name);
    // If the record has a self-reference, its type already exists in a
    // incomplete state. In this case, we must complete it.
    if (mlir::cast<CompositeType>(type).isIncomplete())
      mlir::cast<CompositeType>(type).complete(membersRef);
  } else if (!name && !incomplete) { // anonymous & complete
    type = getChecked(eLoc, context, membersRef);
  } else { // anonymous & incomplete
    parser.emitError(loc, "anonymous composite types must be complete");
    return {};
  }

  return type;
}

void CompositeType::print(mlir::AsmPrinter &printer) const {
  FailureOr<AsmPrinter::CyclicPrintReset> cyclicPrintGuard;
  printer << '<';

  if (getName())
    printer << getName();

  // Current type has already been printed: print as self reference.
  cyclicPrintGuard = printer.tryStartCyclicPrint(*this);
  if (failed(cyclicPrintGuard)) {
    printer << '>';
    return;
  }

  // Type not yet printed: continue printing the entire record.
  if (getName())
    printer << ' ';

  if (isIncomplete()) {
    printer << "incomplete";
  } else {
    printer << "{";
    llvm::interleaveComma(getMembers(), printer);
    printer << "}";
  }

  printer << '>';
}

mlir::LogicalResult
CompositeType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                      llvm::ArrayRef<mlir::Type> members, mlir::StringAttr name,
                      bool incomplete) {
  for (auto type : members) {
    if (llvm::isa<RefType>(type))
      return emitError() << "cannot have a reference type in a composite type";
    if (auto rcTy = llvm::dyn_cast<RcType>(type)) {
      if (rcTy.getFreezingKind().getValue() == FreezingKind::unfrozen)
        return emitError() << "cannot have a non-frozen but freezable RC type "
                              "in a composite type, use mref instead";
    }
  }
  if (name && name.getValue().empty()) {
    emitError() << "an identified composite type cannot have an empty name";
    return mlir::failure();
  }
  return mlir::success();
}

CompositeType CompositeType::get(::mlir::MLIRContext *context,
                                 ArrayRef<Type> members, StringAttr name) {
  return Base::get(context, members, name, false);
}

CompositeType CompositeType::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, ArrayRef<Type> members, StringAttr name) {
  if (failed(verify(emitError, members, name, /*incomplete=*/false)))
    return {};
  return Base::getChecked(emitError, context, members, name,
                          /*incomplete=*/false);
}

CompositeType CompositeType::get(::mlir::MLIRContext *context,
                                 StringAttr name) {
  return Base::get(context, /*members=*/ArrayRef<Type>{}, name,
                   /*incomplete=*/true);
}

CompositeType CompositeType::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, StringAttr name) {
  if (failed(verify(emitError, /*members=*/ArrayRef<Type>{}, name,
                    /*incomplete=*/true)))
    return {};
  return Base::getChecked(emitError, context, ArrayRef<Type>{}, name,
                          /*incomplete=*/true);
}

CompositeType CompositeType::get(::mlir::MLIRContext *context,
                                 ArrayRef<Type> members) {
  return Base::get(context, members, StringAttr{}, /*incomplete=*/false);
}

CompositeType CompositeType::getChecked(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::mlir::MLIRContext *context, ArrayRef<Type> members) {
  if (failed(verify(emitError, members, StringAttr{}, /*incomplete=*/false)))
    return {};
  return Base::getChecked(emitError, context, members, StringAttr{},
                          /*incomplete=*/false);
}

::llvm::ArrayRef<mlir::Type> CompositeType::getMembers() const {
  return getImpl()->members;
}

bool CompositeType::isIncomplete() const { return getImpl()->incomplete; }

mlir::StringAttr CompositeType::getName() const { return getImpl()->name; }

bool CompositeType::getIncomplete() const { return getImpl()->incomplete; }

void CompositeType::complete(ArrayRef<Type> members) {
  if (mutate(members).failed())
    llvm_unreachable("failed to complete struct");
}

llvm::TypeSize
CompositeType::getTypeSizeInBits(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getSize() * 8;
}
uint64_t CompositeType::getABIAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}
uint64_t
CompositeType::getPreferredAlignment(const DataLayout &dataLayout,
                                     DataLayoutEntryListRef params) const {
  return getCompositeLayout(dataLayout).getAlignment().value();
}
// CompositeLayoutInterface methods.
::mlir::reuse_ir::CompositeLayout
CompositeType::getCompositeLayout(::mlir::DataLayout layout) const {
  return {layout, getMembers()};
}

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
ClosureType::getCompositeLayout(::mlir::DataLayout layout) const {
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(getContext());
  auto indexTy =
      mlir::IntegerType::get(getContext(), layout.getTypeSizeInBits(ptrTy));
  return {layout, {ptrTy, ptrTy, indexTy}};
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
  converter.addConversion([](NullableType type) -> Type {
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
