#pragma once

#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROpsEnums.h"
#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h"
#include "ReuseIR/Interfaces/ReuseIRMangleInterface.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <variant>

#define GET_TYPEDEF_CLASSES
#include "ReuseIR/IR/ReuseIROpsTypes.h.inc"

namespace mlir {
namespace reuse_ir {
void populateLLVMTypeConverter(CompositeLayoutCache &cache,
                               mlir::LLVMTypeConverter &converter);

namespace detail {
// forward declaration
template <typename Impl> struct ContainerLikeTypeStorage;
struct CompositeTypeImpl;
struct UnionTypeImpl;
} // namespace detail

template <typename Impl>
class ContainerLikeType
    : public Type::TypeBase<
          ContainerLikeType<Impl>, Type, detail::ContainerLikeTypeStorage<Impl>,
          DataLayoutTypeInterface::Trait,
          ReuseIRCompositeLayoutInterface::Trait, TypeTrait::IsMutable> {
public:
  using Base = Type::TypeBase<
      ContainerLikeType<Impl>, Type, detail::ContainerLikeTypeStorage<Impl>,
      DataLayoutTypeInterface::Trait, ReuseIRCompositeLayoutInterface::Trait,
      TypeTrait::IsMutable>;
  using Base::Base;
  using Base::getChecked;

#if LLVM_VERSION_MAJOR < 20
  using Base::verify;
#else
  using Base::verifyInvariants;
#endif
  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr mlir::StringLiteral name = Impl::name;
  static constexpr mlir::StringLiteral getMnemonic() {
    return Impl::getMnemonic();
  }

  static ContainerLikeType get(MLIRContext *context, ArrayRef<Type> innerTypes,
                               StringAttr name);
  static ContainerLikeType
  getChecked(function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
             ArrayRef<Type> innerTypes, StringAttr name);

  /// Create a identified and incomplete struct type.
  static ContainerLikeType get(MLIRContext *context, StringAttr name);
  static ContainerLikeType
  getChecked(function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
             StringAttr name);

  /// Create a anonymous struct type (always complete).
  static ContainerLikeType get(MLIRContext *context, ArrayRef<Type> innerTypes);
  static ContainerLikeType
  getChecked(function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
             ArrayRef<Type> innerTypes);

  // Parse/print methods.
  static Type parse(AsmParser &odsParser);
  void print(AsmPrinter &odsPrinter) const;

  // Accessors
  ArrayRef<Type> getInnerTypes() const;
  StringAttr getName() const;
  bool getIncomplete() const;

  // Predicates
  bool isComplete() const { return !isIncomplete(); };
  bool isIncomplete() const;

  // Utilities
  size_t getNumElements() const { return getInnerTypes().size(); };

  /// Complete the struct type by mutating its innerTypes and attributes.
  void complete(ArrayRef<Type> innerTypes);

  /// DataLayoutTypeInterface methods.
  ::llvm::TypeSize getTypeSizeInBits(const DataLayout &dataLayout,
                                     DataLayoutEntryListRef params) const {
    return getCompositeLayout(dataLayout).getSize() * 8;
  }
  uint64_t getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const {
    return getCompositeLayout(dataLayout).getAlignment().value();
  }
  uint64_t getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const {
    return getCompositeLayout(dataLayout).getAlignment().value();
  }
  // CompositeLayoutInterface methods.
  ::mlir::reuse_ir::CompositeLayout
  getCompositeLayout(::mlir::DataLayout layout) const {
    return Impl::getCompositeLayout(Base::getContext(), layout,
                                    getInnerTypes());
  }

  /// Validate the struct about to be constructed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Type> members, StringAttr name,
                              bool incomplete) {
    return Impl::verify(emitError, members, name, incomplete);
  }
  static LogicalResult
  verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Type> members, StringAttr name, bool incomplete) {
    return verify(emitError, members, name, incomplete);
  }
};

using CompositeType = ContainerLikeType<detail::CompositeTypeImpl>;
using UnionType = ContainerLikeType<detail::UnionTypeImpl>;
#include "ReuseIRTypeDetails.h.inc"

} // namespace reuse_ir
} // namespace mlir

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
inline bool isProjectable(mlir::Type type) {
  return llvm::TypeSwitch<mlir::Type, bool>(type)
      .Case<CompositeType>([](auto &&) { return true; })
      .Case<ArrayType>([](auto &&) { return true; })
      .Default([](auto &&) { return false; });
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
