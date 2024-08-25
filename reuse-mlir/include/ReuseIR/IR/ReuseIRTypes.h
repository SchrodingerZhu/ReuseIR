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
struct CompositeTypeStorage : public TypeStorage {
  struct KeyTy {
    ArrayRef<Type> members;
    StringAttr name;
    bool incomplete;

    KeyTy(ArrayRef<Type> members, StringAttr name, bool incomplete)
        : members(members), name(name), incomplete(incomplete) {}
  };

  ArrayRef<Type> members;
  StringAttr name;
  bool incomplete;

  CompositeTypeStorage(ArrayRef<Type> members, StringAttr name, bool incomplete)
      : members(members), name(name), incomplete(incomplete) {}

  KeyTy getAsKey() const { return KeyTy(members, name, incomplete); }

  bool operator==(const KeyTy &key) const {
    if (name)
      return name == key.name;
    return (members == key.members) && (name == key.name) &&
           (incomplete == key.incomplete);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    if (key.name)
      return llvm::hash_combine(key.name);
    return llvm::hash_combine(key.members, key.incomplete);
  }

  static CompositeTypeStorage *construct(TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    return new (allocator.allocate<CompositeTypeStorage>())
        CompositeTypeStorage(allocator.copyInto(key.members), key.name,
                             key.incomplete);
  }

  /// Mutates the members and attributes an identified struct.
  ///
  /// Once a record is mutated, it is marked as complete, preventing further
  /// mutations. Anonymous structs are always complete and cannot be mutated.
  /// This method does not fail if a mutation of a complete struct does not
  /// change the struct.
  LogicalResult mutate(TypeStorageAllocator &allocator,
                       ArrayRef<Type> members) {
    // Anonymous structs cannot mutate.
    if (!name)
      return failure();

    // Mutation of complete structs are allowed if they change nothing.
    if (!incomplete)
      return mlir::success(this->members == members);

    // Mutate incomplete struct.
    this->members = allocator.copyInto(members);

    incomplete = false;
    return success();
  }
};
} // namespace detail

class CompositeType
    : public Type::TypeBase<CompositeType, Type, detail::CompositeTypeStorage,
                            DataLayoutTypeInterface::Trait,
                            ReuseIRCompositeLayoutInterface::Trait,
                            TypeTrait::IsMutable> {
public:
  using Base::Base;
  using Base::getChecked;

#if LLVM_VERSION_MAJOR < 20
  using Base::verify;
#endif

  // NOLINTNEXTLINE(readability-identifier-naming)
  static constexpr llvm::StringLiteral name = "reuse_ir.composite";

  static CompositeType get(MLIRContext *context, ArrayRef<Type> members,
                           StringAttr name);
  static CompositeType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, ArrayRef<Type> members,
                                  StringAttr name);

  /// Create a identified and incomplete struct type.
  static CompositeType get(MLIRContext *context, StringAttr name);
  static CompositeType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, StringAttr name);

  /// Create a anonymous struct type (always complete).
  static CompositeType get(MLIRContext *context, ArrayRef<Type> members);
  static CompositeType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  MLIRContext *context, ArrayRef<Type> members);

  /// Validate the struct about to be constructed.
  static LogicalResult verify(function_ref<InFlightDiagnostic()> emitError,
                              ArrayRef<Type> members, StringAttr name,
                              bool incomplete);

  // Parse/print methods.
  static constexpr mlir::StringLiteral getMnemonic() { return {"composite"}; }
  static Type parse(AsmParser &odsParser);
  void print(AsmPrinter &odsPrinter) const;

  // Accessors
  ArrayRef<Type> getMembers() const;
  StringAttr getName() const;
  bool getIncomplete() const;

  // Predicates
  bool isComplete() const { return !isIncomplete(); };
  bool isIncomplete() const;

  // Utilities
  size_t getNumElements() const { return getMembers().size(); };

  /// Complete the struct type by mutating its members and attributes.
  void complete(ArrayRef<Type> members);

  /// DataLayoutTypeInterface methods.
  ::llvm::TypeSize getTypeSizeInBits(const DataLayout &dataLayout,
                                     DataLayoutEntryListRef params) const;
  uint64_t getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const;
  uint64_t getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;
  // CompositeLayoutInterface methods.
  ::mlir::reuse_ir::CompositeLayout
  getCompositeLayout(::mlir::DataLayout layout) const;
};
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
