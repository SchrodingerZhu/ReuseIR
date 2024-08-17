#pragma once

#include "ReuseIR/Common.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/TypeSize.h"
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <variant>

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
class CompositeLayout {
public:
  using FieldKind = std::variant<mlir::Type, size_t>;
  struct Field {
    size_t index;
    llvm::Align alignment;
  };

private:
  llvm::Align alignment = llvm::Align{1};
  llvm::TypeSize size = llvm::TypeSize::getZero();
  llvm::SmallVector<FieldKind> raw_fields;
  llvm::DenseMap<size_t, Field> field_map;

public:
  const llvm::Align &getAlignment() const { return alignment; }
  const llvm::TypeSize &getSize() const { return size; }
  llvm::ArrayRef<FieldKind> getRawFields() const { return raw_fields; }
  Field getField(size_t idx) const { return field_map.at(idx); }

  CompositeLayout(mlir::DataLayout layout, llvm::ArrayRef<mlir::Type> fields)
      : alignment(1), size(0, false), raw_fields{}, field_map{} {
    for (auto [index, type] : llvm::enumerate(fields)) {
      llvm::TypeSize typeSz = layout.getTypeSize(type);
      // skip zero sized element
      if (typeSz.isZero())
        continue;
      size_t typeAlign = layout.getTypeABIAlignment(type);
      alignment = std::max(alignment, llvm::Align(typeAlign));
      llvm::TypeSize alignedSize = llvm::alignTo(size, typeAlign);
      if (alignedSize > size)
        raw_fields.emplace_back(alignedSize - size);
      field_map.insert({index, {raw_fields.size(), llvm::Align{typeAlign}}});
      raw_fields.emplace_back(type);
      size = alignedSize + typeSz;
    }
    llvm::TypeSize alignedSize = llvm::alignTo(size, alignment.value());
    if (alignedSize > size)
      raw_fields.emplace_back(alignedSize - size);
    size = alignedSize;
  }

  mlir::LLVM::LLVMStructType
  getLLVMType(mlir::LLVMTypeConverter &converter) const {
    llvm::SmallVector<mlir::Type> convertedTypes;
    std::transform(
        raw_fields.begin(), raw_fields.end(),
        std::back_inserter(convertedTypes),
        [&](const FieldKind &kind) -> mlir::Type {
          return std::visit(
              [&](auto &&variant) -> mlir::Type {
                using T = std::decay_t<decltype(variant)>;
                if constexpr (std::is_same_v<T, mlir::Type>)
                  return converter.convertType(variant);
                if constexpr (std::is_same_v<T, size_t>)
                  return mlir::LLVM::LLVMArrayType::get(
                      mlir::IntegerType::get(&converter.getContext(), 8),
                      variant);
                llvm_unreachable("conversion logic error");
              },
              kind);
        });
    return mlir::LLVM::LLVMStructType::getLiteral(&converter.getContext(),
                                                  convertedTypes);
  }
};
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "ReuseIR/IR/ReuseIROpsTypes.h.inc"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
inline void populateLLVMTypeConverter(mlir::DataLayout layout,
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
  converter.addConversion([&converter, layout](CompositeType type) -> Type {
    CompositeLayout targetLayout{layout, type.getMemberTypes()};
    return targetLayout.getLLVMType(converter);
  });
  converter.addConversion([](ClosureType type) -> Type {
    auto ptrTy = mlir::LLVM::LLVMPointerType::get(type.getContext());
    return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                            {ptrTy, ptrTy, ptrTy});
  });
  converter.addConversion([&converter](ArrayType type) -> Type {
    auto eltTy = converter.convertType(type.getElementType());
    for (auto size : llvm::reverse(type.getSizes()))
      eltTy = mlir::LLVM::LLVMArrayType::get(eltTy, size);
    return eltTy;
  });
  converter.addConversion([&converter, layout](OpaqueType type) -> Type {
    return type.getCompositeLayout(layout).getLLVMType(converter);
  });
  converter.addConversion([&converter, layout](UnionType type) -> Type {
    return type.getCompositeLayout(layout).getLLVMType(converter);
  });
  converter.addConversion([&converter, layout](VectorType type) -> Type {
    return type.getCompositeLayout(layout).getLLVMType(converter);
  });
  converter.addConversion([&converter, layout](RcBoxType type) -> Type {
    return type.getCompositeLayout(layout).getLLVMType(converter);
  });
  converter.addConversion([&converter, layout](RefType type) -> Type {
    return type.getCompositeLayout(layout).getLLVMType(converter);
  });
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
