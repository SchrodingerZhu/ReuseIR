#pragma once

#include "ReuseIR/Common.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

#define GET_TYPEDEF_CLASSES
#include "ReuseIR/IR/ReuseIROpsTypes.h.inc"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
inline void populateTypeConverter(mlir::LLVMTypeConverter &converter) {
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
  // TODO: Add more conversions here.
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
