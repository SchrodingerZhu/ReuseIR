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
