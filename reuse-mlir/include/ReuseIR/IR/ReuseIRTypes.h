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
void populateLLVMTypeConverter(mlir::LLVMTypeConverter &converter);
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
