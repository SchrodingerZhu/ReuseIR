#pragma once

#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRDialect.h"
#include "ReuseIR/IR/ReuseIRTypes.h"

#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.h.inc"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
