#pragma once

#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIRDialect.h"
#include "ReuseIR/IR/ReuseIRTypes.h"

#pragma GCC visibility push(hidden)
#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROps.h.inc"
#pragma GCC visibility pop

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir
